"""Diff from stair_model.py:
- RoPE replaces the pair-sum input embedding (split-half convention, same as rope_model.py).
  Input is plain wte(idx); position enters via Q/K rotation inside attention.
- Infinite context: caches store K UNROTATED; Q and K are rotated at attention time
  using chunk-relative positions (live token i -> i, cache slabs -> negative positions,
  everything shifted by +W so table indices are non-negative). RoPE scores depend only
  on position DIFFERENCES, so the origin is arbitrary; positions span [-W, W/2), so one
  fixed table of 2*block_size rows serves arbitrarily long streams — no absolute
  position tracking, no table regrowth.
- Standard next-token alignment: logits[i] predicts idx[i+1] (no prev_tokens carry).
- Stair routing unchanged: every query sees recent W/2 from K[l] and older W/2 from
  K[l+1] (W = window = block_size). Requires chunk_size <= block_size/2 (asserted).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


@dataclass
class GPTConfig:
    block_size: int = 1024 # also the attention window size W; expected chunk size at training is W/2
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12 #nh
    n_embd: int = 768
    rope_base: float = 10000.0 # RoPE frequency base (GPT-NeoX / Llama convention)


def _precompute_rope(head_dim, max_seq_len, base):
    """Build cos/sin tables for split-half RoPE, shape (max_seq_len, head_dim) each.
    Same convention as rope_model.py."""
    assert head_dim % 2 == 0
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)              # (max_seq_len, half)
    freqs_full = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
    return freqs_full.cos(), freqs_full.sin()


def _rotate_half(x):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x, cos, sin):
    """x: (B, nh, T, head_dim). cos, sin: (T, head_dim) — broadcast over B, nh."""
    return x * cos + _rotate_half(x) * sin


class StairAttention(nn.Module):
    """Causal attention with strict per-query W/2 + W/2 stair routing (W = window size).
    Recent W/2 positions use K[l] (this layer); older W/2 use K[l+1] (one layer deeper).
    Same routing as stair_model.StairAttention; position is injected via RoPE with
    chunk-relative indices instead of pair-sum embeddings."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        assert config.block_size % 2 == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.window_size = config.block_size
        self._warmup_block_mask = None
        self._steady_block_mask = None

    def precompute_block_masks(self, device):
        W = self.window_size
        half_W = W // 2
        T_new = half_W
        T_low = half_W
        for T_high_total, attr in [(half_W, '_warmup_block_mask'),
                                   (W, '_steady_block_mask')]:
            kv_len = T_high_total + T_low + T_new
            def mask_mod(b, h, q_idx, kv_idx, _tht=T_high_total):
                is_high = kv_idx < _tht
                high_dist = q_idx - (kv_idx - _tht)
                low_dist = q_idx - (kv_idx - _tht - T_low)
                high_ok = (high_dist >= half_W) & (high_dist < W)
                low_ok = (low_dist >= 0) & (low_dist < half_W)
                return torch.where(is_high, high_ok, low_ok)
            setattr(self, attr, create_block_mask(
                mask_mod, B=None, H=None,
                Q_LEN=T_new, KV_LEN=kv_len,
                device=device,
            ))

    def forward(self, x, cos, sin, cache=None):
        """
        x: (B, T_new, C) — current chunk; assume T_new = W/2 in training (asserted ≤ in GPT).
        cos, sin: full (2*block_size, head_dim) RoPE tables.
        cache: optional (low_new, high_old, high_new) — three raw chunk K/V tensors, same
               routing as stair_model (see its docstring). All K entries are UNROTATED;
               rotation is applied fresh each forward with chunk-relative positions,
               which is what makes the context unbounded.
        Returns:
          y: (B, T_new, C)
          (k, v): this chunk's UNROTATED K[l], V[l] — caller routes into the next chunk's cache.
        """
        B, T_new, C = x.size()
        head_size = C // self.n_heads
        W = self.window_size

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T_new, self.n_heads, head_size).transpose(1, 2) # (B, nh, T_new, hd)
        k = k.view(B, T_new, self.n_heads, head_size).transpose(1, 2)
        v = v.view(B, T_new, self.n_heads, head_size).transpose(1, 2)

        # Queries always sit at chunk-relative positions [0, T_new); +W shift keeps
        # table indices non-negative for the cache slabs below.
        q_rot = _apply_rope(q, cos[W:W + T_new], sin[W:W + T_new])

        if cache is None:
            k_rot = _apply_rope(k, cos[W:W + T_new], sin[W:W + T_new])
            y = F.scaled_dot_product_attention(q_rot, k_rot, v, is_causal=True)
            return self._merge_heads(y, B, T_new, C), (k, v)

        low_new, high_old, high_new = cache
        K_low, V_low = low_new
        K_high_new, V_high_new = high_new
        T_low = K_low.size(2)

        if high_old is None:
            K_combined = torch.cat([K_high_new, K_low, k], dim=2)
            V_combined = torch.cat([V_high_new, V_low, v], dim=2)
        else:
            K_high_old, V_high_old = high_old
            K_combined = torch.cat([K_high_old, K_high_new, K_low, k], dim=2)
            V_combined = torch.cat([V_high_old, V_high_new, V_low, v], dim=2)
        T_high_total = K_combined.size(2) - T_low - T_new

        pos = torch.cat([
            torch.arange(W - T_high_total, W, device=q.device),
            torch.arange(W - T_low, W + T_new, device=q.device),
        ])
        k_rot = _apply_rope(K_combined, cos[pos], sin[pos])

        block_mask = self._warmup_block_mask if high_old is None else self._steady_block_mask
        y = flex_attention(q_rot, k_rot, V_combined, block_mask=block_mask)
        return self._merge_heads(y, B, T_new, C), (k, v)

    def _merge_heads(self, y, B, T_new, C):
        y = y.transpose(1, 2).contiguous().view(B, T_new, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = StairAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin, cache=None):
        attn_out, new_kv = self.attn(self.ln_1(x), cos, sin, cache=cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv


class KVOnly(nn.Module):
    """K/V-only projection above the top block. Produces K[L], V[L] from the final
    block's output, used as 'one layer deeper' K for the deepest block's next-chunk
    high_cache. No Q/SDPA/output/residual — just LN + KV projection."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.n_embd)
        self.c_kv = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        head_size = C // self.n_heads
        kv = self.c_kv(self.ln(x))
        k, v = kv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_heads, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, head_size).transpose(1, 2)
        return k, v


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.kv_only = KVOnly(config) # supplies K[L] for layer L-1's high_cache
        # weight sharing
        self.wte.weight = self.lm_head.weight

        # RoPE tables sized 2*block_size: chunk-relative positions span [-W, W/2),
        # shifted by +W into [0, 3W/2). Non-persistent (recomputable from config).
        head_dim = config.n_embd // config.n_heads
        cos, sin = _precompute_rope(head_dim, 2 * config.block_size, config.rope_base)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def precompute_block_masks(self, device):
        for block in self.h:
            block.attn.precompute_block_masks(device)

    def forward(self, idx, kv_caches=None):
        """
        idx: (B, T_new) — this chunk's new token ids. T_new must be <= block_size/2.
        kv_caches: list of n_layers entries, each a tuple (low_new, high_old, high_new),
                   same structure and shift rule as stair_model (see its docstring).
                   K entries are UNROTATED. Pass None (whole list) for the first chunk of
                   a shard; reset to None at each optimizer step boundary.
        Returns:
          logits: (B, T_new, vocab_size) — logits[i] predicts idx[i+1]
          new_kv_caches: list of n_layers (low_new, high_old, high_new) for the next call.
        """
        B, T_new = idx.size()
        W = self.config.block_size
        assert T_new <= W // 2, f"chunk size {T_new} exceeds block_size/2 = {W//2}; strict stair invariant breaks."

        x = self.wte(idx)

        computed_kv = []
        for layer_idx, block in enumerate(self.h):
            layer_cache = None if kv_caches is None else kv_caches[layer_idx]
            x, (k_new, v_new) = block(x, self.rope_cos, self.rope_sin, cache=layer_cache)
            computed_kv.append((k_new, v_new))

        k_top, v_top = self.kv_only(x)

        x_out = self.ln_f(x)
        logits = self.lm_head(x_out)

        new_kv_caches = []
        n_layers = len(self.h)
        for layer_idx in range(n_layers):
            k_l, v_l = computed_kv[layer_idx]
            if layer_idx + 1 < n_layers:
                k_above, v_above = computed_kv[layer_idx + 1]
            else:
                k_above, v_above = k_top, v_top

            next_low_new = (k_l, v_l)
            next_high_new = (k_above, v_above)
            if kv_caches is None:
                next_high_old = None
            else:
                _, _, prev_high_new = kv_caches[layer_idx]
                next_high_old = prev_high_new

            new_kv_caches.append((next_low_new, next_high_old, next_high_new))

        return logits, new_kv_caches
