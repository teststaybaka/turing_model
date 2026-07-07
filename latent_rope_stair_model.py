"""Latent-feedback copy of rope_stair_model.py.

Input at each position is a learned fusion of [token_embedding, latent_embedding].
The final hidden state feeds separate token and latent output heads.

Diff from stair_model.py:
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
    """Streaming stair attention with rolling low/high cache slabs.

    cache is optional (low_rolling, high_rolling):
      low_rolling: same-layer K/V, truncated to W/2
      high_rolling: one-layer-deeper K/V, truncated to W
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        assert config.block_size % 2 == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.window_size = config.block_size
        self._attn_mask_cache = {}

    def _get_attn_mask(self, T_new, T_low, T_high_total, device):
        key = (T_new, T_low, T_high_total, str(device))
        cached = self._attn_mask_cache.get(key)
        if cached is not None:
            return cached
        W = self.window_size
        half_W = W // 2
        q_idx = torch.arange(T_new, device=device)[:, None]
        kv_idx = torch.arange(T_high_total + T_low + T_new, device=device)[None, :]
        # K_combined layout: [K_high | K_low | K_live].
        is_high = kv_idx < T_high_total
        high_dist = q_idx - (kv_idx - T_high_total)
        low_dist = q_idx - (kv_idx - T_high_total - T_low)
        high_ok = (high_dist >= half_W) & (high_dist < W)
        low_ok = (low_dist >= 0) & (low_dist < half_W)
        attn_mask = torch.where(is_high, high_ok, low_ok)[None, None, :, :]
        self._attn_mask_cache[key] = attn_mask
        return attn_mask

    def forward(self, x, cos, sin, cache=None):
        B, T_new, C = x.size()
        head_size = C // self.n_heads
        W = self.window_size

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T_new, self.n_heads, head_size).transpose(1, 2)
        k = k.view(B, T_new, self.n_heads, head_size).transpose(1, 2)
        v = v.view(B, T_new, self.n_heads, head_size).transpose(1, 2)

        q_rot = _apply_rope(q, cos[W:W + T_new], sin[W:W + T_new])

        if cache is None:
            k_rot = _apply_rope(k, cos[W:W + T_new], sin[W:W + T_new])
            y = F.scaled_dot_product_attention(q_rot, k_rot, v, is_causal=True)
            return self._merge_heads(y, B, T_new, C), (k, v)

        low_rolling, high_rolling = cache
        K_low, V_low = low_rolling
        K_high, V_high = high_rolling
        T_low = K_low.size(2)
        T_high_total = K_high.size(2)

        K_combined = torch.cat([K_high, K_low, k], dim=2)
        V_combined = torch.cat([V_high, V_low, v], dim=2)

        pos = torch.cat([
            torch.arange(W - T_high_total, W, device=q.device),
            torch.arange(W - T_low, W + T_new, device=q.device),
        ])
        k_rot = _apply_rope(K_combined, cos[pos], sin[pos])

        attn_mask = self._get_attn_mask(T_new, T_low, T_high_total, q.device)
        y = F.scaled_dot_product_attention(q_rot, k_rot, V_combined, attn_mask=attn_mask)
        return self._merge_heads(y, B, T_new, C), (k, v)

    def _merge_heads(self, y, B, T_new, C):
        y = y.transpose(1, 2).contiguous().view(B, T_new, C)
        return self.c_proj(y)


class MLP(nn.Module):
    """SwiGLU MLP — identical to mamba3_model.py so the MLP block is the same across
    all three models (controls for the MLP as a confound in the comparison)."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = 4 * config.n_embd
        self.w_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.w_up = nn.Linear(config.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class SwiGLUProject(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        hidden = 4 * out_dim
        self.w_gate = nn.Linear(in_dim, hidden, bias=False)
        self.w_up = nn.Linear(in_dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, out_dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


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
        self.input_mlp = SwiGLUProject(2 * config.n_embd, config.n_embd)
        self.token_mlp = SwiGLUProject(config.n_embd, config.n_embd)
        self.latent_mlp = SwiGLUProject(config.n_embd, config.n_embd)
        self.token_ln = nn.LayerNorm(config.n_embd)
        self.latent_ln = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.kv_only = KVOnly(config)

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

    def _build_input(self, idx, latent):
        B, T = idx.size()
        token_x = self.wte(idx)
        if latent is None:
            latent = token_x.new_zeros(B, T, self.config.n_embd)
        assert latent.shape == (B, T, self.config.n_embd)
        return self.input_mlp(torch.cat([token_x, latent.to(dtype=token_x.dtype)], dim=-1))

    def forward(self, idx, latent=None, kv_caches=None):
        """
        idx: (B, T_new) token ids. T_new must be <= block_size/2.
        latent: optional (B, T_new, n_embd) latent feedback inputs.
        kv_caches: list of (low_rolling, high_rolling) per layer, or None.
        Returns logits, latent_out, and new rolling stair caches.
        """
        B, T_new = idx.size()
        W = self.config.block_size
        half_W = W // 2
        assert T_new <= half_W, f"chunk size {T_new} exceeds block_size/2 = {half_W}; strict stair invariant breaks."

        x = self._build_input(idx, latent)

        computed_kv = []
        for layer_idx, block in enumerate(self.h):
            layer_cache = None if kv_caches is None else kv_caches[layer_idx]
            x, (k_new, v_new) = block(x, self.rope_cos, self.rope_sin, cache=layer_cache)
            computed_kv.append((k_new, v_new))

        k_top, v_top = self.kv_only(x)

        x_out = self.ln_f(x)
        token_state = self.token_ln(self.token_mlp(x_out))
        latent_out = self.latent_ln(self.latent_mlp(x_out))
        logits = self.lm_head(token_state)

        new_kv_caches = []
        n_layers = len(self.h)
        for layer_idx in range(n_layers):
            k_l, v_l = computed_kv[layer_idx]
            if layer_idx + 1 < n_layers:
                k_above, v_above = computed_kv[layer_idx + 1]
            else:
                k_above, v_above = k_top, v_top

            if kv_caches is None:
                k_low, v_low = k_l, v_l
                k_high, v_high = k_above, v_above
            else:
                prev_low, prev_high = kv_caches[layer_idx]
                k_prev_low, v_prev_low = prev_low
                k_prev_high, v_prev_high = prev_high
                k_low = torch.cat([k_prev_low, k_l], dim=2)
                v_low = torch.cat([v_prev_low, v_l], dim=2)
                k_high = torch.cat([k_prev_high, k_above], dim=2)
                v_high = torch.cat([v_prev_high, v_above], dim=2)

            if k_low.size(2) > half_W:
                k_low = k_low[:, :, -half_W:, :]
                v_low = v_low[:, :, -half_W:, :]
            if k_high.size(2) > W:
                k_high = k_high[:, :, -W:, :]
                v_high = v_high[:, :, -W:, :]

            new_kv_caches.append(((k_low, v_low), (k_high, v_high)))

        return logits, latent_out, new_kv_caches
