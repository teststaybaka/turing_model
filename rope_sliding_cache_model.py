"""Diff from model.py (standard GPT):
- RoPE replaces learned positional embeddings (split-half convention, same as rope_model.py).
- Each attention head has a sliding window of size W = block_size, attending to the last W K/V pairs.
- KV cache carry: each forward() returns new K/V for the chunk, stacks them on top of the next step's K/V pairs.
- Infinite context: the cache stores K UNROTATED; Q and K are rotated at attention time
  using cache-relative indices (key j of [K_cache | K_new] -> j, query q -> T_cache + q).
  RoPE scores depend only on index DIFFERENCES, so the origin is arbitrary; with a bounded
  window the indices never exceed 2*block_size, so one fixed cos/sin table serves
  arbitrarily long streams — no absolute position tracking, no table regrowth, no drift.
- Standard next-token alignment: logits[i] predicts idx[i+1] (no pair-sum, no prev_tokens).
- FlexAttention for the cache-carry branch (same as stair/multi_scale models): a block mask
  skips the ~half of [K_cache | K_new] outside each query's window, vs a dense attn_mask
  which would force SDPA off the flash kernel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

@dataclass
class GPTConfig:
  block_size: int = 1024 # also the sliding-window size and KV cache size
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


class CausalSelfAttention(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    assert config.n_embd % config.n_heads == 0
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.n_heads = config.n_heads
    self.n_embd = config.n_embd
    # sliding-window size in tokens; also the size of the KV cache.
    self.window_size = config.block_size
    # BlockMask cache keyed by (T_new, T_cache, device). Training sees one steady-state
    # shape; eval tail chunks add a few more.
    self._block_mask_cache = {}

  def _get_block_mask(self, T_new, T_cache, device):
    key = (T_new, T_cache, str(device))
    cached = self._block_mask_cache.get(key)
    if cached is not None:
      return cached
    W = self.window_size
    def mask_mod(b, h, q_idx, kv_idx):
      # K_all layout: [K_cache (T_cache) | k_live (T_new)]. Query at intra-chunk
      # index q sits at K_all index q + T_cache; it attends to the last W keys.
      dist = q_idx + T_cache - kv_idx
      return (dist >= 0) & (dist < W)
    block_mask = create_block_mask(
        mask_mod, B=None, H=None,
        Q_LEN=T_new, KV_LEN=T_cache + T_new,
        device=device,
    )
    self._block_mask_cache[key] = block_mask
    return block_mask

  def forward(self, x, cos, sin, kv_cache=None):
    """
    x: (B, T_new, C) — chunk's input embeddings.
    cos, sin: full (2*block_size, head_dim) RoPE tables; sliced here by cache-relative index.
    kv_cache: optional (K_cache, V_cache), each (B, n_heads, T_cache, head_size).
              K_cache is UNROTATED — rotation is applied fresh each forward with
              window-relative indices, which is what makes the context unbounded.
              Caller must detach these if no gradient should flow through them.
              Pass None for the prefill chunk.
    Returns:
      y: (B, T_new, C)
      (k, v): newly computed UNROTATED K, V for this chunk, each (B, n_heads, T_new, head_size).
              Caller uses these to build the next step's KV cache.
    """
    B, T_new, C = x.size()
    head_size = C // self.n_heads

    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    q = q.view(B, T_new, self.n_heads, head_size).transpose(1, 2) # (B, nh, T_new, hs)
    k = k.view(B, T_new, self.n_heads, head_size).transpose(1, 2)
    v = v.view(B, T_new, self.n_heads, head_size).transpose(1, 2)

    if kv_cache is None:
      # Prefill chunk: no cached context, standard causal attention within the chunk.
      q_rot = _apply_rope(q, cos[:T_new], sin[:T_new])
      k_rot = _apply_rope(k, cos[:T_new], sin[:T_new])
      y = F.scaled_dot_product_attention(q_rot, k_rot, v, is_causal=True)
    else:
      K_cache, V_cache = kv_cache
      T_cache = K_cache.size(2)
      T_all = T_cache + T_new
      K_all = torch.cat([K_cache, k], dim=2) # (B, nh, T_all, hs), unrotated
      V_all = torch.cat([V_cache, v], dim=2)

      # Cache-relative rotation: key j -> index j, query q -> index T_cache + q.
      # Relative angle between query and key depends only on their distance, so this
      # is exactly equivalent to absolute-position RoPE — but indices stay < 2W forever.
      k_rot = _apply_rope(K_all, cos[:T_all], sin[:T_all])
      q_rot = _apply_rope(q, cos[T_cache:T_all], sin[T_cache:T_all])

      # Sliding-window mask, same rule as sliding_cache_model: query at intra-chunk
      # index q_i attends to K_all indices [q_i + T_cache - W + 1, q_i + T_cache].
      # FlexAttention block mask skips the ~half of K_all outside each query's window.
      block_mask = self._get_block_mask(T_new, T_cache, q.device)
      y = flex_attention(q_rot, k_rot, V_all, block_mask=block_mask)

    y = y.transpose(1, 2).contiguous().view(B, T_new, C) # re-assemble heads
    y = self.c_proj(y)
    return y, (k, v)


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
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)

  def forward(self, x, cos, sin, kv_cache=None):
    attn_out, new_kv = self.attn(self.ln_1(x), cos, sin, kv_cache=kv_cache)
    x = x + attn_out
    x = x + self.mlp(self.ln_2(x))
    return x, new_kv


class GPT(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.config = config
    self.transformer = nn.ModuleDict({
        'wte': nn.Embedding(config.vocab_size, config.n_embd),
        'h': nn.ModuleList(Block(config) for _ in range(config.n_layers)),
        'ln_f': nn.LayerNorm(config.n_embd),
    })
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    # weight sharing
    self.transformer['wte'].weight = self.lm_head.weight

    # RoPE tables sized 2*block_size: max cache-relative index is
    # T_cache + T_new - 1 <= 2W - 1. Non-persistent (recomputable from config).
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

  def forward(self, idx, kv_caches=None):
    """
    idx: (B, T_new) — this chunk's new token ids. T_new <= block_size.
    kv_caches: list of (K, V) per layer, each (B, n_heads, T_cache, head_size), UNROTATED.
               None for the first chunk of a shard. Caller must detach between optimizer steps.
    Returns:
      logits: (B, T_new, vocab_size) — logits[i] predicts idx[i+1]
      new_kv_caches: list of (K, V) per layer for the next call (NOT detached)
    """
    B, T_new = idx.size()
    assert T_new <= self.config.block_size, "chunk exceeds block_size"

    x = self.transformer['wte'](idx)

    new_kv_caches = []
    for layer_idx, block in enumerate(self.transformer.h):
      layer_cache = None if kv_caches is None else kv_caches[layer_idx]
      x, new_kv = block(x, self.rope_cos, self.rope_sin, kv_cache=layer_cache)
      new_kv_caches.append(new_kv)

    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)
    return logits, new_kv_caches
