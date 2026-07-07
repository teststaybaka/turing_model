"""Latent-feedback copy of rope_sliding_cache_model.py.

Input at each position is a learned fusion of [token_embedding, latent_embedding].
The final hidden state feeds separate token and latent output heads.

Diff from model.py (standard GPT):
- RoPE replaces learned positional embeddings (split-half convention, same as rope_model.py).
- Each attention head has a sliding window of size W = block_size, attending to the last W K/V pairs.
- KV cache carry: each forward() returns new K/V for the chunk, stacks them on top of the next step's K/V pairs.
- Infinite context: the cache stores K UNROTATED; Q and K are rotated at attention time
  using cache-relative indices (key j of [K_cache | K_new] -> j, query q -> T_cache + q).
  RoPE scores depend only on index DIFFERENCES, so the origin is arbitrary; with a bounded
  window the indices never exceed 2*block_size, so one fixed cos/sin table serves
  arbitrarily long streams — no absolute position tracking, no table regrowth, no drift.
- Standard next-token alignment: logits[i] predicts idx[i+1] (no pair-sum, no prev_tokens).
- The cache-carry branch uses a materialized boolean SDPA mask instead of
  FlexAttention, which avoids FlexAttention shape-specialization issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

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
    self._attn_mask_cache = {}

  def _get_attn_mask(self, T_new, T_cache, device):
    key = (T_new, T_cache, str(device))
    cached = self._attn_mask_cache.get(key)
    if cached is not None:
      return cached
    W = self.window_size
    q_idx = torch.arange(T_new, device=device)[:, None]
    kv_idx = torch.arange(T_cache + T_new, device=device)[None, :]
    # K_all layout: [K_cache (T_cache) | k_live (T_new)]. Query q sits at
    # K_all index q + T_cache and attends to the last W keys.
    dist = q_idx + T_cache - kv_idx
    attn_mask = ((dist >= 0) & (dist < W))[None, None, :, :]
    self._attn_mask_cache[key] = attn_mask
    return attn_mask

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

      # Sliding-window mask: query q_i attends to K_all indices
      # [q_i + T_cache - W + 1, q_i + T_cache].
      attn_mask = self._get_attn_mask(T_new, T_cache, q.device)
      y = F.scaled_dot_product_attention(q_rot, k_rot, V_all, attn_mask=attn_mask)

    y = y.transpose(1, 2).contiguous().view(B, T_new, C) # re-assemble heads
    y = self.c_proj(y)
    return y, (k, v)


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
    self.input_mlp = SwiGLUProject(2 * config.n_embd, config.n_embd)
    self.token_mlp = SwiGLUProject(config.n_embd, config.n_embd)
    self.latent_mlp = SwiGLUProject(config.n_embd, config.n_embd)
    self.token_ln = nn.LayerNorm(config.n_embd)
    self.latent_ln = nn.LayerNorm(config.n_embd)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

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

  def _build_input(self, idx, latent):
    B, T = idx.size()
    token_x = self.transformer['wte'](idx)
    if latent is None:
      latent = token_x.new_zeros(B, T, self.config.n_embd)
    assert latent.shape == (B, T, self.config.n_embd)
    return self.input_mlp(torch.cat([token_x, latent.to(dtype=token_x.dtype)], dim=-1))

  def forward(self, idx, latent=None, kv_caches=None):
    """
    idx: (B, T_new) token ids. T_new <= block_size.
    latent: optional (B, T_new, n_embd) latent feedback inputs.
    kv_caches: list of accumulated (K, V) rolling windows per layer, or None.
    Returns logits, latent_out, and new accumulated K/V windows.
    """
    B, T_new = idx.size()
    assert T_new <= self.config.block_size, "chunk exceeds block_size"

    x = self._build_input(idx, latent)

    fresh_kv = []
    for layer_idx, block in enumerate(self.transformer.h):
      layer_cache = None if kv_caches is None else kv_caches[layer_idx]
      x, new_kv = block(x, self.rope_cos, self.rope_sin, kv_cache=layer_cache)
      fresh_kv.append(new_kv)

    x = self.transformer.ln_f(x)
    token_state = self.token_ln(self.token_mlp(x))
    latent_out = self.latent_ln(self.latent_mlp(x))
    logits = self.lm_head(token_state)

    new_kv_caches = []
    max_cache = self.config.block_size - 1
    for layer_idx, (k_new, v_new) in enumerate(fresh_kv):
      if kv_caches is None:
        k_all, v_all = k_new, v_new
      else:
        k_prev, v_prev = kv_caches[layer_idx]
        k_all = torch.cat([k_prev, k_new], dim=2)
        v_all = torch.cat([v_prev, v_new], dim=2)
      if k_all.size(2) > max_cache:
        k_all = k_all[:, :, -max_cache:, :]
        v_all = v_all[:, :, -max_cache:, :]
      new_kv_caches.append((k_all, v_all))

    return logits, latent_out, new_kv_caches
