"""Hybrid stair/sliding model — staircase feedback confined to the bottom layers.

Motivation: in the full stair model every layer's K/V serves a double duty — it is
both this layer's recent-context key AND the layer-below's fed-back "high" slab. The
deeper that feedback climbs, the more it may confuse a small model. Here we split the
roles BY LAYER instead of by projection:

  - Bottom `n_stair_layers` layers: unchanged StairAttention. Each attends to its own
    recent W/2 (low) plus the older W/2 from one layer deeper (high, fed back). For the
    4-layer config this is layers 0 and 1, with high caches from layers 1 and 2.
  - Remaining layers: plain sliding-window attention (rope_sliding_cache_model), window
    W over the layer's OWN history — high cache is NOT staired/skewed. The top such
    layer produces the final embedding that feeds the MLP head.

So the front of the stack maintains long memory via the staircase recurrence and the
back of the stack reads it locally to predict. The KVOnly head from the pure stair
model is removed: it only existed to source the top layer's high cache, which no
longer exists (the top layers use their own sliding history).

Same harness contract as the other models:
  GPT(config); forward(idx, kv_caches=None) -> (logits, new_kv_caches)
Per-layer cache carried across chunks with full BPTT (no detach), None on the first
chunk. The cache STRUCTURE depends on the layer type:
  - stair layer:   (low_new, high_old, high_new)  — same 3-tuple as rope_stair_model
  - sliding layer: (K_prev, V_prev)               — the layer's own previous chunk K/V

Requires chunk_size <= block_size/2 (strict stair invariant), same as rope_stair_model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from rope_stair_model import StairAttention, _precompute_rope
from rope_sliding_cache_model import CausalSelfAttention


@dataclass
class GPTConfig:
    block_size: int = 1024 # attention window W; expected chunk size at training is W/2
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12 #nh
    n_embd: int = 768
    rope_base: float = 10000.0 # RoPE frequency base (GPT-NeoX / Llama convention)
    n_stair_layers: int = 2 # bottom layers use StairAttention; the rest use sliding-window


class MLP(nn.Module):
    """SwiGLU MLP — identical to the other models so the MLP block is not a confound."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = 4 * config.n_embd
        self.w_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.w_up = nn.Linear(config.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    """One transformer block. `attn` is either a StairAttention or a sliding-window
    CausalSelfAttention; both share the (x, cos, sin, cache) forward signature and
    return (y, (k, v)) with y already projected and head-merged."""
    def __init__(self, config: GPTConfig, is_stair: bool):
        super().__init__()
        self.is_stair = is_stair
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = StairAttention(config) if is_stair else CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin, cache=None):
        attn_out, new_kv = self.attn(self.ln_1(x), cos, sin, cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert 0 < config.n_stair_layers < config.n_layers, \
            "need at least one stair layer and one sliding layer"
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = nn.ModuleList(
            Block(config, is_stair=(i < config.n_stair_layers)) for i in range(config.n_layers)
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight sharing
        self.wte.weight = self.lm_head.weight

        # RoPE tables sized 2*block_size, covering both the stair chunk-relative range
        # [-W, W/2) (shifted +W) and the sliding cache-relative range [0, 2W). Non-persistent.
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
        idx: (B, T_new) — this chunk's new token ids. T_new must be <= block_size/2.
        kv_caches: list of n_layers entries, None for the first chunk of a shard.
                   Stair layers carry (low_new, high_old, high_new); sliding layers carry
                   (K_prev, V_prev). K entries are UNROTATED. Reset to None at each
                   optimizer-step boundary.
        Returns:
          logits: (B, T_new, vocab_size) — logits[i] predicts idx[i+1]
          new_kv_caches: list of n_layers caches for the next call.
        """
        B, T_new = idx.size()
        W = self.config.block_size
        n_stair = self.config.n_stair_layers
        assert T_new <= W // 2, f"chunk size {T_new} exceeds block_size/2 = {W//2}; stair invariant breaks."

        x = self.wte(idx)

        computed_kv = []
        for layer_idx, block in enumerate(self.h):
            layer_cache = None if kv_caches is None else kv_caches[layer_idx]
            x, (k_new, v_new) = block(x, self.rope_cos, self.rope_sin, cache=layer_cache)
            computed_kv.append((k_new, v_new))

        logits = self.lm_head(self.ln_f(x))

        new_kv_caches = []
        for layer_idx in range(self.config.n_layers):
            if layer_idx < n_stair:
                # Stair routing, identical to rope_stair_model: recent W/2 from this layer
                # (low), older W/2 from one layer deeper (high). layer_idx+1 always exists
                # because n_stair < n_layers.
                next_low_new = computed_kv[layer_idx]
                next_high_new = computed_kv[layer_idx + 1]
                if kv_caches is None:
                    next_high_old = None
                else:
                    next_high_old = kv_caches[layer_idx][2] # previous chunk's high_new
                new_kv_caches.append((next_low_new, next_high_old, next_high_new))
            else:
                # Sliding layer: carry only this chunk's own K/V (one chunk = W/2), giving
                # a bounded window W = [prev chunk | live] over the layer's own history.
                new_kv_caches.append(computed_kv[layer_idx])

        return logits, new_kv_caches
