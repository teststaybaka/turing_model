"""Latent-feedback copy of mamba3_model.py.

Input at each position is a learned fusion of [token_embedding, latent_embedding].
The final hidden state feeds separate token and latent output heads.

Mamba-3 (ICLR 2026, "Mamba-3: Improved Sequence Modeling using State Space
Principles", openreview HwCvaJOiCj) as a recurrent baseline for the infinite-context
Turing-machine tasks.

Same harness contract as rope_sliding_cache_model / rope_stair_model:
  GPT(config); forward(idx, kv_caches=None) -> (logits, new_kv_caches)
The per-layer carry IS the SSM recurrent state, propagated across chunks with full
BPTT (no detach) — the SSM analogue of this project's KV-cache carry. None on the
first chunk of a shard.

Faithful to the paper's three contributions:

1. Exponential-trapezoidal discretization (Prop 1/4). The state-input integral is a
   data-dependent convex combination of the two interval endpoints (a width-2 conv on
   the state-input, INSIDE the recurrence):
       h_t = a_t h_{t-1} + b_t (B_{t-1} x_{t-1}^T) + g_t (B_t x_t^T)
   with a_t = e^{dt_t A_t}, b_t = (1-lam_t) dt_t e^{dt_t A_t}, g_t = lam_t dt_t,
   lam_t in [0,1] data-dependent (lam_t=1 recovers Mamba-2 Euler).

2. Complex-valued / rotational state (Prop 2/4). The real-valued equivalent of a
   complex SSM applies a block-diagonal of 2x2 rotations R(dt_t * theta_t) to the
   state each step — "data-dependent RoPE on the SSM state" — which lets the state
   represent rotational (e.g. parity) dynamics that real non-negative eigenvalues
   cannot. We apply the rotation directly to the carried state (frame-local form,
   equivalent to the paper's cumulative-product RoPE trick); rotations and the scalar
   decay/input terms are linear, so a_t and b_t share the step rotation:
       h_t = R(dt_t theta_t) [ a_t h_{t-1} + b_t v_{t-1} ] + g_t v_t,   v_t = B_t x_t^T
       y_t = C_t^T h_t

3. MIMO (Sec 3.3, optional, off by default). Lifts B/x and C to rank `mimo_rank`,
   increasing arithmetic intensity. Default SISO (rank 1) for fair comparison, exactly
   as the paper's headline SISO model.

Architecture (Sec 3.4): Llama-style, each layer = pre-norm Mamba-3 mixer (+residual)
then pre-norm SwiGLU MLP (+residual). BCNorm = RMSNorm on B and C, plus learnable
head-wise channel biases. Gated SSM output (SiLU gate). No short conv1d and no
post-gate RMSNorm (the trapezoidal implicit conv + B/C biases replace the conv).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024   # expected chunk size; the SSM has no hard window, this only sizes nothing here
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12        # number of SSM heads (and MLP unaffected)
    n_embd: int = 768
    # Mamba-3 specific (defaulted so the 5-field GPTConfig(...) calls in train.py still work):
    expand: int = 2          # d_inner = expand * n_embd
    d_state: int = 16        # SSM state size N per head; must be even (2x2 rotation blocks)
    mimo_rank: int = 1       # 1 = SISO (paper default); >1 = MIMO


def _rms_norm(x, weight, eps=1e-5):
    """RMSNorm over the last dim. weight broadcasts over the last dim."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight).to(dtype)


class Mamba3Mixer(nn.Module):
    """One Mamba-3 SSM mixer (SISO by default). Carries (g, v_prev) across chunks."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_state % 2 == 0, "d_state must be even (2x2 rotation blocks)"
        self.n_heads = config.n_heads
        self.d_inner = config.expand * config.n_embd
        assert self.d_inner % config.n_heads == 0
        self.head_dim = self.d_inner // config.n_heads      # P
        self.d_state = config.d_state                       # N
        self.R = config.mimo_rank
        nh, N, P, R = self.n_heads, self.d_state, self.head_dim, self.R

        # Input projection -> gate z (d_inner) and SSM input x (d_inner * R for MIMO).
        self.in_proj = nn.Linear(config.n_embd, self.d_inner + self.d_inner * R, bias=False)
        # Data-dependent SSM parameters, all per token.
        self.B_proj = nn.Linear(config.n_embd, nh * N * R, bias=False)
        self.C_proj = nn.Linear(config.n_embd, nh * N * R, bias=False)
        self.dt_proj = nn.Linear(config.n_embd, nh, bias=True)      # Delta (softplus)
        # Static per-head decay A = -exp(A_log), log-spaced timescales (Mamba-2 init),
        # NOT data-dependent — matches the official Mamba-3 / state-spaces convention and
        # gives heads diverse memory horizons at init. ndim==1 so train.py won't weight-decay it.
        self.A_log = nn.Parameter(torch.log(torch.arange(1, nh + 1, dtype=torch.float32)))
        self.lam_proj = nn.Linear(config.n_embd, nh, bias=True)     # trapezoidal lambda (sigmoid)
        self.theta_proj = nn.Linear(config.n_embd, nh * (N // 2), bias=False)  # rotation angles

        # BCNorm (RMSNorm on B, C over the state dim N) weights, shared across heads.
        self.b_norm_w = nn.Parameter(torch.ones(N))
        self.c_norm_w = nn.Parameter(torch.ones(N))
        # Head-wise, channel-wise learnable biases added to B, C after BCNorm.
        self.b_bias = nn.Parameter(torch.zeros(nh, N))
        self.c_bias = nn.Parameter(torch.zeros(nh, N))
        # Log-spaced base frequencies (RoPE-like) added to the data-dependent angles to
        # break head/pair symmetry and seed diverse rotational dynamics.
        half = N // 2
        base_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer('theta_base', base_freq, persistent=False)

        self.out_proj = nn.Linear(self.d_inner, config.n_embd, bias=False)

    def reset_dt_bias(self):
        """dt bias init so softplus(dt_bias) lands in a reasonable Delta range. Called
        AFTER GPT's generic _init_weights (which would otherwise zero this bias)."""
        with torch.no_grad():
            dt = torch.exp(torch.rand(self.n_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus

    def _rotate_state(self, s, cos, sin):
        """Rotate s: (B, nh, N, P) along the N axis by block-diagonal 2x2 rotations.
        cos, sin: (B, nh, N) built by duplicating the (B, nh, N/2) angle tables
        (split-half convention, same as the RoPE models)."""
        half = s.size(2) // 2
        s1, s2 = s[:, :, :half, :], s[:, :, half:, :]
        rot = torch.cat((-s2, s1), dim=2)
        return s * cos.unsqueeze(-1) + rot * sin.unsqueeze(-1)

    def forward(self, x, state=None):
        """
        x: (B, T, C) chunk input.
        state: (g, v_prev) carry from the previous chunk, each (B, nh, N, P*R-ish);
               None on the first chunk. Returned (NOT detached) for the next chunk.
        Returns y: (B, T, C), new_state: (g, v_prev).
        """
        B, T, C = x.size()
        nh, N, P, R = self.n_heads, self.d_state, self.head_dim, self.R

        zx = self.in_proj(x)
        z, xin = zx.split([self.d_inner, self.d_inner * R], dim=-1)
        xin = xin.view(B, T, nh, P, R)                                  # SSM input (MIMO rank R)

        Bm = self.B_proj(x).view(B, T, nh, N, R)
        Cm = self.C_proj(x).view(B, T, nh, N, R)
        # BCNorm over N, then head-wise channel bias.
        Bm = _rms_norm(Bm.transpose(-1, -2), self.b_norm_w).transpose(-1, -2) + self.b_bias[None, None, :, :, None]
        Cm = _rms_norm(Cm.transpose(-1, -2), self.c_norm_w).transpose(-1, -2) + self.c_bias[None, None, :, :, None]

        dt = F.softplus(self.dt_proj(x))                               # (B,T,nh)  > 0
        A = -torch.exp(self.A_log)                                     # (nh,)     < 0, static
        lam = torch.sigmoid(self.lam_proj(x))                          # (B,T,nh) in (0,1)
        alpha = torch.exp(dt * A)                                      # (B,T,nh) in (0,1)
        beta = (1.0 - lam) * dt * alpha                                # (B,T,nh)
        gamma = lam * dt                                               # (B,T,nh)

        # Per-step rotation angles: dt * (data-dependent theta + log-spaced base freq).
        theta = self.theta_proj(x).view(B, T, nh, N // 2) + self.theta_base
        ang = dt.unsqueeze(-1) * theta                                 # (B,T,nh,N/2)
        cos = torch.cat((torch.cos(ang), torch.cos(ang)), dim=-1)      # (B,T,nh,N)
        sin = torch.cat((torch.sin(ang), torch.sin(ang)), dim=-1)

        # Run the recurrence in fp32 for stability under bf16 autocast.
        # State-input outer product v_t = B_t (x)_t^T, contracted over the MIMO rank R:
        #   v_t[n, p] = sum_r B_t[n, r] * x_t[p, r]   -> (B, nh, N, P)
        Bm = Bm.float(); xin = xin.float(); Cm = Cm.float()
        alpha = alpha.float(); beta = beta.float(); gamma = gamma.float()
        cos = cos.float(); sin = sin.float()

        if state is None:
            g = x.new_zeros(B, nh, N, P, dtype=torch.float32)
            v_prev = x.new_zeros(B, nh, N, P, dtype=torch.float32)
        else:
            g, v_prev = state
            g = g.float(); v_prev = v_prev.float()

        ys = []
        for t in range(T):
            v_t = torch.einsum('bhnr,bhpr->bhnp', Bm[:, t], xin[:, t])
            # h_t = R(dt theta) [ alpha h_{t-1} (+ beta v_{t-1}) ] + gamma v_t
            # h_t = R(dt theta) [ alpha h_{t-1} + beta v_{t-1} ] + gamma v_t
            g = self._rotate_state(
                alpha[:, t, :, None, None] * g + beta[:, t, :, None, None] * v_prev,
                cos[:, t], sin[:, t],
            ) + gamma[:, t, :, None, None] * v_t
            v_prev = v_t
            y_t = torch.einsum('bhnr,bhnp->bhpr', Cm[:, t], g).reshape(B, nh * P * R)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                                     # (B, T, nh*P*R)
        if R > 1:
            # MIMO: reduce the output rank back to d_inner (mean over R).
            y = y.view(B, T, nh * P, R).mean(-1)
        y = y.to(x.dtype)
        y = y * F.silu(z)                                              # gated SSM output
        y = self.out_proj(y)
        return y, (g.to(x.dtype), v_prev.to(x.dtype))


class SwiGLU(nn.Module):
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
        self.norm_1 = nn.LayerNorm(config.n_embd)
        self.mixer = Mamba3Mixer(config)
        self.norm_2 = nn.LayerNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, state=None):
        mix_out, new_state = self.mixer(self.norm_1(x), state=state)
        x = x + mix_out
        x = x + self.mlp(self.norm_2(x))
        return x, new_state


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
        self.apply(self._init_weights)
        for module in self.modules():
            if isinstance(module, Mamba3Mixer):
                module.reset_dt_bias()

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
        idx: (B, T_new) token ids.
        latent: optional (B, T_new, n_embd) latent feedback inputs.
        kv_caches: list of per-layer SSM states (g, v_prev), or None.
        Returns logits, latent_out, and new per-layer recurrent states.
        """
        x = self._build_input(idx, latent)
        new_states = []
        for layer_idx, block in enumerate(self.transformer.h):
            layer_state = None if kv_caches is None else kv_caches[layer_idx]
            x, new_state = block(x, state=layer_state)
            new_states.append(new_state)
        x = self.transformer.ln_f(x)
        token_state = self.token_ln(self.token_mlp(x))
        latent_out = self.latent_ln(self.latent_mlp(x))
        logits = self.lm_head(token_state)
        return logits, latent_out, new_states
