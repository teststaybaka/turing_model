"""Mamba-3 actor-critic for parallel dense tick-level RL.

Each tick input is:
  head reads, previous factorized actions, random latent noise

The actor emits four factorized action policies. The critic emits one value for
each action factor so moves and writes can receive independent dense rewards.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_ACTION_FACTORS = 4


@dataclass
class GPTConfig:
    block_size: int = 1024
    read_vocab_size: int = 16
    move_vocab_size: int = 5
    write_vocab_size: int = 18
    token_embd: int = 32
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    expand: int = 2
    d_state: int = 16
    mimo_rank: int = 1


def _rms_norm(x, weight, eps=1e-5):
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight).to(dtype)


class Mamba3Mixer(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_state % 2 == 0, "d_state must be even"
        self.n_heads = config.n_heads
        self.d_inner = config.expand * config.n_embd
        assert self.d_inner % config.n_heads == 0
        self.head_dim = self.d_inner // config.n_heads
        self.d_state = config.d_state
        self.R = config.mimo_rank
        nh, N, R = self.n_heads, self.d_state, self.R

        self.in_proj = nn.Linear(
            config.n_embd, self.d_inner + self.d_inner * R, bias=False
        )
        self.B_proj = nn.Linear(config.n_embd, nh * N * R, bias=False)
        self.C_proj = nn.Linear(config.n_embd, nh * N * R, bias=False)
        self.dt_proj = nn.Linear(config.n_embd, nh, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, nh + 1, dtype=torch.float32))
        )
        self.lam_proj = nn.Linear(config.n_embd, nh, bias=True)
        self.theta_proj = nn.Linear(config.n_embd, nh * (N // 2), bias=False)

        self.b_norm_w = nn.Parameter(torch.ones(N))
        self.c_norm_w = nn.Parameter(torch.ones(N))
        self.b_bias = nn.Parameter(torch.zeros(nh, N))
        self.c_bias = nn.Parameter(torch.zeros(nh, N))

        half = N // 2
        base_freq = 1.0 / (
            10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half)
        )
        self.register_buffer("theta_base", base_freq, persistent=False)
        self.out_proj = nn.Linear(self.d_inner, config.n_embd, bias=False)

    def reset_dt_bias(self):
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(self.n_heads)
                * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            )
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def _rotate_state(self, s, cos, sin):
        half = s.size(2) // 2
        s1, s2 = s[:, :, :half, :], s[:, :, half:, :]
        rot = torch.cat((-s2, s1), dim=2)
        return s * cos.unsqueeze(-1) + rot * sin.unsqueeze(-1)

    def forward(self, x, state=None):
        B, T, _ = x.size()
        nh, N, P, R = self.n_heads, self.d_state, self.head_dim, self.R

        zx = self.in_proj(x)
        z, xin = zx.split([self.d_inner, self.d_inner * R], dim=-1)
        xin = xin.view(B, T, nh, P, R)

        Bm = self.B_proj(x).view(B, T, nh, N, R)
        Cm = self.C_proj(x).view(B, T, nh, N, R)
        Bm = (
            _rms_norm(Bm.transpose(-1, -2), self.b_norm_w).transpose(-1, -2)
            + self.b_bias[None, None, :, :, None]
        )
        Cm = (
            _rms_norm(Cm.transpose(-1, -2), self.c_norm_w).transpose(-1, -2)
            + self.c_bias[None, None, :, :, None]
        )

        dt = F.softplus(self.dt_proj(x))
        A = -torch.exp(self.A_log)
        lam = torch.sigmoid(self.lam_proj(x))
        alpha = torch.exp(dt * A)
        beta = (1.0 - lam) * dt * alpha
        gamma = lam * dt

        theta = (
            self.theta_proj(x).view(B, T, nh, N // 2) + self.theta_base
        )
        ang = dt.unsqueeze(-1) * theta
        cos = torch.cat((torch.cos(ang), torch.cos(ang)), dim=-1)
        sin = torch.cat((torch.sin(ang), torch.sin(ang)), dim=-1)

        Bm = Bm.float()
        xin = xin.float()
        Cm = Cm.float()
        alpha = alpha.float()
        beta = beta.float()
        gamma = gamma.float()
        cos = cos.float()
        sin = sin.float()

        if state is None:
            g = x.new_zeros(B, nh, N, P, dtype=torch.float32)
            v_prev = x.new_zeros(B, nh, N, P, dtype=torch.float32)
        else:
            g, v_prev = state
            g = g.float()
            v_prev = v_prev.float()

        ys = []
        for t in range(T):
            v_t = torch.einsum("bhnr,bhpr->bhnp", Bm[:, t], xin[:, t])
            g = self._rotate_state(
                alpha[:, t, :, None, None] * g
                + beta[:, t, :, None, None] * v_prev,
                cos[:, t],
                sin[:, t],
            ) + gamma[:, t, :, None, None] * v_t
            v_prev = v_t
            y_t = torch.einsum(
                "bhnr,bhnp->bhpr", Cm[:, t], g
            ).reshape(B, nh * P * R)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        if R > 1:
            y = y.view(B, T, nh * P, R).mean(-1)
        y = y.to(x.dtype)
        y = y * F.silu(z)
        y = self.out_proj(y)
        return y, (g.to(x.dtype), v_prev.to(x.dtype))


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
        self.mlp = SwiGLUProject(config.n_embd, config.n_embd)

    def forward(self, x, state=None):
        mix_out, new_state = self.mixer(self.norm_1(x), state=state)
        x = x + mix_out
        x = x + self.mlp(self.norm_2(x))
        return x, new_state


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.read_emb = nn.Embedding(config.read_vocab_size, config.token_embd)
        self.move_emb = nn.Embedding(config.move_vocab_size, config.token_embd)
        self.write_emb = nn.Embedding(config.write_vocab_size, config.token_embd)
        self.input_mlp = SwiGLUProject(
            6 * config.token_embd + config.n_embd, config.n_embd
        )
        self.h = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.head0_move_mlp = SwiGLUProject(config.n_embd, config.n_embd)
        self.head1_move_mlp = SwiGLUProject(config.n_embd, config.n_embd)
        self.head0_write_mlp = SwiGLUProject(config.n_embd, config.n_embd)
        self.head1_write_mlp = SwiGLUProject(config.n_embd, config.n_embd)
        self.latent_mlp = SwiGLUProject(config.n_embd, config.n_embd)
        self.critic_mlp = SwiGLUProject(config.n_embd, config.n_embd)

        self.head0_move_ln = nn.LayerNorm(config.n_embd)
        self.head1_move_ln = nn.LayerNorm(config.n_embd)
        self.head0_write_ln = nn.LayerNorm(config.n_embd)
        self.head1_write_ln = nn.LayerNorm(config.n_embd)
        self.latent_ln = nn.LayerNorm(config.n_embd)
        self.critic_ln = nn.LayerNorm(config.n_embd)

        self.head0_move_head = nn.Linear(
            config.n_embd, config.move_vocab_size, bias=False
        )
        self.head1_move_head = nn.Linear(
            config.n_embd, config.move_vocab_size, bias=False
        )
        self.head0_write_head = nn.Linear(
            config.n_embd, config.write_vocab_size, bias=False
        )
        self.head1_write_head = nn.Linear(
            config.n_embd, config.write_vocab_size, bias=False
        )
        self.critic_head = nn.Linear(
            config.n_embd, NUM_ACTION_FACTORS, bias=True
        )

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

    def _build_input(self, reads, prev_actions, latent):
        assert reads.size(-1) == 2
        assert prev_actions.size(-1) == NUM_ACTION_FACTORS
        head0_read_x = self.read_emb(reads[..., 0])
        head1_read_x = self.read_emb(reads[..., 1])
        head0_move_x = self.move_emb(prev_actions[..., 0])
        head1_move_x = self.move_emb(prev_actions[..., 1])
        head0_write_x = self.write_emb(prev_actions[..., 2])
        head1_write_x = self.write_emb(prev_actions[..., 3])
        if latent is None:
            latent = head0_read_x.new_zeros(
                *head0_read_x.shape[:-1], self.config.n_embd
            )
        assert latent.shape == (
            *head0_read_x.shape[:-1],
            self.config.n_embd,
        )
        return self.input_mlp(
            torch.cat(
                [
                    head0_read_x,
                    head1_read_x,
                    head0_move_x,
                    head1_move_x,
                    head0_write_x,
                    head1_write_x,
                    latent.to(dtype=head0_read_x.dtype),
                ],
                dim=-1,
            )
        )

    def forward(self, reads, prev_actions, latent=None, states=None):
        x = self._build_input(reads, prev_actions, latent)
        new_states = []
        for layer_idx, block in enumerate(self.h):
            layer_state = None if states is None else states[layer_idx]
            x, new_state = block(x, state=layer_state)
            new_states.append(new_state)
        x = self.ln_f(x)

        logits = (
            self.head0_move_head(self.head0_move_ln(self.head0_move_mlp(x))),
            self.head1_move_head(self.head1_move_ln(self.head1_move_mlp(x))),
            self.head0_write_head(self.head0_write_ln(self.head0_write_mlp(x))),
            self.head1_write_head(self.head1_write_ln(self.head1_write_mlp(x))),
        )
        values = self.critic_head(self.critic_ln(self.critic_mlp(x)))
        latent_out = self.latent_ln(self.latent_mlp(x))
        return logits, values, latent_out, new_states
