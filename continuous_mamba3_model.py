"""Mamba-3 recurrent transducer with tied input and output embeddings.

Each categorical input stream occupies one fixed-width slice of the Mamba
residual stream. The final residual stream is split into the same slices and
decoded with the transposes of the corresponding input embedding tables. The
categorical outputs therefore use exactly the same vocabularies as the inputs.

Scalar streams use tied scalar-to-vector and vector-to-scalar projections. A
single scalar stream is used for previous reward input and expected reward
output in the continuous-episode experiment.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    input_streams: tuple[tuple[str, int], ...] = (
        ("read", 17),
        ("read", 17),
    )
    action_input_streams: tuple[tuple[str, int], ...] = (
        ("move", 3),
        ("move", 3),
        ("write", 18),
        ("write", 18),
    )
    token_embd: int = 16
    n_layers: int = 12
    n_heads: int = 4
    expand: int = 2
    d_state: int = 16
    mimo_rank: int = 1
    num_scalar_streams: int = 1

    @property
    def stream_specs(self):
        return self.input_streams + self.action_input_streams

    @property
    def num_inputs(self):
        return len(self.input_streams)

    @property
    def num_outputs(self):
        return len(self.stream_specs)

    @property
    def num_action_inputs(self):
        return len(self.action_input_streams)

    @property
    def n_embd(self):
        return (
            self.num_outputs + self.num_scalar_streams
        ) * self.token_embd


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
        if config.num_inputs == 0:
            raise ValueError("at least one input stream is required")
        if config.num_scalar_streams < 0:
            raise ValueError("num_scalar_streams cannot be negative")
        if config.token_embd <= 0:
            raise ValueError("token_embd must be positive")
        if config.expand * config.n_embd % config.n_heads:
            raise ValueError(
                "expanded embedding width must be divisible by n_heads"
            )

        group_vocab_sizes = {}
        for group, vocab_size in config.stream_specs:
            if not group or "." in group:
                raise ValueError(f"invalid embedding group name {group!r}")
            if vocab_size <= 0:
                raise ValueError(f"{group!r} has invalid vocab size {vocab_size}")
            previous_size = group_vocab_sizes.setdefault(group, vocab_size)
            if previous_size != vocab_size:
                raise ValueError(
                    f"embedding group {group!r} uses both {previous_size} "
                    f"and {vocab_size} tokens"
                )

        self.stream_embeddings = nn.ModuleDict(
            {
                group: nn.Embedding(vocab_size, config.token_embd)
                for group, vocab_size in group_vocab_sizes.items()
            }
        )
        if config.num_scalar_streams:
            self.scalar_embeddings = nn.Parameter(
                torch.empty(config.num_scalar_streams, config.token_embd)
            )
        else:
            self.register_parameter("scalar_embeddings", None)
        self.h = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.apply(self._init_weights)
        if self.scalar_embeddings is not None:
            nn.init.normal_(self.scalar_embeddings, mean=0.0, std=0.02)
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

    def _build_input(self, reads, prev_actions, scalar_inputs):
        if reads.size(-1) != self.config.num_inputs:
            raise ValueError(
                f"expected {self.config.num_inputs} input streams, "
                f"got {reads.size(-1)}"
            )
        if prev_actions.size(-1) != self.config.num_action_inputs:
            raise ValueError(
                f"expected {self.config.num_action_inputs} previous actions, "
                f"got {prev_actions.size(-1)}"
            )

        pieces = [
            self.stream_embeddings[group](reads[..., stream])
            for stream, (group, _) in enumerate(self.config.input_streams)
        ]
        pieces.extend(
            self.stream_embeddings[group](prev_actions[..., stream])
            for stream, (group, _) in enumerate(
                self.config.action_input_streams
            )
        )
        reference = pieces[0]
        if self.config.num_scalar_streams:
            expected_shape = (
                *reference.shape[:-1],
                self.config.num_scalar_streams,
            )
            if scalar_inputs is None or scalar_inputs.shape != expected_shape:
                actual_shape = (
                    None if scalar_inputs is None else tuple(scalar_inputs.shape)
                )
                raise ValueError(
                    f"expected scalar input shape {expected_shape}, "
                    f"got {actual_shape}"
                )
            scalar_inputs = scalar_inputs.to(
                device=reference.device,
                dtype=reference.dtype,
            )
            scalar_slots = (
                scalar_inputs.unsqueeze(-1) * self.scalar_embeddings
            )
            pieces.extend(scalar_slots.unbind(dim=-2))
        elif scalar_inputs is not None:
            raise ValueError(
                "scalar inputs require num_scalar_streams to be positive"
            )
        return torch.cat(pieces, dim=-1)

    def forward(
        self,
        reads,
        prev_actions,
        states=None,
        scalar_inputs=None,
    ):
        """Return tied categorical logits, scalar predictions, and states."""
        x = self._build_input(reads, prev_actions, scalar_inputs)
        new_states = []
        for layer_idx, block in enumerate(self.h):
            layer_state = None if states is None else states[layer_idx]
            x, new_state = block(x, state=layer_state)
            new_states.append(new_state)
        x = self.ln_f(x)

        slots = x.split(self.config.token_embd, dim=-1)
        token_slots = slots[: self.config.num_outputs]
        logits = tuple(
            F.linear(slot, self.stream_embeddings[group].weight)
            for slot, (group, _) in zip(
                token_slots,
                self.config.stream_specs,
            )
        )
        scalar_outputs = None
        if self.config.num_scalar_streams:
            scalar_slots = torch.stack(
                slots[self.config.num_outputs :],
                dim=-2,
            )
            scalar_outputs = (
                scalar_slots * self.scalar_embeddings
            ).sum(dim=-1)
        return logits, scalar_outputs, new_states
