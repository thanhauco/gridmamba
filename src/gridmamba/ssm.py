"""Mamba-style selective state-space block in pure PyTorch.

The recurrence h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * u_t is input-dependent
("selective"): dt, B and C are all projected from the input at every step.
Two scan backends compute the same thing:

* ``sequential_scan``: a reference O(L) Python loop.
* ``parallel_scan``: a Hillis-Steele associative scan with O(log L) depth, using
  the operator (a1, b1) . (a2, b2) = (a1 * a2, a2 * b1 + b2).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sequential_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve h_t = a_t * h_{t-1} + b_t along dim 1 with h_{-1} = 0."""
    h = torch.zeros_like(b[:, 0])
    out = []
    for t in range(b.shape[1]):
        h = a[:, t] * h + b[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


def parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Same result as ``sequential_scan`` in ceil(log2 L) vectorized steps."""
    length = b.shape[1]
    offset = 1
    while offset < length:
        b = torch.cat([b[:, :offset], a[:, offset:] * b[:, :-offset] + b[:, offset:]], dim=1)
        a = torch.cat([a[:, :offset], a[:, offset:] * a[:, :-offset]], dim=1)
        offset *= 2
    return b


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    parallel: bool = True,
) -> torch.Tensor:
    """Discretize with zero-order hold on A (Euler on B) and run the scan.

    Shapes: u, delta (batch, L, E); A (E, N); B, C (batch, L, N); D (E,).
    """
    dA = torch.exp(delta.unsqueeze(-1) * A)  # (batch, L, E, N)
    dBu = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
    h = (parallel_scan if parallel else sequential_scan)(dA, dBu)
    y = (h * C.unsqueeze(2)).sum(-1)
    return y + u * D


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        parallel: bool = True,
    ):
        super().__init__()
        d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.parallel = parallel

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # dt bias = softplus^-1 of a log-uniform sample in [dt_min, dt_max], as in Mamba.
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        # S4D-real initialization: A_n = -(n + 1).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv1d(u.transpose(1, 2))[..., :length].transpose(1, 2)  # causal depthwise conv
        u = F.silu(u)
        dt, B, C = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        y = selective_scan(u, delta, A, B, C, self.D, parallel=self.parallel)
        return self.out_proj(y * F.silu(z))
