"""Sparse top-k Mixture-of-Experts feed-forward layer.

Each token is routed to its ``top_k`` highest-probability experts, and their
outputs are combined with renormalized gate weights. A Switch-style
load-balancing loss, E * sum_e f_e * P_e, keeps routing healthy. Here f_e is
the fraction of routed slots and P_e the mean router probability of expert e.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.w_up = nn.Linear(d_model, d_hidden, bias=False)
        self.w_down = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class SparseMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_experts: int = 6,
        top_k: int = 2,
        hidden_mult: int = 2,
        router_jitter: float = 0.01,
    ):
        super().__init__()
        assert 1 <= top_k <= n_experts
        self.n_experts = n_experts
        self.top_k = top_k
        self.router_jitter = router_jitter
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(d_model, hidden_mult * d_model) for _ in range(n_experts))
        self.register_buffer("usage", torch.zeros(n_experts), persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        flat = x.reshape(-1, x.shape[-1])
        if self.training and self.router_jitter > 0:
            flat_in = flat * torch.empty_like(flat).uniform_(1 - self.router_jitter, 1 + self.router_jitter)
        else:
            flat_in = flat
        logits = self.router(flat_in)
        probs = logits.softmax(dim=-1)
        gate, idx = probs.topk(self.top_k, dim=-1)
        gate = gate / gate.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            token, slot = (idx == e).nonzero(as_tuple=True)
            if token.numel() == 0:
                continue
            out.index_add_(0, token, expert(flat[token]) * gate[token, slot].unsqueeze(-1))

        counts = F.one_hot(idx, self.n_experts).sum(dim=(0, 1)).float()
        frac = counts / counts.sum()
        balance = self.n_experts * (frac * probs.mean(dim=0)).sum()
        if not self.training:
            self.usage += counts.detach()
        return out.view_as(x), {"balance": balance}
