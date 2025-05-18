"""GridMamba: patched Mamba + sparse-MoE encoder with a non-crossing quantile head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from gridmamba.moe import SparseMoE
from gridmamba.ssm import MambaBlock


@dataclass
class ModelConfig:
    n_features: int = 6
    context: int = 168
    horizon: int = 24
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
    d_model: int = 64
    n_layers: int = 3
    patch_len: int = 6
    d_state: int = 16
    n_experts: int = 6
    top_k: int = 2
    dropout: float = 0.1


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class HybridLayer(nn.Module):
    """Pre-norm residual layer: selective SSM for token mixing, sparse MoE for channel mixing."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm_ssm = RMSNorm(cfg.d_model)
        self.ssm = MambaBlock(cfg.d_model, d_state=cfg.d_state)
        self.norm_moe = RMSNorm(cfg.d_model)
        self.moe = SparseMoE(cfg.d_model, n_experts=cfg.n_experts, top_k=cfg.top_k)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = h + self.drop(self.ssm(self.norm_ssm(h)))
        moe_out, aux = self.moe(self.norm_moe(h))
        return h + self.drop(moe_out), aux


def monotone_quantiles(raw: torch.Tensor, min_gap: float = 1e-4) -> torch.Tensor:
    """Map unconstrained outputs to sorted quantiles around the median.

    The center channel is the median. Other quantiles are offset from it by
    cumulative softplus increments, so the quantiles can never cross. ``min_gap``
    keeps them strictly ordered even where softplus underflows to zero.
    """
    m = raw.shape[-1] // 2
    mid = raw[..., m : m + 1]
    up = mid + torch.cumsum(F.softplus(raw[..., m + 1 :]) + min_gap, dim=-1)
    down = mid - torch.cumsum(F.softplus(raw[..., :m].flip(-1)) + min_gap, dim=-1).flip(-1)
    return torch.cat([down, mid, up], dim=-1)


class GridMamba(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        q = cfg.quantiles
        assert cfg.context % cfg.patch_len == 0, "context must be divisible by patch_len"
        assert len(q) % 2 == 1 and q[len(q) // 2] == 0.5, "quantiles must be odd-length and centered on 0.5"
        self.cfg = cfg
        self.n_patches = cfg.context // cfg.patch_len
        self.patch_embed = nn.Linear(cfg.patch_len * cfg.n_features, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, cfg.d_model) * 0.02)
        self.layers = nn.ModuleList(HybridLayer(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(cfg.dropout),
            nn.Linear(self.n_patches * cfg.d_model, cfg.horizon * len(q)),
        )
        self.register_buffer("quantiles", torch.tensor(q), persistent=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (batch, context, n_features), with raw load in channel 0."""
        batch = x.shape[0]
        # RevIN: normalize the target channel per instance, then undo it on the output.
        load = x[..., 0]
        loc = load.mean(dim=1, keepdim=True)
        scale = torch.sqrt(load.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = torch.cat([((load - loc) / scale).unsqueeze(-1), x[..., 1:]], dim=-1)

        h = self.patch_embed(x.reshape(batch, self.n_patches, -1)) + self.pos
        balance = h.new_zeros(())
        z_loss = h.new_zeros(())
        for layer in self.layers:
            h, aux = layer(h)
            balance = balance + aux["balance"]
            z_loss = z_loss + aux["z"]

        raw = self.head(self.norm(h)).view(batch, self.cfg.horizon, -1)
        q_norm = monotone_quantiles(raw)
        return {
            "q_norm": q_norm,
            "quantiles": q_norm * scale.unsqueeze(-1) + loc.unsqueeze(-1),
            "loc": loc,
            "scale": scale,
            "balance": balance / len(self.layers),
            "z": z_loss / len(self.layers),
        }

    def expert_usage(self) -> torch.Tensor:
        """Routing counts accumulated in eval mode, shape (n_layers, n_experts)."""
        return torch.stack([layer.moe.usage for layer in self.layers])

    def reset_expert_usage(self) -> None:
        for layer in self.layers:
            layer.moe.usage.zero_()
