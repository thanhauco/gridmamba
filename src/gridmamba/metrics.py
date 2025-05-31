"""Point, quantile and interval forecast metrics."""

from __future__ import annotations

import numpy as np
import torch


def pinball_loss(q_pred: torch.Tensor, y: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """Mean quantile (pinball) loss. q_pred: (..., Q), y: (...)."""
    diff = y.unsqueeze(-1) - q_pred
    return torch.maximum(quantiles * diff, (quantiles - 1) * diff).mean()


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(pred - y).mean())


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(((pred - y) ** 2).mean()))


def coverage(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> float:
    return float(((y >= lo) & (y <= hi)).mean())


def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float((hi - lo).mean())


def interval_score(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, alpha: float) -> float:
    """Winkler / interval score: width plus 2/alpha times the miss distance. Lower is better."""
    below = (lo - y) * (y < lo)
    above = (y - hi) * (y > hi)
    return float(((hi - lo) + 2.0 / alpha * (below + above)).mean())


def interval_report(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, float]:
    return {
        "coverage": coverage(lo, hi, y),
        "width": mean_width(lo, hi),
        "interval_score": interval_score(lo, hi, y, alpha),
    }
