"""Conformal calibration for multi-horizon interval forecasts.

* ``SplitConformal``: symmetric intervals around point forecasts, from absolute residuals.
* ``CQR``: Conformalized Quantile Regression (Romano et al., 2019). It widens or
  narrows the model's own quantile band by a calibrated per-horizon margin.
* ``adaptive_conformal``: Adaptive Conformal Inference (Gibbs & Candes, 2021).
  It tunes the miscoverage level online so long-run coverage tracks the target
  under distribution shift. Nonconformity scores come from a rolling window.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> np.ndarray:
    """Finite-sample-corrected (1 - alpha) quantile along axis 0."""
    n = scores.shape[0]
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(scores, level, axis=0, method="higher")


def cqr_scores(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(lo - y, y - hi)


@dataclass
class SplitConformal:
    alpha: float
    qhat: np.ndarray | None = field(default=None, init=False)

    def fit(self, pred: np.ndarray, y: np.ndarray) -> SplitConformal:
        self.qhat = conformal_quantile(np.abs(y - pred), self.alpha)
        return self

    def predict(self, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return pred - self.qhat, pred + self.qhat


@dataclass
class CQR:
    alpha: float
    qhat: np.ndarray | None = field(default=None, init=False)

    def fit(self, lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> CQR:
        self.qhat = conformal_quantile(cqr_scores(lo, hi, y), self.alpha)
        return self

    def predict(self, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return lo - self.qhat, hi + self.qhat


def adaptive_conformal(
    lo: np.ndarray,
    hi: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cal_scores: np.ndarray,
    alpha: float,
    gamma: float = 0.03,
    window: int = 240,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Online CQR with an adaptive miscoverage level, one level per horizon step.

    Rows sharing a ``groups`` key (a forecast origin) are predicted together. Their
    outcomes are revealed before the next key, so ``groups`` must be sorted and
    spaced at least one horizon apart. The quantile level is clipped to [0, 1],
    which caps interval width at the widest recent score instead of going infinite.

    Returns adjusted (lo, hi) and the trajectory of alpha_t, shape (n_groups, H).
    """
    n_h = y.shape[1]
    alpha_t = np.full(n_h, alpha, dtype=float)
    history = [deque(cal_scores[-window:, h], maxlen=window) for h in range(n_h)]
    out_lo, out_hi = np.empty_like(lo), np.empty_like(hi)
    trajectory = []

    keys, starts = np.unique(groups, return_index=True)
    bounds = list(starts) + [len(groups)]
    for i in range(len(keys)):
        rows = slice(bounds[i], bounds[i + 1])
        levels = np.clip(1 - alpha_t, 0.0, 1.0)
        qhat = np.array([np.quantile(np.fromiter(history[h], float), levels[h], method="higher") for h in range(n_h)])
        out_lo[rows], out_hi[rows] = lo[rows] - qhat, hi[rows] + qhat
        trajectory.append(alpha_t.copy())

        # Outcomes are revealed: update the miscoverage level and the score window.
        miss = (y[rows] < out_lo[rows]) | (y[rows] > out_hi[rows])
        alpha_t += gamma * (alpha - miss.mean(axis=0))
        for h, scores in enumerate(cqr_scores(lo[rows], hi[rows], y[rows]).T):
            history[h].extend(scores)

    return out_lo, out_hi, np.asarray(trajectory)
