"""Conformal calibration for multi-horizon interval forecasts.

* ``SplitConformal``: symmetric intervals around point forecasts, from absolute residuals.
* ``CQR``: Conformalized Quantile Regression (Romano et al., 2019). It widens or
  narrows the model's own quantile band by a calibrated per-horizon margin.
"""

from __future__ import annotations

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

