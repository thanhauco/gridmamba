"""Seasonal-naive baselines computed directly from the context window."""

from __future__ import annotations

import numpy as np


def daily_naive(X: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed 24 hours (load at t0 - 24 + h)."""
    return X[:, -24 : -24 + horizon or None, 0]


def weekly_naive(X: np.ndarray, horizon: int) -> np.ndarray:
    """Same hour one week ago (load at t0 - 168 + h). Requires context >= 168."""
    start = X.shape[1] - 168
    return X[:, start : start + horizon, 0]
