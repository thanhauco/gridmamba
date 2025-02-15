"""Synthetic multi-region hourly electricity load with a late regime shift.

Each region has its own base load and climate. Load depends on a daily profile
(morning and evening peaks), a weekend dip, a U-shaped temperature response
(heating and cooling), a slow trend, and heteroscedastic AR(1) noise.

Near the end of the timeline, EV-charging adoption ramps in. It adds a
late-night peak and extra volatility that never appear in training data, so the
test period has a real distribution shift for adaptive conformal prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURES = ("load", "temp", "hour_sin", "hour_cos", "dow_sin", "dow_cos")


@dataclass(frozen=True)
class DataConfig:
    n_regions: int = 8
    n_days: int = 730
    shift_at: float = 0.90  # fraction of the timeline where EV adoption is half-way
    seed: int = 7


def _circ_bump(hour: np.ndarray, center: float, width: float) -> np.ndarray:
    d = np.abs(hour - center)
    d = np.minimum(d, 24 - d)
    return np.exp(-(d**2) / (2 * width**2))


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    eps = rng.normal(0.0, sigma, n)
    out = np.empty(n)
    out[0] = eps[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def generate(cfg: DataConfig) -> dict[str, np.ndarray | int]:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_days * 24
    t = np.arange(n)
    hour = t % 24
    dow = (t // 24) % 7
    doy = (t / 24) % 365.25
    shift_idx = int(cfg.shift_at * cfg.n_days) * 24
    ramp = 1.0 / (1.0 + np.exp(-(t - shift_idx) / (24 * 7)))

    loads = np.empty((cfg.n_regions, n))
    temps = np.empty((cfg.n_regions, n))
    for r in range(cfg.n_regions):
        base = rng.uniform(40, 160)
        climate = rng.uniform(-6, 6)
        temp = (
            14
            + climate
            + 11 * np.sin(2 * np.pi * (doy - 110) / 365.25)
            + 4 * np.sin(2 * np.pi * (hour - 9) / 24)
            + _ar1(rng, n, 0.985, 0.45)
        )
        profile = (
            0.70
            + 0.18 * _circ_bump(hour, 8, 1.8)
            + 0.30 * _circ_bump(hour, 19, 2.2)
            - 0.12 * _circ_bump(hour, 3, 2.5)
        )
        weekend = np.where(dow >= 5, 0.86, 1.0)
        thermal = 0.0025 * np.maximum(temp - 21, 0) ** 2 + 0.0015 * np.maximum(13 - temp, 0) ** 2
        trend = 1 + 0.06 * t / n
        ev = ramp * rng.uniform(0.18, 0.32) * _circ_bump(hour, 23, 1.5)
        mean_load = base * trend * weekend * (profile + thermal + ev)
        vol = 0.025 + 0.02 * _circ_bump(hour, 19, 3) + 0.04 * ramp
        loads[r] = mean_load * (1 + vol * _ar1(rng, n, 0.6, 0.8))
        temps[r] = temp

    return {"load": loads, "temp": temps, "hour": hour, "dow": dow, "shift_idx": shift_idx}

