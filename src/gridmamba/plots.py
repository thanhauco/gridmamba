"""Report figure: example forecast, rolling coverage, per-horizon error, expert routing."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _rolling_mean(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="valid")


def make_report(
    path: Path,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    origin: np.ndarray,
    median: np.ndarray,
    bands: dict[str, tuple[np.ndarray, np.ndarray]],
    shift_idx: int,
    alpha: float,
    horizon_mae: dict[str, np.ndarray],
    expert_usage: np.ndarray,
    example: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = {"raw quantiles": "#9aa0a6", "CQR": "#1a73e8", "ACI": "#e8710a"}

    # 1. Example day-ahead forecast after the regime shift.
    ax = axes[0, 0]
    past = X[example, -72:, 0]
    t_past = np.arange(-72, 0)
    t_fut = np.arange(Y.shape[1])
    ax.plot(t_past, past, color="black", lw=1.2, label="history")
    ax.plot(t_fut, Y[example], color="black", lw=1.2, ls="--", label="actual")
    ax.plot(t_fut, median[example], color="#188038", lw=1.8, label="median forecast")
    for name in ("CQR", "ACI"):
        lo, hi = bands[name]
        ax.fill_between(t_fut, lo[example], hi[example], color=colors[name], alpha=0.25, label=f"{name} {1 - alpha:.0%}")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_title("Day-ahead forecast after the EV-charging shift")
    ax.set_xlabel("hours from forecast origin")
    ax.set_ylabel("load (MW)")
    ax.legend(fontsize=8, loc="upper left")

    # 2. Rolling coverage through the test period.
    ax = axes[0, 1]
    keys, inv = np.unique(origin, return_inverse=True)
    days = (keys - shift_idx) / 24
    k = 7
    for name, (lo, hi) in bands.items():
        hit = ((Y >= lo) & (Y <= hi)).mean(axis=1)
        per_day = np.bincount(inv, weights=hit) / np.bincount(inv)
        ax.plot(days[k - 1 :], _rolling_mean(per_day, k), color=colors[name], lw=1.8, label=name)
    ax.axhline(1 - alpha, color="black", ls=":", lw=1, label="target")
    ax.axvspan(-7, 7, color="#fbbc04", alpha=0.15, label="EV adoption ramp")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("7-day rolling coverage on the test period")
    ax.set_xlabel("days relative to regime shift")
    ax.set_ylabel("coverage")
    ax.legend(fontsize=8, loc="lower left")

    # 3. MAE per horizon step.
    ax = axes[1, 0]
    for name, values in horizon_mae.items():
        ax.plot(np.arange(1, len(values) + 1), values, marker="o", ms=3, label=name)
    ax.set_title("Test MAE by forecast horizon")
    ax.set_xlabel("hours ahead")
    ax.set_ylabel("MAE (MW)")
    ax.legend(fontsize=8)

    # 4. MoE routing distribution.
    ax = axes[1, 1]
    share = expert_usage / expert_usage.sum(axis=1, keepdims=True)
    im = ax.imshow(share, cmap="viridis", aspect="auto", vmin=0)
    for (i, j), v in np.ndenumerate(share):
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", color="white" if v < share.max() * 0.6 else "black", fontsize=8)
    ax.set_xticks(range(share.shape[1]), [f"E{j}" for j in range(share.shape[1])])
    ax.set_yticks(range(share.shape[0]), [f"layer {i}" for i in range(share.shape[0])])
    ax.set_title("Sparse-MoE routing share (test set, top-2)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("GridMamba: selective SSM + sparse MoE + adaptive conformal prediction", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
