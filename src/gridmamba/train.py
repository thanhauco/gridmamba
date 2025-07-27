"""Training loop: AdamW, warmup + cosine schedule, early stopping on validation pinball loss."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from gridmamba.data import Windows
from gridmamba.metrics import pinball_loss
from gridmamba.model import GridMamba


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    balance_coef: float = 1e-2
    z_coef: float = 1e-3
    grad_clip: float = 1.0
    patience: int = 5
    device: str = "cpu"


def _loss(model: GridMamba, out: dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    # Pinball loss in RevIN-normalized units, so every region counts the same whatever its scale.
    y_norm = (y - out["loc"]) / out["scale"]
    return pinball_loss(out["q_norm"], y_norm, model.quantiles)


def _batches(n: int, batch_size: int, shuffle: bool, rng: np.random.Generator):
    order = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, batch_size):
        yield order[i : i + batch_size]


@torch.no_grad()
def predict(model: GridMamba, X: np.ndarray, batch_size: int = 1024, device: str = "cpu") -> np.ndarray:
    """Return quantile forecasts in original units, shape (N, horizon, Q)."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        out.append(model(xb)["quantiles"].cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def evaluate_loss(model: GridMamba, win: Windows, cfg: TrainConfig) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(win), 1024):
        xb = torch.from_numpy(win.X[i : i + 1024]).to(cfg.device)
        yb = torch.from_numpy(win.Y[i : i + 1024]).to(cfg.device)
        total += _loss(model, model(xb), yb).item() * len(yb)
        count += len(yb)
    return total / count


def fit(model: GridMamba, train: Windows, val: Windows, cfg: TrainConfig, seed: int = 0, log=print) -> list[dict]:
    rng = np.random.default_rng(seed)
    model.to(cfg.device)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "A_log" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2 or "A_log" in n]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
    )
    steps_per_epoch = math.ceil(len(train) / cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    warmup = max(1, int(cfg.warmup_frac * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warmup
        if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))),
    )

    X = torch.from_numpy(train.X)
    Y = torch.from_numpy(train.Y)
    best, best_state, bad_epochs, history = math.inf, None, 0, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running, balance_sum, n_batches = 0.0, 0.0, 0
        for idx in _batches(len(train), cfg.batch_size, True, rng):
            xb, yb = X[idx].to(cfg.device), Y[idx].to(cfg.device)
            out = model(xb)
            task = _loss(model, out, yb)
            loss = task + cfg.balance_coef * out["balance"] + cfg.z_coef * out["z"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            running += task.item()
            balance_sum += out["balance"].item()
            n_batches += 1

        val_loss = evaluate_loss(model, val, cfg)
        record = {
            "epoch": epoch,
            "train_pinball": running / n_batches,
            "val_pinball": val_loss,
            "balance": balance_sum / n_batches,
            "seconds": time.perf_counter() - t0,
        }
        history.append(record)
        improved = val_loss < best - 1e-5
        log(
            f"epoch {epoch:3d} | train {record['train_pinball']:.4f} | val {val_loss:.4f} | "
            f"balance {record['balance']:.3f} | {record['seconds']:.1f}s{'  *' if improved else ''}"
        )
        if improved:
            best, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                log(f"early stopping after {epoch} epochs (best val {best:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
