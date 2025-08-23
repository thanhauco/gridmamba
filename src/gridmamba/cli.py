"""End-to-end pipeline: generate data, train, calibrate, evaluate, and write a report."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from gridmamba.baselines import daily_naive, weekly_naive
from gridmamba.conformal import CQR, SplitConformal, adaptive_conformal, cqr_scores
from gridmamba.data import FEATURES, DataConfig, build_features, generate, make_splits, make_windows
from gridmamba.metrics import interval_report, mae, rmse
from gridmamba.model import GridMamba, ModelConfig
from gridmamba.plots import make_report
from gridmamba.train import TrainConfig, fit, predict


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--experts", type=int, default=6)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.1, help="target miscoverage (0.1 means 90%% intervals)")
    p.add_argument("--aci-gamma", type=float, default=0.03)
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    return p.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    context, horizon = 168, 24
    alpha = args.alpha

    # Data
    data_cfg = DataConfig()
    data = generate(data_cfg)
    splits = make_splits(data_cfg)
    feats = build_features(data, splits.train_end)
    train = make_windows(feats, 0, splits.train_end, context, horizon, args.train_stride)
    val = make_windows(feats, splits.train_end, splits.val_end, context, horizon, 24)
    cal = make_windows(feats, splits.val_end, splits.cal_end, context, horizon, 24)
    test = make_windows(feats, splits.cal_end, splits.test_end, context, horizon, 24)
    print(f"windows: train {len(train)} | val {len(val)} | cal {len(cal)} | test {len(test)}")

    # Model
    q = (alpha / 2, 0.25, 0.5, 0.75, 1 - alpha / 2)
    model_cfg = ModelConfig(
        n_features=len(FEATURES),
        context=context,
        horizon=horizon,
        quantiles=q,
        d_model=args.d_model,
        n_layers=args.layers,
        n_experts=args.experts,
        top_k=args.top_k,
    )
    model = GridMamba(model_cfg)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
    history = fit(model, train, val, train_cfg, seed=args.seed)

    # Predict
    q_cal = predict(model, cal.X, device=args.device)
    model.reset_expert_usage()
    q_test = predict(model, test.X, device=args.device)
    usage = model.expert_usage().cpu().numpy()
    lo_c, hi_c = q_cal[..., 0], q_cal[..., -1]
    lo_t, med_t, hi_t = q_test[..., 0], q_test[..., len(q) // 2], q_test[..., -1]

    # Calibrate
    cqr = CQR(alpha).fit(lo_c, hi_c, cal.Y)
    cqr_lo, cqr_hi = cqr.predict(lo_t, hi_t)
    aci_lo, aci_hi, alpha_path = adaptive_conformal(
        lo_t, hi_t, test.Y, test.origin, cqr_scores(lo_c, hi_c, cal.Y), alpha, gamma=args.aci_gamma
    )
    bands = {"raw quantiles": (lo_t, hi_t), "CQR": (cqr_lo, cqr_hi), "ACI": (aci_lo, aci_hi)}

    # Baselines, conformalized on the same calibration set
    point = {"GridMamba (median)": med_t}
    for name, fn in (("daily naive", daily_naive), ("weekly naive", weekly_naive)):
        sc = SplitConformal(alpha).fit(fn(cal.X, horizon), cal.Y)
        point[name] = fn(test.X, horizon)
        bands[f"{name} + conformal"] = sc.predict(point[name])

    # Evaluate on the full test period and on the pre- and post-shift segments
    shift = data["shift_idx"]
    segments = {
        "all": np.ones(len(test), bool),
        "pre_shift": test.origin < shift - 14 * 24,
        "post_shift": test.origin >= shift + 14 * 24,
    }
    results = {"n_test_windows": len(test), "alpha": alpha, "point": {}, "intervals": {}}
    for name, pred in point.items():
        results["point"][name] = {"mae": mae(pred, test.Y), "rmse": rmse(pred, test.Y)}
    for name, (lo, hi) in bands.items():
        results["intervals"][name] = {
            seg: interval_report(lo[m], hi[m], test.Y[m], alpha) for seg, m in segments.items()
        }
    results["history"] = history
    results["expert_usage"] = usage.tolist()
    results["cqr_margin_per_horizon"] = cqr.qhat.tolist()
    results["aci_final_alpha"] = alpha_path[-1].tolist()

    _print_tables(results)
    post = np.flatnonzero(segments["post_shift"])
    example = post[len(post) // 2] if len(post) else len(test) - 1
    make_report(
        args.out / "report.png",
        X=test.X,
        Y=test.Y,
        origin=test.origin,
        median=med_t,
        bands={k: bands[k] for k in ("raw quantiles", "CQR", "ACI")},
        shift_idx=shift,
        alpha=alpha,
        horizon_mae={k: np.abs(v - test.Y).mean(axis=0) for k, v in point.items()},
        expert_usage=usage,
        example=example,
    )
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2))
    torch.save({"config": model_cfg.__dict__, "state_dict": model.state_dict()}, args.out / "gridmamba.pt")
    print(f"\nwrote {args.out / 'metrics.json'}, {args.out / 'report.png'}, {args.out / 'gridmamba.pt'}")
    return results


def _print_tables(results: dict) -> None:
    print("\nPoint forecasts (test)")
    print(f"  {'model':<22}{'MAE':>9}{'RMSE':>9}")
    for name, m in results["point"].items():
        print(f"  {name:<22}{m['mae']:>9.3f}{m['rmse']:>9.3f}")
    target = 1 - results["alpha"]
    print(f"\nIntervals (target coverage {target:.0%}): coverage / width / interval score")
    print(f"  {'method':<28}{'all':>24}{'pre-shift':>24}{'post-shift':>24}")
    for name, segs in results["intervals"].items():
        cells = "".join(
            f"{s['coverage']:>8.1%} {s['width']:>6.2f} {s['interval_score']:>7.2f}" for s in segs.values()
        )
        print(f"  {name:<28}{cells}")


if __name__ == "__main__":
    main()
