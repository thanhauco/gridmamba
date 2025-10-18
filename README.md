# GridMamba

Probabilistic day-ahead electricity-load forecasting built from recent sequence-modeling and uncertainty-quantification techniques, all in plain PyTorch:

| Component | Technique | File |
|---|---|---|
| Token mixing | Mamba-style **selective state-space model**: input-dependent Δ, B, C; ZOH discretization; S4D-real init | [`ssm.py`](src/gridmamba/ssm.py) |
| Scan | **Parallel associative scan** (Hillis–Steele, O(log L) depth), tested against a sequential reference | [`ssm.py`](src/gridmamba/ssm.py) |
| Channel mixing | **Sparse Mixture-of-Experts**, top-2 routing over SwiGLU experts, Switch load-balancing loss, ST-MoE router z-loss | [`moe.py`](src/gridmamba/moe.py) |
| Input | **PatchTST-style patching** plus **RevIN** instance normalization, for scale-equivariant forecasts | [`model.py`](src/gridmamba/model.py) |
| Output | **Non-crossing quantile head** (median plus cumulative-softplus offsets), trained with pinball loss | [`model.py`](src/gridmamba/model.py) |
| Calibration | **Conformalized Quantile Regression**, with finite-sample guarantee under exchangeability | [`conformal.py`](src/gridmamba/conformal.py) |
| Drift | **Adaptive Conformal Inference**, online miscoverage control under distribution shift | [`conformal.py`](src/gridmamba/conformal.py) |

## The task

Eight synthetic regions of hourly load over two years ([`data.py`](src/gridmamba/data.py)). Load is driven by daily and weekly profiles, a U-shaped temperature response, trend, and heteroscedastic AR(1) noise. Each forecast issued at midnight predicts the next 24 hours from the past 168 hours.

Near the end of the timeline, **EV-charging adoption** ramps in. It adds a late-night peak and roughly doubles volatility. The model never sees this in training or calibration, so the test period has a real distribution shift.

```
|------ train 60% ------|- val 10% -|--- calibration 15% ---|------ test 15% ------|
                                                                   ^ EV shift
```

## Quick start

```bash
uv sync
uv run pytest
uv run gridmamba --epochs 20
```

A run takes about 3 minutes on an Apple-silicon GPU (`--device auto` picks CUDA, then MPS, then CPU). It writes `outputs/metrics.json`, `outputs/report.png` and a checkpoint.

## Project layout

```
src/gridmamba/
  data.py        synthetic multi-region load generator, splits, windowing
  ssm.py         selective scan (parallel + sequential) and MambaBlock
  moe.py         top-k sparse MoE with auxiliary losses
  model.py       RevIN -> patch embed -> [Mamba + MoE] x N -> quantile head
  conformal.py   split conformal, CQR, adaptive conformal inference
  metrics.py     pinball, MAE/RMSE, coverage, width, interval score
  baselines.py   daily / weekly seasonal naive
  train.py       AdamW + warmup/cosine, early stopping
  plots.py       report figure
  cli.py         end-to-end pipeline
tests/           scan equivalence, causality, RevIN equivariance, conformal coverage
```

## References

- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023)
- Fedus, Zoph & Shazeer, *Switch Transformers* (2021); Zoph et al., *ST-MoE* (2022)
- Nie et al., *A Time Series is Worth 64 Words (PatchTST)* (2023)
- Kim et al., *Reversible Instance Normalization for Accurate Time-Series Forecasting* (2022)
- Romano, Patterson & Candès, *Conformalized Quantile Regression* (2019)
- Gibbs & Candès, *Adaptive Conformal Inference Under Distribution Shift* (2021)
