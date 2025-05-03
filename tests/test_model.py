import torch

from gridmamba.model import GridMamba, ModelConfig
from gridmamba.moe import SparseMoE


def test_moe_routes_top_k_and_backprops():
    torch.manual_seed(0)
    moe = SparseMoE(d_model=16, n_experts=4, top_k=2)
    x = torch.randn(3, 10, 16, requires_grad=True)
    out, aux = moe(x)
    assert out.shape == x.shape
    assert aux["balance"].item() >= 0.99  # the minimum, 1.0, is reached under perfectly uniform routing
    (out.sum() + aux["balance"] + aux["z"]).backward()
    assert x.grad is not None
    assert moe.router.weight.grad is not None


def test_moe_tracks_usage_in_eval():
    moe = SparseMoE(d_model=8, n_experts=3, top_k=2).eval()
    moe(torch.randn(2, 5, 8))
    assert moe.usage.sum().item() == 2 * 5 * 2


def test_forward_shapes_and_scale_equivariance():
    torch.manual_seed(0)
    cfg = ModelConfig(context=48, horizon=12, patch_len=6, d_model=32, n_layers=2, n_experts=4)
    model = GridMamba(cfg).eval()
    x = torch.randn(4, 48, cfg.n_features)
    x[..., 0] = x[..., 0].abs() * 5 + 50
    out = model(x)
    assert out["quantiles"].shape == (4, 12, 5)
    # With RevIN, scaling and shifting the load channel transforms the forecast the same way.
    x2 = x.clone()
    x2[..., 0] = x[..., 0] * 3 + 100
    torch.testing.assert_close(model(x2)["quantiles"], out["quantiles"] * 3 + 100, rtol=1e-4, atol=1e-3)
