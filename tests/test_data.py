import numpy as np

from gridmamba.data import DataConfig, build_features, generate, make_splits, make_windows


def _small():
    cfg = DataConfig(n_regions=2, n_days=60, seed=3)
    data = generate(cfg)
    splits = make_splits(cfg)
    return cfg, data, splits, build_features(data, splits.train_end)


def test_generated_load_is_positive_and_shaped():
    cfg, data, _, feats = _small()
    assert data["load"].shape == (2, 60 * 24)
    assert (data["load"] > 0).all()
    assert feats.shape == (2, 60 * 24, 6)


def test_windows_align_context_and_targets():
    _, data, splits, feats = _small()
    win = make_windows(feats, splits.val_end, splits.test_end, 168, 24, 24)
    assert (win.origin % 24 == 0).all()
    assert (np.diff(win.origin) >= 0).all()
    i = 3
    r, t0 = win.region[i], win.origin[i]
    np.testing.assert_allclose(win.Y[i], data["load"][r, t0 : t0 + 24], rtol=1e-6)
    np.testing.assert_allclose(win.X[i, -1, 0], data["load"][r, t0 - 1], rtol=1e-6)
