import numpy as np

from gridmamba.conformal import CQR, SplitConformal


def test_cqr_reaches_target_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    n, h, alpha = 4000, 3, 0.1
    y = rng.standard_t(df=3, size=(2 * n, h))
    lo, hi = np.full_like(y, -0.5), np.full_like(y, 0.5)  # deliberately too narrow
    cqr = CQR(alpha).fit(lo[:n], hi[:n], y[:n])
    new_lo, new_hi = cqr.predict(lo[n:], hi[n:])
    cov = ((y[n:] >= new_lo) & (y[n:] <= new_hi)).mean()
    assert 0.88 <= cov <= 0.93


def test_split_conformal_symmetric_intervals():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(3000, 2))
    sc = SplitConformal(0.2).fit(np.zeros((1500, 2)), y[:1500])
    lo, hi = sc.predict(np.zeros((1500, 2)))
    cov = ((y[1500:] >= lo) & (y[1500:] <= hi)).mean()
    assert 0.77 <= cov <= 0.83
    np.testing.assert_allclose(-lo, hi)

