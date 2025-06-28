import numpy as np

from gridmamba.conformal import CQR, SplitConformal, adaptive_conformal, cqr_scores


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


def test_aci_recovers_coverage_after_shift():
    rng = np.random.default_rng(2)
    n_cal, n_test, alpha = 500, 3000, 0.1
    lo, hi = -np.ones((n_test, 1)), np.ones((n_test, 1))
    cal_y = rng.normal(size=(n_cal, 1))
    test_y = rng.normal(size=(n_test, 1)) * np.where(np.arange(n_test) < 500, 1.0, 3.0)[:, None]
    cal_scores = cqr_scores(-np.ones((n_cal, 1)), np.ones((n_cal, 1)), cal_y)

    static_lo, static_hi = CQR(alpha).fit(-np.ones((n_cal, 1)), np.ones((n_cal, 1)), cal_y).predict(lo, hi)
    aci_lo, aci_hi, path = adaptive_conformal(
        lo, hi, test_y, np.arange(n_test), cal_scores, alpha, gamma=0.01, window=200
    )
    late = slice(1500, None)
    static_cov = ((test_y[late] >= static_lo[late]) & (test_y[late] <= static_hi[late])).mean()
    aci_cov = ((test_y[late] >= aci_lo[late]) & (test_y[late] <= aci_hi[late])).mean()
    assert static_cov < 0.6
    assert abs(aci_cov - (1 - alpha)) < 0.03
    assert path.shape == (n_test, 1)
