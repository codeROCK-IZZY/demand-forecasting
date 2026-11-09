from __future__ import annotations

import numpy as np
import pytest

from forecast import backtest as bt
from forecast import data, models, reconcile
from forecast.evaluate import conformal_level


@pytest.fixture(scope="module")
def chain() -> data.Chain:
    return data.generate(seed=5)


@pytest.fixture(scope="module")
def hz(chain: data.Chain) -> bt.Hierarchy:
    return bt.build(chain)


def test_the_hierarchy_adds_up(chain: data.Chain, hz: bt.Hierarchy) -> None:
    assert hz.s.shape == (136, 120)
    day = 800
    assert hz.y[0, day] == pytest.approx(np.nansum(chain.sales[:, day]))
    assert hz.y[1:4, day].sum() == pytest.approx(hz.y[0, day])


def test_late_store_has_no_history_before_it_opens(chain: data.Chain, hz: bt.Hierarchy) -> None:
    late = chain.store_of == 5
    assert np.isnan(chain.sales[late, :364]).all()
    assert not np.isnan(chain.sales[late, 400]).any()
    assert np.isnan(hz.y[hz.labels.index("store 6"), 100])


def test_promotions_lift_sales(chain: data.Chain) -> None:
    on = chain.sales[chain.promo > 0]
    off = chain.sales[(chain.promo == 0) & ~np.isnan(chain.sales)]
    assert np.nanmean(on) > 1.2 * np.nanmean(off)


def test_seasonal_naive_repeats_last_week() -> None:
    y = np.arange(30, dtype=float)[None, :]
    f = models.seasonal_naive(y, 21, 10)
    assert f[0].tolist() == [14, 15, 16, 17, 18, 19, 20, 14, 15, 16]


def test_holt_winters_learns_a_weekly_pattern() -> None:
    pattern = np.array([10, 10, 10, 10, 20, 30, 15], float)
    y = np.tile(pattern, 60)[None, :]
    f = models.holt_winters(y, 400, 14)
    assert np.abs(f[0] - np.tile(pattern, 2)[[(400 + k) % 7 for k in range(14)]]).max() < 1.0


def test_features_use_only_the_past(hz: bt.Hierarchy) -> None:
    origin = 700
    x1, _ = models.features(hz.y, hz.cal, origin, 7)
    y2 = hz.y.copy()
    y2[:, origin:] = 1e6  # change the future
    x2, _ = models.features(y2, hz.cal, origin, 7)
    assert np.array_equal(np.nan_to_num(x1), np.nan_to_num(x2))


def test_reconciliation_is_coherent_and_keeps_coherent_forecasts(hz: bt.Hierarchy) -> None:
    rng = np.random.default_rng(0)
    bottom = rng.random((120, 5)) * 10
    coherent = hz.s @ bottom
    noisy = coherent + rng.normal(0, 1, coherent.shape)
    w = reconcile.shrunk_covariance(rng.normal(size=(136, 300)))
    rec = reconcile.mint(hz.s, noisy, w)
    assert np.allclose(hz.s @ rec[-120:], rec)
    assert np.allclose(reconcile.mint(hz.s, coherent, w), coherent)
    assert np.allclose(reconcile.bottom_up(hz.s, noisy, 120), hz.s @ noisy[-120:])
    clipped = reconcile.nonnegative(hz.s, rec - 50, 120)
    assert (clipped >= 0).all() and np.allclose(hz.s @ clipped[-120:], clipped)


def test_shrinkage_gives_a_usable_covariance() -> None:
    rng = np.random.default_rng(1)
    w = reconcile.shrunk_covariance(rng.normal(size=(50, 20)))  # more series than samples
    assert np.all(np.linalg.eigvalsh(w) > 0)


def test_conformal_level_is_a_finite_sample_quantile() -> None:
    scores = np.arange(1, 101, dtype=float)
    assert conformal_level(scores, 0.8) == 81.0
