import numpy as np
import pytest

from volsurf.pricing import black76


@pytest.fixture
def grid():
    T, k, sigma = np.meshgrid(
        np.array([7, 30, 90, 365, 730]) / 365.0,
        np.linspace(-1.0, 0.6, 33),
        np.array([0.05, 0.15, 0.3, 0.6, 1.2]),
        indexing="ij",
    )
    F = 4000.0
    return F, F * np.exp(k), T, sigma


def test_put_call_parity(grid):
    F, K, T, sigma = grid
    D = 0.97
    c = black76.price(F, K, T, sigma, True, D)
    p = black76.price(F, K, T, sigma, False, D)
    np.testing.assert_allclose(c - p, D * (F - K), atol=1e-9 * F)


@pytest.mark.parametrize("is_call", [True, False])
def test_implied_vol_round_trip(grid, is_call):
    F, K, T, sigma = grid
    D = 0.98
    prices = black76.price(F, K, T, sigma, is_call, D)
    # Only test where the time value is resolvable in double precision.
    otm = black76._otm_price(F, K, T, sigma)
    resolvable = otm / F > 1e-10
    iv = black76.implied_vol(prices, F, K, T, is_call, D)
    err = np.abs(iv - sigma)[resolvable]
    assert resolvable.mean() > 0.6
    assert np.all(np.isfinite(err))
    assert err.max() < 1e-8


def test_implied_vol_rejects_arbitrage_prices():
    F, K, T = 100.0, np.array([90.0, 100.0, 110.0]), 0.5
    below_intrinsic = np.array([9.9, 0.0, 0.0])
    above_upper = np.array([100.0, 100.0, 100.0])
    assert np.all(np.isnan(black76.implied_vol(below_intrinsic, F, K, T, True)))
    assert np.all(np.isnan(black76.implied_vol(above_upper, F, K, T, True)))


def test_vega_matches_finite_difference():
    F, K, T, sigma, h = 4000.0, 3800.0, 0.25, 0.2, 1e-6
    fd = (black76.price(F, K, T, sigma + h) - black76.price(F, K, T, sigma - h)) / (2 * h)
    assert black76.vega(F, K, T, sigma) == pytest.approx(fd, rel=1e-7)
