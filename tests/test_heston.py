import numpy as np
import pytest
from scipy.integrate import quad

from volsurf.pricing import black76, heston

# Fang & Oosterlee (2008), Section 5.2: reference call price 5.785155450.
FO_PARAMS = heston.HestonParams(kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711, v0=0.0175)


def lewis_call(F, K, T, cf):
    """Undiscounted call via the Lewis (2001) single-integral representation."""
    x = np.log(F / K)

    def integrand(u):
        return (np.exp(1j * u * x) * cf(u - 0.5j)).real / (u**2 + 0.25)

    integral, _ = quad(integrand, 0.0, np.inf, limit=500, epsabs=1e-13, epsrel=1e-12)
    return F - np.sqrt(F * K) / np.pi * integral


def test_fang_oosterlee_reference():
    price = heston.cos_price(100.0, 100.0, 1.0, FO_PARAMS, is_call=True)[0]
    assert price == pytest.approx(5.785155450, abs=5e-8)


def test_martingale():
    assert heston.char_func(-1j, 1.0, FO_PARAMS) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("T", [7 / 365, 0.25, 1.0, 2.0])
def test_cos_matches_lewis_integral(T):
    F = 4000.0
    p = heston.HestonParams(kappa=2.0, theta=0.04, sigma=0.6, rho=-0.7, v0=0.03)
    std = np.sqrt(heston.expected_total_variance(T, p))
    K = F * np.exp(np.linspace(-3.0, 2.0, 9) * std)
    cos = heston.cos_price(F, K, T, p, is_call=True)
    ref = np.array([lewis_call(F, k, T, lambda u: heston.char_func(u, T, p)) for k in K])
    np.testing.assert_allclose(cos, ref, atol=1e-9 * F)


def test_lewis_reproduces_black76():
    F, K, T, s = 4000.0, 3700.0, 0.5, 0.22
    ref = lewis_call(F, K, T, lambda u: np.exp(-0.5 * s**2 * T * (1j * u + u**2)))
    assert ref == pytest.approx(black76.price(F, K, T, s), rel=1e-9)


def test_vanishing_vol_of_vol_gives_flat_smile():
    p = heston.HestonParams(kappa=1.0, theta=0.04, sigma=1e-4, rho=0.0, v0=0.04)
    iv = heston.implied_vol_slice(100.0, np.linspace(80, 120, 9), 0.5, p)
    np.testing.assert_allclose(iv, 0.2, atol=1e-5)


def test_negative_correlation_gives_downward_skew():
    p = heston.HestonParams(kappa=2.0, theta=0.04, sigma=0.6, rho=-0.7, v0=0.04)
    iv = heston.implied_vol_slice(100.0, np.array([85.0, 100.0, 115.0]), 0.5, p)
    assert iv[0] > iv[1] > iv[2]
