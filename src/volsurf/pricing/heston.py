"""Heston model: characteristic function and COS-method European option prices.

Everything is expressed on the forward: the characteristic function is that of ``ln(F_T / F_0)``
under the T-forward measure, so rates and dividends enter only through ``F`` and ``D``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from volsurf.pricing import black76


@dataclass(frozen=True)
class HestonParams:
    kappa: float  # mean-reversion speed of variance
    theta: float  # long-run variance
    sigma: float  # volatility of variance
    rho: float  # spot-variance correlation
    v0: float  # initial variance

    @property
    def feller(self) -> bool:
        return 2.0 * self.kappa * self.theta >= self.sigma**2


def char_func(u, T: float, p: HestonParams):
    """Characteristic function of ``ln(F_T / F_0)`` (Albrecher et al. "little trap" form)."""
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    beta = p.kappa - p.rho * p.sigma * iu
    d = np.sqrt(beta**2 + p.sigma**2 * (iu + u**2))
    g = (beta - d) / (beta + d)
    edt = np.exp(-d * T)
    C = (p.kappa * p.theta / p.sigma**2) * (
        (beta - d) * T - 2.0 * np.log((1.0 - g * edt) / (1.0 - g))
    )
    Dv = (beta - d) / p.sigma**2 * (1.0 - edt) / (1.0 - g * edt)
    return np.exp(C + Dv * p.v0)


def cumulants(T: float, p: HestonParams) -> tuple[float, float]:
    """First two cumulants of ``ln(F_T / F_0)`` (Fang & Oosterlee 2008)."""
    k, th, s, r, v0 = p.kappa, p.theta, p.sigma, p.rho, p.v0
    e = np.exp(-k * T)
    c1 = (1.0 - e) * (th - v0) / (2.0 * k) - 0.5 * th * T
    c2 = (
        s * T * k * e * (v0 - th) * (8.0 * k * r - 4.0 * s)
        + k * r * s * (1.0 - e) * (16.0 * th - 8.0 * v0)
        + 2.0 * th * k * T * (-4.0 * k * r * s + s**2 + 4.0 * k**2)
        + s**2 * ((th - 2.0 * v0) * e**2 + th * (6.0 * e - 7.0) + 2.0 * v0)
        + 8.0 * k**2 * (v0 - th) * (1.0 - e)
    ) / (8.0 * k**3)
    return c1, c2


def expected_total_variance(T, p: HestonParams):
    """E[integral_0^T v_t dt]; a smooth, increasing proxy for ATM total implied variance."""
    T = np.asarray(T, dtype=float)
    return p.theta * T + (p.v0 - p.theta) * (1.0 - np.exp(-p.kappa * T)) / p.kappa


def _chi_psi(omega, a, d):
    """COS payoff integrals of e^y and 1 against cos(omega (y - a)) over [a, d]."""
    arg = omega * (d - a)
    cos_t, sin_t = np.cos(arg), np.sin(arg)
    chi = (cos_t * np.exp(d) - np.exp(a) + omega * sin_t * np.exp(d)) / (1.0 + omega**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = np.where(omega == 0.0, d - a, sin_t / omega)
    return chi, psi


def cos_price(F, K, T: float, p: HestonParams, is_call=True, D=1.0, N: int = 1024, L: float = 30.0):
    """Discounted European option prices for one maturity and many strikes via the COS method.

    The truncation interval for ``y = ln(F_T / K)`` is centred per strike on its mean, so the
    expansion frequencies and the characteristic function are shared across strikes. The interval
    is sized from the second cumulant only, so ``L`` is wider than the usual c2 + sqrt(c4) rule to
    cover Heston's heavy tails; the defaults reach ~1e-12 on the Fang-Oosterlee benchmark.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    c1, c2 = cumulants(T, p)
    half_width = L * np.sqrt(abs(c2))
    width = 2.0 * half_width
    a = np.log(F / K) + c1 - half_width
    upper = np.minimum(a + width, 0.0)  # put payoff K(1 - e^y) is nonzero only for y < 0

    n = np.arange(N)
    omega = (n * np.pi / width)[:, None]
    # e^{i u (x - a)} with x - a = half_width - c1, identical for all strikes.
    phi = char_func(omega[:, 0], T, p) * np.exp(1j * omega[:, 0] * (half_width - c1))
    chi, psi = _chi_psi(omega, a[None, :], upper[None, :])
    coeffs = 2.0 / width * K[None, :] * (psi - chi)
    terms = phi.real[:, None] * coeffs
    terms[0] *= 0.5
    put = np.where(a < 0.0, terms.sum(axis=0), 0.0)
    put = np.maximum(put, np.maximum(K - F, 0.0))
    call = put + F - K
    return D * np.where(is_call, call, put)


def implied_vol_slice(F, K, T: float, p: HestonParams, **cos_kwargs):
    """Black-76 implied volatilities of Heston prices, priced as out-of-the-money options."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    is_call = K >= F
    prices = cos_price(F, K, T, p, is_call, 1.0, **cos_kwargs)
    return black76.implied_vol(prices, F, K, T, is_call, 1.0)
