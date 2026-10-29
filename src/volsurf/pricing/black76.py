"""Black-76 pricing on the forward and implied-volatility inversion.

All functions are vectorized over NumPy arrays and broadcast their arguments. ``D`` is the
discount factor to expiry, so prices are ``D`` times the undiscounted forward price. Inversion is
done on the undiscounted out-of-the-money price, which keeps precision in the wings.
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _npdf(x):
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _d1_d2(F, K, T, sigma):
    s = sigma * np.sqrt(T)
    d1 = np.log(F / K) / s + 0.5 * s
    return d1, d1 - s


def _otm_price(F, K, T, sigma):
    """Undiscounted price of the out-of-the-money option (call if K >= F, else put)."""
    d1, d2 = _d1_d2(F, K, T, sigma)
    call = F * ndtr(d1) - K * ndtr(d2)
    put = K * ndtr(-d2) - F * ndtr(-d1)
    return np.where(K >= F, call, put)


def price(F, K, T, sigma, is_call=True, D=1.0):
    """Discounted Black-76 price of a European call or put."""
    F, K, T, sigma, is_call, D = np.broadcast_arrays(F, K, T, sigma, is_call, D)
    d1, d2 = _d1_d2(F, K, T, sigma)
    call = F * ndtr(d1) - K * ndtr(d2)
    put = K * ndtr(-d2) - F * ndtr(-d1)
    return D * np.where(is_call, call, put)


def vega(F, K, T, sigma, D=1.0):
    """Sensitivity of the discounted price to ``sigma`` (identical for calls and puts)."""
    d1, _ = _d1_d2(F, K, T, sigma)
    return D * F * _npdf(d1) * np.sqrt(T)


def delta(F, K, T, sigma, is_call=True, D=1.0):
    """Sensitivity of the discounted price to the forward ``F``."""
    d1, _ = _d1_d2(F, K, T, sigma)
    return D * np.where(is_call, ndtr(d1), -ndtr(-d1))


def implied_vol(
    price,
    F,
    K,
    T,
    is_call=True,
    D=1.0,
    *,
    sigma_max: float = 10.0,
    tol: float = 1e-13,
    max_iter: int = 200,
):
    """Invert Black-76 prices to implied volatilities.

    Uses Newton steps safeguarded by a bisection bracket, so every admissible price converges.
    Prices outside the no-arbitrage bounds (at or below intrinsic, at or above the upper bound)
    return NaN.
    """
    price, F, K, T, is_call, D = (
        np.array(a, dtype=float) for a in np.broadcast_arrays(price, F, K, T, is_call, D)
    )
    is_call = is_call.astype(bool)

    # Put-call parity: subtracting intrinsic value gives the out-of-the-money price.
    intrinsic = np.maximum(np.where(is_call, F - K, K - F), 0.0)
    target = price / D - intrinsic
    upper = np.minimum(F, K)
    valid = np.isfinite(target) & (target > 0.0) & (target < upper) & (T > 0.0)

    # Safe placeholders for invalid entries keep the iteration warning-free.
    target = np.where(valid, target, 0.5 * upper)
    T = np.where(valid, T, 1.0)

    lo = np.zeros_like(target)
    hi = np.full_like(target, sigma_max)
    # Initial guess: ATM Brenner-Subrahmanyam blended with the moneyness-implied scale.
    sigma = np.sqrt(2.0 * np.pi / T) * target / np.sqrt(F * K)
    sigma = np.clip(np.maximum(sigma, np.sqrt(2.0 * np.abs(np.log(F / K)) / T)), 0.05, 2.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(max_iter):
            f = _otm_price(F, K, T, sigma) - target
            hi = np.where(f > 0.0, sigma, hi)
            lo = np.where(f <= 0.0, sigma, lo)
            d1, _ = _d1_d2(F, K, T, sigma)
            newton = sigma - f / (F * _npdf(d1) * np.sqrt(T))
            bisect = ~np.isfinite(newton) | (newton <= lo) | (newton >= hi)
            new = np.where(bisect, 0.5 * (lo + hi), newton)
            converged = np.abs(new - sigma) <= tol * np.maximum(sigma, 1.0)
            sigma = new
            if np.all(converged | ~valid):
                break

    valid &= sigma < sigma_max * (1.0 - 1e-9)
    return np.where(valid, sigma, np.nan)


# Forward-normalized prices in log space (F = 1, K = e^k, undiscounted), parametrized by total
# variance w = sigma^2 T. They stay finite far in the wings, where plain prices underflow and the
# call/put formulas cancel catastrophically, which matters for price-space mixtures of surfaces.


def _log1mexp(x):
    """log(1 - exp(x)) for x <= 0, accurate on both sides of -log 2."""
    with np.errstate(divide="ignore"):
        return np.where(x > -np.log(2.0), np.log(-np.expm1(x)), np.log1p(-np.exp(x)))


def log_otm_price(k, w):
    """Log of the normalized out-of-the-money price: call for k >= 0, put for k < 0."""
    k, w = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(w, dtype=float))
    v = np.sqrt(w)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = -k / v + 0.5 * v
        d2 = d1 - v
        # Both prices are A - B with log A >= log B.
        log_a = np.where(k >= 0, log_ndtr(d1), k + log_ndtr(-d2))
        log_b = np.where(k >= 0, k + log_ndtr(d2), log_ndtr(-d1))
        return log_a + _log1mexp(np.minimum(log_b - log_a, 0.0))


def total_vol_from_log_price(log_p, k, lo, hi, *, tol: float = 1e-13, max_iter: int = 100):
    """Total implied vol v = sigma sqrt(T) solving ``log_otm_price(k, v^2) = log_p``.

    The root must lie in the bracket [lo, hi] (e.g. the smallest and largest component vols of a
    mixture, since the OTM price is increasing in v). Newton steps on the log price, with
    d log P / dv = n(d1) / P, are safeguarded by bisection inside the bracket.
    """
    log_p, k, lo, hi = (np.array(a, dtype=float) for a in np.broadcast_arrays(log_p, k, lo, hi))
    v = 0.5 * (lo + hi)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(max_iter):
            g = log_otm_price(k, v * v) - log_p
            hi = np.where(g > 0.0, v, hi)
            lo = np.where(g <= 0.0, v, lo)
            d1 = -k / v + 0.5 * v
            slope = np.exp(-0.5 * d1 * d1 - np.log(_SQRT_2PI) - log_otm_price(k, v * v))
            newton = v - g / slope
            bisect = ~np.isfinite(newton) | (newton <= lo) | (newton >= hi)
            new = np.where(bisect, 0.5 * (lo + hi), newton)
            done = (np.abs(new - v) <= tol * v) | (hi - lo <= tol * v)
            v = new
            if np.all(done):
                break
    return v

