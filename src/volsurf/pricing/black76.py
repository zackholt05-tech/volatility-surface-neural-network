"""Black-76 pricing on the forward and implied-volatility inversion.

All functions are vectorized over NumPy arrays and broadcast their arguments. ``D`` is the
discount factor to expiry, so prices are ``D`` times the undiscounted forward price. Inversion is
done on the undiscounted out-of-the-money price, which keeps precision in the wings.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

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
