"""Raw SVI slice parameterization (Gatheral 2004) and a per-slice SVI surface.

``raw_svi`` is written with arithmetic operators only, so it accepts NumPy arrays or torch tensors.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from volsurf.models.base import W_FLOOR, SliceSurface


def raw_svi(k, a, b, rho, m, sigma):
    """Total implied variance w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))."""
    x = k - m
    return a + b * (rho * x + (x * x + sigma * sigma) ** 0.5)


def _to_raw(p):
    """Map fitted parameters (v_min, b_frac, rho, m, sigma) to raw SVI (a, b, rho, m, sigma).

    ``v_min`` >= 0 is the slice's minimum total variance, so w > 0 everywhere, and
    b = b_frac * 2 / (1 + |rho|) keeps both wing slopes within Lee's bound of 2.
    """
    v_min, b_frac, rho, m, sigma = p
    b = b_frac * 2.0 / (1.0 + abs(rho))
    a = v_min - b * sigma * np.sqrt(1.0 - rho**2)
    return a, b, rho, m, sigma


def fit_raw_svi(k, w, weight, T, n_starts: int = 3, max_nfev: int = 400):
    """Calibrate one slice by weighted least squares in implied vol.

    Direct bounded least squares over all five parameters from several starting points (rather
    than the quasi-explicit split of Zeliade 2009); the bounds enforce positive total variance and
    Lee-compliant wings, but not butterfly freeness. Returns raw parameters or None.
    """
    iv = np.sqrt(w / T)
    sd_iv = 1.0 / (np.sqrt(weight) * 2.0 * iv * T)  # back to the IV error scale
    k_lo, k_hi = k.min(), k.max()
    k_star = k[np.argmin(w)]
    lower = [0.0, 0.0, -0.999, k_lo - 1.0, 1e-3]
    upper = [w.max(), 1.0, 0.999, k_hi + 1.0, 2.0]
    starts = [
        [0.9 * w.min(), 0.3, -0.6, k_star, 0.1],
        [0.5 * w.min(), 0.6, -0.9, k_star + 0.05, 0.3 * (k_hi - k_lo) + 1e-3],
        [0.95 * w.min(), 0.1, -0.3, 0.0, 0.05],
    ][:n_starts]

    def residuals(p):
        model = raw_svi(k, *_to_raw(p))
        return (np.sqrt(np.maximum(model, W_FLOOR) / T) - iv) / sd_iv

    best = None
    for x0 in starts:
        x0 = np.clip(x0, np.add(lower, 1e-9), np.subtract(upper, 1e-9))
        sol = least_squares(residuals, x0, bounds=(lower, upper), max_nfev=max_nfev)
        if best is None or sol.cost < best.cost:
            best = sol
    return _to_raw(best.x) if best is not None and np.isfinite(best.cost) else None


class SVISurface(SliceSurface):
    """Independent raw SVI fits per maturity, joined linearly in total variance."""

    name = "svi"

    def fit_slice(self, k, w, weight, T):
        params = fit_raw_svi(k, w, weight, T)
        if params is None:
            return None
        return lambda x, p=params: raw_svi(x, *p)
