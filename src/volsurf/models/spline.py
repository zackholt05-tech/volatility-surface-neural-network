"""Per-slice smoothing splines in total variance, joined linearly in T."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import UnivariateSpline

from volsurf.models.base import W_FLOOR, SliceSurface


class SplineSurface(SliceSurface):
    """Cubic smoothing spline of w(k) per maturity.

    Smoothing follows the discrepancy principle: knots are added until the weighted residual sum
    of squares falls to ``smooth * n``, i.e. residuals of about one bid-ask half-spread. (GCV
    smoothing collapses on short-dated slices with strongly heteroscedastic spreads.) Outside the
    quoted strikes each slice continues linearly with its end slope. No arbitrage constraint is
    imposed.
    """

    name = "spline"

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def fit_slice(self, k, w, weight, T):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spline = UnivariateSpline(k, w, w=np.sqrt(weight), s=self.smooth * len(k), k=3)
        d = spline.derivative()
        k0, k1 = k[0], k[-1]
        w0, w1, s0, s1 = spline(k0), spline(k1), d(k0), d(k1)

        def fn(x):
            x = np.asarray(x, dtype=float)
            out = spline(np.clip(x, k0, k1))
            out = np.where(x < k0, w0 + s0 * (x - k0), out)
            out = np.where(x > k1, w1 + s1 * (x - k1), out)
            return np.maximum(out, W_FLOOR)

        return fn
