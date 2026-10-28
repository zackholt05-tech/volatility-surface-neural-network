"""Common interface for implied-volatility surface models.

A model is fitted on one quote date (``FitData``) and then evaluated anywhere through a
vectorized total-variance function ``w(k, T)``. Per-slice models (SVI, splines) share
``SliceSurface``, which joins independently fitted slices by interpolating total variance linearly
in T at fixed k and holds implied vol flat outside the quoted maturities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from volsurf.eval.dayset import FitData

W_FLOOR = 1e-10


class SurfaceModel(ABC):
    name: str = "surface"

    def __init__(self) -> None:
        self.n_failed = 0  # slices (or restarts) whose fit failed, reported by the harness

    @abstractmethod
    def fit(self, data: FitData) -> SurfaceModel: ...

    @abstractmethod
    def w(self, k, T) -> np.ndarray:
        """Total implied variance at broadcastable arrays ``k`` and ``T``."""

    def iv(self, k, T) -> np.ndarray:
        w = self.w(k, T)
        return np.sqrt(np.maximum(w, W_FLOOR) / np.asarray(T, dtype=float))


SliceFn = Callable[[np.ndarray], np.ndarray]


class SliceSurface(SurfaceModel):
    """Surface assembled from per-slice fits ``w_i(k)`` at maturities ``T_i``."""

    min_quotes = 5

    @abstractmethod
    def fit_slice(self, k, w, weight, T) -> SliceFn:
        """Fit one maturity; ``weight`` is the inverse squared total-variance error scale."""

    def fit(self, data: FitData) -> SliceSurface:
        # Residual scale in total variance: dw = 2 iv T d(iv), so sd_w = 2 iv T s.
        weight = 1.0 / (2.0 * data.iv * data.T * data.s) ** 2
        Ts, fns = [], []
        for i, T in enumerate(data.slice_T):
            m = data.slice_id == i
            if m.sum() < self.min_quotes:
                continue
            try:
                fn = self.fit_slice(data.k[m], data.w[m], weight[m], T)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                fn = None
            if fn is None:
                self.n_failed += 1
                continue
            Ts.append(T)
            fns.append(fn)
        if not fns:
            raise RuntimeError(f"{self.name}: no slice could be fitted")
        self.slice_T = np.asarray(Ts)
        self.slice_fns = fns
        return self

    def w(self, k, T) -> np.ndarray:
        k, T = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(T, dtype=float))
        shape = k.shape
        k, T = k.ravel(), T.ravel()
        Ts = self.slice_T
        W = np.stack([np.maximum(fn(k), W_FLOOR) for fn in self.slice_fns])  # (n_slices, n)

        hi = (
            np.clip(np.searchsorted(Ts, T), 1, len(Ts) - 1)
            if len(Ts) > 1
            else np.zeros_like(T, int)
        )
        lo = np.maximum(hi - 1, 0)
        cols = np.arange(len(k))
        w_lo, w_hi = W[lo, cols], W[hi, cols]
        span = Ts[hi] - Ts[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(span > 0, (T - Ts[lo]) / span, 0.0)
        out = w_lo + frac * (w_hi - w_lo)
        out = np.where(T < Ts[0], W[0] * T / Ts[0], out)  # flat implied vol outside
        out = np.where(T > Ts[-1], W[-1] * T / Ts[-1], out)
        return out.reshape(shape)
