"""Static no-arbitrage conditions on the total implied variance surface w(k, T).

Here k = log(K / F_T) is log forward-moneyness and w = sigma_imp^2 T. A surface is free of static
arbitrage when (Roper 2010; Gatheral & Jacquier 2014):

* calendar:   dw/dT >= 0 at fixed k,
* butterfly:  Durrleman's g(k) >= 0 on every slice,
* wings:      w(k) / |k| <= 2 as |k| -> infinity (Lee 2004), and w > 0.

Checks exist for gridded surfaces (finite differences, NumPy) and for differentiable surfaces
(autograd, torch). The autograd derivatives are also what the training penalties are built on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch


def durrleman_g(k, w, w_k, w_kk):
    """Durrleman's butterfly function; the risk-neutral density is nonnegative iff g >= 0."""
    return (1.0 - k * w_k / (2.0 * w)) ** 2 - 0.25 * w_k**2 * (1.0 / w + 0.25) + 0.5 * w_kk


@dataclass
class ArbitrageReport:
    n_points: int
    calendar_frac: float  # share of points with dw/dT < -tol
    calendar_max: float  # worst calendar violation (magnitude of the negative dw/dT or step)
    butterfly_frac: float  # share of points with g < -tol
    butterfly_min_g: float
    positivity_frac: float  # share of points with w <= 0
    wing_max_ratio: float  # max w / |k| on the outermost strikes of each slice

    @property
    def arbitrage_free(self) -> bool:
        return (
            self.calendar_frac == 0.0
            and self.butterfly_frac == 0.0
            and self.positivity_frac == 0.0
            and self.wing_max_ratio <= 2.0
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _report(w, g, calendar, wing_ratio, tol) -> ArbitrageReport:
    return ArbitrageReport(
        n_points=int(w.size),
        calendar_frac=float(np.mean(calendar < -tol)) if calendar.size else 0.0,
        calendar_max=float(max(0.0, -calendar.min())) if calendar.size else 0.0,
        butterfly_frac=float(np.mean(g < -tol)),
        butterfly_min_g=float(g.min()),
        positivity_frac=float(np.mean(w <= 0.0)),
        wing_max_ratio=float(np.max(wing_ratio)),
    )


def fd_slice_derivatives(w, k):
    """Second-order finite-difference w_k and w_kk at interior points of a (possibly nonuniform)
    k-grid. ``w`` has shape (..., len(k)); returns interior (k, w, w_k, w_kk)."""
    h1 = k[1:-1] - k[:-2]
    h2 = k[2:] - k[1:-1]
    wm, w0, wp = w[..., :-2], w[..., 1:-1], w[..., 2:]
    w_k = (
        (-h2 / (h1 * (h1 + h2))) * wm + ((h2 - h1) / (h1 * h2)) * w0 + (h1 / (h2 * (h1 + h2))) * wp
    )
    w_kk = 2.0 * (h2 * wm - (h1 + h2) * w0 + h1 * wp) / (h1 * h2 * (h1 + h2))
    return k[1:-1], w0, w_k, w_kk


def check_grid(w, k, T, tol: float = 1e-6) -> ArbitrageReport:
    """Check a surface sampled on a grid; ``w`` has shape (len(T), len(k)), T ascending.

    Calendar arbitrage is the discrete condition w(k, T_{i+1}) >= w(k, T_i); butterfly uses
    finite-difference derivatives at interior strikes.
    """
    w, k, T = np.asarray(w, float), np.asarray(k, float), np.asarray(T, float)
    if w.shape != (T.size, k.size):
        raise ValueError(f"w has shape {w.shape}, expected {(T.size, k.size)}")
    ki, wi, w_k, w_kk = fd_slice_derivatives(w, k)
    g = durrleman_g(ki, wi, w_k, w_kk)
    calendar = np.diff(w, axis=0)
    edges = [0, -1]
    wing_ratio = w[:, edges] / np.abs(k[edges])
    return _report(w, g, calendar, wing_ratio, tol)


SurfaceFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def surface_derivatives(w_fn: SurfaceFn, k: torch.Tensor, T: torch.Tensor, create_graph=False):
    """Pointwise w, dw/dk, d2w/dk2 and dw/dT of a differentiable surface.

    ``w_fn`` must act elementwise on (k, T). With ``create_graph=True`` the derivatives stay
    differentiable with respect to the surface's parameters, for use in training penalties.
    """
    k = k.detach().clone().requires_grad_(True)
    T = T.detach().clone().requires_grad_(True)
    w = w_fn(k, T)
    w_k, w_T = torch.autograd.grad(w.sum(), (k, T), create_graph=True, allow_unused=True)
    w_T = torch.zeros_like(T) if w_T is None else w_T
    if w_k.requires_grad:
        (w_kk,) = torch.autograd.grad(w_k.sum(), k, create_graph=create_graph, allow_unused=True)
    else:
        w_kk = None
    w_kk = torch.zeros_like(k) if w_kk is None else w_kk
    if not create_graph:
        w, w_k, w_T = w.detach(), w_k.detach(), w_T.detach()
    return w, w_k, w_kk, w_T


def check_torch(w_fn: SurfaceFn, k, T, tol: float = 1e-6, dtype=torch.float64) -> ArbitrageReport:
    """Check a differentiable surface on the tensor grid k x T using exact autograd derivatives."""
    k_t = torch.as_tensor(np.asarray(k), dtype=dtype)
    T_t = torch.as_tensor(np.asarray(T), dtype=dtype)
    TT, KK = torch.meshgrid(T_t, k_t, indexing="ij")
    w, w_k, w_kk, w_T = surface_derivatives(w_fn, KK, TT)
    g = durrleman_g(KK, w, w_k, w_kk)
    w, g, w_T = (x.numpy() for x in (w, g, w_T))
    k_np = k_t.numpy()
    edges = [0, -1]
    wing_ratio = w[:, edges] / np.abs(k_np[edges])
    return _report(w, g, w_T, wing_ratio, tol)
