"""Collocation grids on which arbitrage penalties are enforced during training.

The grid covers every quoted maturity plus log-spaced maturities in between, and at each maturity
a uniform k-range that extends past the fitted strikes by ``widen`` times the quoted width on each
side. Penalizing only at quote locations would leave the surface free to violate the conditions
between and beyond the quotes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from volsurf.eval.dayset import FitData


@dataclass
class CollocationConfig:
    n_T: int = 20  # log-spaced maturities added to the quoted ones
    n_k: int = 41  # points per maturity
    widen: float = 1.0  # extend the fitted k-range by this share of its width on each side
    T_extend: tuple[float, float] = (0.8, 1.25)  # multiply the shortest / longest quoted maturity


@dataclass
class CollocationGrid:
    k: torch.Tensor  # flattened (n_T * n_k,)
    T: torch.Tensor
    edge: torch.Tensor  # boolean: first or last point of each maturity row

    def __len__(self) -> int:
        return self.k.numel()


def build_grid(data: FitData, cfg: CollocationConfig, dtype=torch.float64) -> CollocationGrid:
    slice_T, lo, hi = [], [], []
    for i, T in enumerate(data.slice_T):
        m = data.slice_id == i
        if m.any():
            slice_T.append(T)
            lo.append(data.k[m].min())
            hi.append(data.k[m].max())
    slice_T, lo, hi = map(np.asarray, (slice_T, lo, hi))

    t0, t1 = slice_T.min() * cfg.T_extend[0], slice_T.max() * cfg.T_extend[1]
    T_grid = np.union1d(np.geomspace(t0, t1, cfg.n_T), slice_T)
    lo_T, hi_T = np.interp(T_grid, slice_T, lo), np.interp(T_grid, slice_T, hi)
    width = hi_T - lo_T
    lo_T, hi_T = lo_T - cfg.widen * width, hi_T + cfg.widen * width

    frac = np.linspace(0.0, 1.0, cfg.n_k)
    K = lo_T[:, None] + frac[None, :] * (hi_T - lo_T)[:, None]
    TT = np.broadcast_to(T_grid[:, None], K.shape)
    edge = np.zeros(K.shape, dtype=bool)
    edge[:, [0, -1]] = True
    return CollocationGrid(
        k=torch.as_tensor(K.ravel().copy(), dtype=dtype),
        T=torch.as_tensor(TT.ravel().copy(), dtype=dtype),
        edge=torch.as_tensor(edge.ravel().copy()),
    )
