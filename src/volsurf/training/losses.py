"""Soft no-arbitrage penalties for differentiable surfaces (in the spirit of Ackerer et al. 2020).

All terms are dimensionless hinges evaluated on a collocation grid:

* calendar:  T dw/dT / w >= margin   (d log w / d log T; equals 1 for a flat-vol surface),
* butterfly: Durrleman g(k) >= margin,
* wing:      |dw/dk| <= 2 - margin at the outermost collocation strikes (Lee's moment bound).

``power=1`` gives the hinge of Ackerer et al.; ``power=2`` a squared hinge that is smooth at the
boundary. The margin pushes the surface strictly inside the admissible region, so finite
evaluation tolerances do not count residual boundary contact as a violation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from volsurf.arbitrage.conditions import durrleman_g, surface_derivatives
from volsurf.training.collocation import CollocationGrid


@dataclass
class PenaltyConfig:
    calendar: float = 0.0
    butterfly: float = 0.0
    wing: float = 0.0
    margin: float = 1e-3
    power: int = 1

    @property
    def active(self) -> bool:
        return max(self.calendar, self.butterfly, self.wing) > 0.0


def arbitrage_penalties(w_fn, grid: CollocationGrid, cfg: PenaltyConfig) -> dict[str, torch.Tensor]:
    w, w_k, w_kk, w_T = surface_derivatives(w_fn, grid.k, grid.T, create_graph=True)

    def hinge(x: torch.Tensor) -> torch.Tensor:
        return torch.relu(cfg.margin - x) ** cfg.power

    return {
        "calendar": hinge(grid.T * w_T / w).mean(),
        "butterfly": hinge(durrleman_g(grid.k, w, w_k, w_kk)).mean(),
        "wing": hinge(2.0 - w_k[grid.edge].abs()).mean(),
    }
