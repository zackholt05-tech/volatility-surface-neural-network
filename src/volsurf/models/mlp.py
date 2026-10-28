"""Neural implied-volatility surface fitted to one quote date.

The network maps (k / sqrt(T), sqrt(T)) to implied volatility through a smooth MLP. Two output
parameterizations are supported:

* ``prior="none"``: iv = softplus(net) + iv_floor, so w = iv^2 T is positive and vanishes as T -> 0;
* ``prior="ssvi"``: w = w_SSVI(k, T) exp(net), a multiplicative correction to an arbitrage-free
  SSVI fit of the same day (the network starts at zero, i.e. at the prior).

Smooth activations keep w twice differentiable in k, which the arbitrage penalties need. This
module is the unpenalized baseline; the arbitrage-aware variant adds penalty terms to its loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from volsurf.eval.dayset import FitData
from volsurf.models.base import SurfaceModel
from volsurf.models.ssvi import SSVISurface

DTYPE = torch.float64
ACTIVATIONS = {"softplus": nn.Softplus, "tanh": nn.Tanh, "silu": nn.SiLU}


@dataclass
class MLPConfig:
    hidden: tuple[int, ...] = (64, 64, 64)
    activation: str = "softplus"
    prior: str = "none"  # "none" or "ssvi"
    steps: int = 500  # Adam warm-up steps
    lr: float = 5e-3
    lr_final: float = 1e-4
    lbfgs_iters: int = 500  # L-BFGS refinement after Adam (0 disables)
    iv_floor: float = 0.005


def _mlp(hidden, activation) -> nn.Sequential:
    layers, width = [], 2
    for h in hidden:
        layers += [nn.Linear(width, h), ACTIVATIONS[activation]()]
        width = h
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers).to(DTYPE)


def _softplus_inv(y: float) -> float:
    return float(np.log(np.expm1(y)))


class MLPSurface(SurfaceModel):
    name = "mlp"

    def __init__(self, cfg: MLPConfig | None = None, seed: int = 0) -> None:
        super().__init__()
        self.cfg = cfg or MLPConfig()
        self.seed = seed

    # -- surface in torch -------------------------------------------------------------------
    def _features(self, k: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        sqrt_T = torch.sqrt(T)
        x = torch.stack([k / sqrt_T, sqrt_T], dim=-1)
        return (x - self._mu) / self._sd

    def _prior_w(self, k: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        Ts, th = self._prior_T, self._prior_theta
        idx = torch.clamp(torch.searchsorted(Ts, T), 1, len(Ts) - 1)
        t0, t1 = Ts[idx - 1], Ts[idx]
        theta = th[idx - 1] + (T - t0) / (t1 - t0) * (th[idx] - th[idx - 1])
        theta = torch.where(T < Ts[0], th[0] * T / Ts[0], theta)
        theta = torch.where(T > Ts[-1], th[-1] * T / Ts[-1], theta)
        p = self._prior
        phi = p.eta / (theta**p.gamma * (1.0 + theta) ** (1.0 - p.gamma))
        x = phi * k + p.rho
        return 0.5 * theta * (1.0 + p.rho * phi * k + torch.sqrt(x * x + 1.0 - p.rho**2))

    def w_torch(self, k: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Differentiable total variance for elementwise tensors ``k`` and ``T``."""
        out = self.net(self._features(k, T)).squeeze(-1)
        if self.cfg.prior == "ssvi":
            return self._prior_w(k, T) * torch.exp(out)
        iv = nn.functional.softplus(out) + self.cfg.iv_floor
        return iv * iv * T

    # -- fitting ----------------------------------------------------------------------------
    def fit(self, data: FitData) -> MLPSurface:
        cfg = self.cfg
        torch.manual_seed(self.seed)
        k = torch.as_tensor(data.k, dtype=DTYPE)
        T = torch.as_tensor(data.T, dtype=DTYPE)
        iv = torch.as_tensor(data.iv, dtype=DTYPE)
        s = torch.as_tensor(data.s, dtype=DTYPE)

        raw = torch.stack([k / torch.sqrt(T), torch.sqrt(T)], dim=-1)
        self._mu, self._sd = raw.mean(0), raw.std(0)
        self.net = _mlp(cfg.hidden, cfg.activation)
        last = self.net[-1]
        with torch.no_grad():
            if cfg.prior == "ssvi":
                prior = SSVISurface().fit(data)
                self._prior = prior
                self._prior_T = torch.as_tensor(prior.slice_T, dtype=DTYPE)
                self._prior_theta = torch.as_tensor(prior.theta, dtype=DTYPE)
                last.weight.zero_()
                last.bias.zero_()
            elif cfg.prior == "none":
                last.weight.mul_(0.1)
                last.bias.fill_(_softplus_inv(float(np.median(data.iv)) - cfg.iv_floor))
            else:
                raise ValueError(f"unknown prior {cfg.prior!r}")

        def objective() -> torch.Tensor:
            iv_model = torch.sqrt(self.w_torch(k, T) / T)
            return torch.mean(((iv_model - iv) / s) ** 2)

        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps, eta_min=cfg.lr_final)
        for _ in range(cfg.steps):
            opt.zero_grad()
            loss = objective()
            loss.backward()
            opt.step()
            sched.step()

        if cfg.lbfgs_iters:
            lbfgs = torch.optim.LBFGS(
                self.net.parameters(),
                max_iter=cfg.lbfgs_iters,
                history_size=50,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-10,
                tolerance_change=1e-14,
            )

            def closure():
                lbfgs.zero_grad()
                value = objective()
                value.backward()
                return value

            lbfgs.step(closure)

        with torch.no_grad():
            self.final_loss = float(objective())
        if not np.isfinite(self.final_loss):
            self.n_failed += 1
        return self

    def w(self, k, T) -> np.ndarray:
        k, T = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(T, dtype=float))
        k_t = torch.tensor(k.ravel(), dtype=DTYPE)
        T_t = torch.tensor(T.ravel(), dtype=DTYPE)
        with torch.no_grad():
            return self.w_torch(k_t, T_t).numpy().reshape(k.shape)
