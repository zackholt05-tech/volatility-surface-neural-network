"""Neural implied-volatility surface fitted to one quote date, optionally arbitrage-penalized.

A smooth MLP maps two features of (k, T) to a surface. Features:

* ``features="k_sqrtT"``: (k / sqrt(T), sqrt(T));
* ``features="k_theta"``: (k / sqrt(theta(T)), sqrt(T)), moneyness in units of the ATM total
  standard deviation of an SSVI fit, which makes smiles of different maturities look alike.

Output parameterizations:

* ``prior="none"``: iv = softplus(net) + iv_floor, so w = iv^2 T is positive and vanishes as T -> 0;
* ``prior="ssvi"``: w = w_SSVI(k, T) exp(3 tanh(net / 3)), a bounded multiplicative correction to
  an arbitrage-free SSVI fit of the same day (the network starts at zero, i.e. at the prior).

The loss is the mean squared residual in bid-ask half-spreads plus, when enabled, soft
no-arbitrage penalties on a collocation grid (``volsurf.training.losses``). Smooth activations
keep w twice differentiable in k, which the butterfly penalty needs. Fits can be warm-started
from the previous day's model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from volsurf.eval.dayset import FitData
from volsurf.models.base import SurfaceModel
from volsurf.models.ssvi import SSVISurface
from volsurf.training.collocation import CollocationConfig, build_grid
from volsurf.training.losses import PenaltyConfig, arbitrage_penalties

DTYPE = torch.float64
ACTIVATIONS = {"softplus": nn.Softplus, "tanh": nn.Tanh, "silu": nn.SiLU}


@dataclass
class MLPConfig:
    hidden: tuple[int, ...] = (64, 64, 64)
    activation: str = "softplus"
    prior: str = "none"  # "none" or "ssvi"
    features: str = "k_sqrtT"  # "k_sqrtT" or "k_theta"
    time_feature: str = (
        "sqrt"  # second input: "sqrt" (sqrt T) or "log" (log T, finer at the short end)
    )
    steps: int = 500  # Adam warm-up steps
    lr: float = 5e-3
    lr_final: float = 1e-4
    lbfgs_iters: int = 500  # L-BFGS refinement after Adam (0 disables)
    warm_steps: int = 100  # Adam steps when initialized from the previous day
    warm_lbfgs_iters: int = 300
    iv_floor: float = 0.005
    penalty: PenaltyConfig = field(default_factory=PenaltyConfig)
    collocation: CollocationConfig = field(default_factory=CollocationConfig)


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
        self.penalties: dict[str, float] = {}

    # -- surface in torch -------------------------------------------------------------------
    def _theta(self, T: torch.Tensor) -> torch.Tensor:
        Ts, th = self._prior_T, self._prior_theta
        idx = torch.clamp(torch.searchsorted(Ts, T), 1, len(Ts) - 1)
        t0, t1 = Ts[idx - 1], Ts[idx]
        theta = th[idx - 1] + (T - t0) / (t1 - t0) * (th[idx] - th[idx - 1])
        theta = torch.where(T < Ts[0], th[0] * T / Ts[0], theta)
        return torch.where(T > Ts[-1], th[-1] * T / Ts[-1], theta)

    def _prior_w(self, k: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        p, theta = self._prior, self._theta(T)
        phi = p.eta / (theta**p.gamma * (1.0 + theta) ** (1.0 - p.gamma))
        x = phi * k + p.rho
        return 0.5 * theta * (1.0 + p.rho * phi * k + torch.sqrt(x * x + 1.0 - p.rho**2))

    def _raw_features(self, k: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        scale = torch.sqrt(self._theta(T)) if self.cfg.features == "k_theta" else torch.sqrt(T)
        t = torch.log(T) if self.cfg.time_feature == "log" else torch.sqrt(T)
        return torch.stack([k / scale, t], dim=-1)

    def w_torch(self, k: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Differentiable total variance for elementwise tensors ``k`` and ``T``."""
        out = self.net((self._raw_features(k, T) - self._mu) / self._sd).squeeze(-1)
        if self.cfg.prior == "ssvi":
            # Bounded log-correction (factor within e^{+-3}) keeps early training from overflowing.
            return self._prior_w(k, T) * torch.exp(3.0 * torch.tanh(out / 3.0))
        iv = nn.functional.softplus(out) + self.cfg.iv_floor
        return iv * iv * T

    # -- fitting ----------------------------------------------------------------------------
    def _set_prior(self, data: FitData) -> None:
        cfg = self.cfg
        if (
            cfg.prior not in ("none", "ssvi")
            or cfg.features not in ("k_sqrtT", "k_theta")
            or cfg.time_feature not in ("sqrt", "log")
        ):
            raise ValueError(f"unknown option in {cfg}")
        if cfg.prior == "ssvi" or cfg.features == "k_theta":
            prior = SSVISurface().fit(data)
            self._prior = prior
            self._prior_T = torch.as_tensor(prior.slice_T, dtype=DTYPE)
            self._prior_theta = torch.as_tensor(prior.theta, dtype=DTYPE)

    def _init_net(self, data: FitData, k, T) -> None:
        cfg = self.cfg
        raw = self._raw_features(k, T)
        self._mu, self._sd = raw.mean(0), raw.std(0)
        self.net = _mlp(cfg.hidden, cfg.activation)
        last = self.net[-1]
        with torch.no_grad():
            if cfg.prior == "ssvi":
                last.weight.zero_()
                last.bias.zero_()
            else:
                last.weight.mul_(0.1)
                last.bias.fill_(_softplus_inv(float(np.median(data.iv)) - cfg.iv_floor))

    def fit(self, data: FitData, init: MLPSurface | None = None) -> MLPSurface:
        """Fit to one day; ``init`` warm-starts from a previous day's fitted model."""
        cfg = self.cfg
        torch.manual_seed(self.seed)
        k, T, iv, s = (torch.as_tensor(x, dtype=DTYPE) for x in (data.k, data.T, data.iv, data.s))
        self._set_prior(data)
        if init is None:
            self._init_net(data, k, T)
            steps, lbfgs_iters = cfg.steps, cfg.lbfgs_iters
        else:
            self.net = copy.deepcopy(init.net)
            self._mu, self._sd = init._mu, init._sd  # keep the input map fixed across days
            steps, lbfgs_iters = cfg.warm_steps, cfg.warm_lbfgs_iters

        pen_cfg = cfg.penalty
        grid = build_grid(data, cfg.collocation) if pen_cfg.active else None
        weights = {
            "calendar": pen_cfg.calendar,
            "butterfly": pen_cfg.butterfly,
            "wing": pen_cfg.wing,
        }

        def objective(details: bool = False):
            iv_model = torch.sqrt(self.w_torch(k, T) / T)
            fit_loss = torch.mean(((iv_model - iv) / s) ** 2)
            if grid is None:
                return (fit_loss, {}) if details else fit_loss
            terms = arbitrage_penalties(self.w_torch, grid, pen_cfg)
            total = fit_loss + sum(weights[name] * value for name, value in terms.items())
            return (total, {"fit": fit_loss} | terms) if details else total

        opt = torch.optim.Adam(
            self.net.parameters(), lr=cfg.lr if init is None else cfg.lr_final * 10
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps, 1), eta_min=cfg.lr_final)
        for _ in range(steps):
            opt.zero_grad()
            loss = objective()
            loss.backward()
            opt.step()
            sched.step()

        if lbfgs_iters:
            lbfgs = torch.optim.LBFGS(
                self.net.parameters(),
                max_iter=lbfgs_iters,
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

        total, parts = objective(details=True)
        self.final_loss = float(total.detach())
        self.penalties = {name: float(v.detach()) for name, v in parts.items()}
        if not np.isfinite(self.final_loss):
            self.n_failed += 1
        return self

    def w(self, k, T) -> np.ndarray:
        k, T = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(T, dtype=float))
        k_t = torch.tensor(k.ravel(), dtype=DTYPE)
        T_t = torch.tensor(T.ravel(), dtype=DTYPE)
        with torch.no_grad():
            return self.w_torch(k_t, T_t).numpy().reshape(k.shape)
