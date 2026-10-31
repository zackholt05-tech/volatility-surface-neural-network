"""Cross-day conditional surface model: a set encoder maps a day's quotes to a latent vector z, and
a decoder maps z to an arbitrage-free surface.

The decoder outputs the rates of M eSSVI surfaces on a fixed maturity grid (the always-feasible
parametrization of ``essvi.knots_from_rates``), interpolated linearly in T and mixed in price space
with maturity-independent weights, exactly as the per-day mixtures of Phase 5. Every decoded surface
is therefore free of static arbitrage for any z, including latents that were never encoded from a
chain (forecasts, interpolations between days). The encoder is a permutation-invariant set network
(DeepSets) over the observed quotes, so it accepts any subset of a chain. Inference either stops at
the encoded z (amortized) or refines z by minimizing the fit loss on the observed quotes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from volsurf.eval.dayset import FitData
from volsurf.models.base import W_FLOOR, SurfaceModel
from volsurf.models.essvi import RATE_FLOOR, THETA_FLOOR, U_MAX, _steps, essvi_w, interp_knots
from volsurf.models.neural_essvi import (
    _log_otm_price_torch,
    _minimize_lbfgs,
    mixture_total_variance,
)
from volsurf.pricing.black76 import log_otm_price, total_vol_from_log_price

# Dense at the short end, where per-day mixtures are most accurate and quoted expiries are closest.
GRID_DAYS = (5, 7, 9, 11, 14, 17, 21, 25, 30, 37, 45, 55, 67, 82, 100, 122, 150, 182, 225, 275,
             335, 410, 500, 610, 740, 900)  # fmt: skip
RATE_BOUND = 15.0  # decoder rates stay in (-15, 15): far from softplus under- and overflow
LOGIT_BOUND = 10.0


@dataclass
class ConditionalConfig:
    latent_dim: int = 8
    components: int = 4
    enc_hidden: int = 128
    dec_hidden: tuple[int, ...] = (256, 256)


# ---- fixed-grid eSSVI knots ------------------------------------------------------------------


def grid_knots(rates: torch.Tensor, grid: torch.Tensor):
    """``knots_from_rates`` for rates of shape (..., n, 3), batched over the leading axes."""
    dT, dlog = _steps(grid, torch)
    fv = F.softplus(rates[..., 0])
    r_plus = F.softplus(rates[..., 1]) + RATE_FLOOR
    r_minus = F.softplus(rates[..., 2]) + RATE_FLOOR
    u_plus = -U_MAX * torch.expm1(-torch.cumsum(r_plus * dlog, -1))
    u_minus = -U_MAX * torch.expm1(-torch.cumsum(r_minus * dlog, -1))
    s = u_plus + u_minus
    floor = s * torch.maximum(u_plus, u_minus) / 8.0
    inc = fv * dT + THETA_FLOOR
    theta = [floor[..., 0] + inc[..., 0]]
    for i in range(1, grid.shape[0]):
        theta.append(
            torch.maximum(theta[-1] * s[..., i] / s[..., i - 1], floor[..., i]) + inc[..., i]
        )
    return torch.stack(theta, -1), u_plus, u_minus


def interp_coefficients(T, grid: np.ndarray):
    """Indices and weights reproducing ``interp_knots`` on a fixed grid:
    theta(T) = a_th theta[lo] + b_th theta[hi] and u(T) = a_u u[lo] + b_u u[hi]."""
    T = np.asarray(T, dtype=float)
    n = len(grid)
    hi = np.clip(np.searchsorted(grid, T), 1, n - 1)
    lo = hi - 1
    frac = (T - grid[lo]) / (grid[hi] - grid[lo])
    a_th, b_th, a_u, b_u = 1.0 - frac, frac, 1.0 - frac, frac
    short, long = T < grid[0], T > grid[-1]
    lo = np.where(short, 0, np.where(long, n - 1, lo))
    hi = np.where(short, 0, np.where(long, n - 1, hi))
    a_th = np.where(short, T / grid[0], np.where(long, T / grid[-1], a_th))
    a_u = np.where(short, T / grid[0], np.where(long, 1.0, a_u))
    b_th = np.where(short | long, 0.0, b_th)
    b_u = np.where(short | long, 0.0, b_u)
    return lo, hi, np.stack([a_th, b_th, a_u, b_u])


# ---- batched quotes and the mixture implied vol ----------------------------------------------


@dataclass
class QuoteBatch:
    """Quotes of several days, flattened, with the day each belongs to."""

    k: torch.Tensor
    T: torch.Tensor
    iv: torch.Tensor
    s: torch.Tensor
    day: torch.Tensor  # 0..n_days-1
    lo: torch.Tensor
    hi: torch.Tensor
    coef: torch.Tensor  # (4, n): a_th, b_th, a_u, b_u
    n_days: int

    @classmethod
    def from_arrays(cls, k, T, iv, s, day, n_days: int, grid: np.ndarray) -> QuoteBatch:
        lo, hi, coef = interp_coefficients(T, grid)
        t = torch.as_tensor
        return cls(t(k), t(T), t(iv), t(s), t(day), t(lo), t(hi), t(coef), n_days)

    def features(self) -> torch.Tensor:
        """Encoder inputs: log-moneyness over sqrt(T), log T, implied vol and log half-spread.
        Nothing is normalized by quantities fitted on the full chain, so held-out quotes cannot
        leak into the encoding of a partial one."""
        return torch.stack(
            [self.k / torch.sqrt(self.T), torch.log(self.T), self.iv, torch.log(self.s)], -1
        )


def _mixture_vol_np(k, w, log_pi):
    """Total implied vol of a price mixture with per-quote weights; w and log_pi are (n, M)."""
    log_mix = np.logaddexp.reduce(log_pi + log_otm_price(k[:, None], w), axis=1)
    v = np.sqrt(w)
    return total_vol_from_log_price(log_mix, k, v.min(axis=1), v.max(axis=1))


def batch_iv(rates: torch.Tensor, logits: torch.Tensor, grid: torch.Tensor, q: QuoteBatch):
    """Model implied vols at the quotes of each day; rates (n_days, M, n, 3), logits (n_days, M).

    For mixtures, the implied vol is solved in log-price space and differentiated implicitly
    (d sigma = d log P / (d log P / d sigma)), as in the per-day mixtures."""
    th, up, um = grid_knots(rates, grid)  # (n_days, M, n)
    a_th, b_th, a_u, b_u = q.coef[:, :, None]
    th_q = a_th * th[q.day, :, q.lo] + b_th * th[q.day, :, q.hi]  # (n_quotes, M)
    up_q = a_u * up[q.day, :, q.lo] + b_u * up[q.day, :, q.hi]
    um_q = a_u * um[q.day, :, q.lo] + b_u * um[q.day, :, q.hi]
    w = torch.clamp(essvi_w(q.k[:, None], th_q, up_q, um_q), min=W_FLOOR)
    if w.shape[1] == 1:
        return torch.sqrt(w[:, 0] / q.T)
    log_pi = torch.log_softmax(logits, -1)[q.day]
    log_p = torch.logsumexp(log_pi + _log_otm_price_torch(q.k[:, None], w), -1)
    v = torch.as_tensor(_mixture_vol_np(q.k.numpy(), w.detach().numpy(), log_pi.detach().numpy()))
    d1 = -q.k / v + 0.5 * v
    log_slope = -0.5 * d1 * d1 - 0.5 * np.log(2.0 * np.pi) - log_p.detach()
    return (v + (log_p - log_p.detach()) * torch.exp(-log_slope)) / torch.sqrt(q.T)


def per_day_loss(iv_model: torch.Tensor, q: QuoteBatch) -> torch.Tensor:
    """Mean squared spread-normalized IV residual of each day."""
    sq = ((iv_model - q.iv) / q.s) ** 2
    total = torch.zeros(q.n_days, dtype=sq.dtype).index_add(0, q.day, sq)
    count = torch.zeros(q.n_days, dtype=sq.dtype).index_add(0, q.day, torch.ones_like(sq))
    return total / count


# ---- networks --------------------------------------------------------------------------------


class SetEncoder(nn.Module):
    """DeepSets: a per-quote network, mean and max pooling within each day, then a day network."""

    def __init__(self, n_features: int, hidden: int, latent_dim: int) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(n_features, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )  # fmt: skip
        self.rho = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, latent_dim)
        )

    def forward(self, x: torch.Tensor, day: torch.Tensor, n_days: int) -> torch.Tensor:
        h = self.phi(x)
        total = h.new_zeros(n_days, h.shape[1]).index_add(0, day, h)
        count = h.new_zeros(n_days).index_add(0, day, torch.ones_like(h[:, 0]))
        mean = total / count[:, None]
        index = day[:, None].expand_as(h)
        peak = h.new_zeros(n_days, h.shape[1]).scatter_reduce(
            0, index, h, "amax", include_self=False
        )
        return self.rho(torch.cat([mean, peak], -1))


class SurfaceDecoder(nn.Module):
    """z -> bounded rates of M eSSVI surfaces on the grid, and mixture logits."""

    def __init__(self, latent_dim: int, components: int, n_grid: int, hidden) -> None:
        super().__init__()
        self.M, self.n = components, n_grid
        layers, width = [], latent_dim
        for h in hidden:
            layers += [nn.Linear(width, h), nn.SiLU()]
            width = h
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(width, components * n_grid * 3 + components)

    def forward(self, z: torch.Tensor):
        raw = self.out(self.body(z))
        rates = RATE_BOUND * torch.tanh(raw[..., : -self.M] / RATE_BOUND)
        logits = LOGIT_BOUND * torch.tanh(raw[..., -self.M :] / LOGIT_BOUND)
        return rates.reshape(*z.shape[:-1], self.M, self.n, 3), logits

    @torch.no_grad()
    def init_output(self, rates: np.ndarray, weights: np.ndarray, scale: float = 0.01) -> None:
        """Start every z near one sane surface: output bias at its rates and log-weights, small
        output weights."""
        r = np.clip(np.asarray(rates, dtype=float), -0.98 * RATE_BOUND, 0.98 * RATE_BOUND)
        logit = np.clip(np.log(np.maximum(weights, 1e-6)), -0.98 * LOGIT_BOUND, None)
        logit -= logit.max()
        bias = np.r_[
            RATE_BOUND * np.arctanh(r / RATE_BOUND).ravel(),
            LOGIT_BOUND * np.arctanh(logit / LOGIT_BOUND),
        ]
        self.out.bias.copy_(torch.as_tensor(bias))
        self.out.weight.mul_(scale)


class ConditionalNet(nn.Module):
    def __init__(self, cfg: ConditionalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("grid", torch.tensor(GRID_DAYS, dtype=torch.float64) / 365.0)
        self.register_buffer("feat_mean", torch.zeros(4, dtype=torch.float64))
        self.register_buffer("feat_std", torch.ones(4, dtype=torch.float64))
        self.encoder = SetEncoder(4, cfg.enc_hidden, cfg.latent_dim)
        self.decoder = SurfaceDecoder(
            cfg.latent_dim, cfg.components, len(GRID_DAYS), cfg.dec_hidden
        )
        self.double()

    def encode(self, q: QuoteBatch) -> torch.Tensor:
        x = (q.features() - self.feat_mean) / self.feat_std
        return self.encoder(x, q.day, q.n_days)

    def iv(self, z: torch.Tensor, q: QuoteBatch) -> torch.Tensor:
        rates, logits = self.decoder(z)
        return batch_iv(rates, logits, self.grid, q)

    def save(self, path, meta: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"config": asdict(self.cfg), "state": self.state_dict(), "meta": meta or {}}, path
        )

    @classmethod
    def load(cls, path) -> ConditionalNet:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = blob["config"]
        cfg["dec_hidden"] = tuple(cfg["dec_hidden"])
        net = cls(ConditionalConfig(**cfg))
        net.load_state_dict(blob["state"])
        net.meta = blob.get("meta", {})
        return net.eval()


_LOADED: dict[str, ConditionalNet] = {}


def load_cached(path) -> ConditionalNet:
    """One copy per process, so per-day evaluation times inference rather than loading."""
    key = str(Path(path).resolve())
    if key not in _LOADED:
        _LOADED[key] = ConditionalNet.load(key)
    return _LOADED[key]


# ---- per-day surface for the evaluation harness ----------------------------------------------


class ConditionalSurface(SurfaceModel):
    """Surface decoded from the latent of one day's observed quotes.

    ``refine_iters = 0`` uses the encoder's z (amortized inference); otherwise z is refined by
    L-BFGS on the spread-normalized fit loss over the observed quotes, starting from the encoding.
    """

    name = "conditional"

    def __init__(self, checkpoint: str, refine_iters: int = 0) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.refine_iters = refine_iters

    def fit(self, data: FitData) -> ConditionalSurface:
        net = load_cached(self.checkpoint)
        grid = net.grid.numpy()
        q = QuoteBatch.from_arrays(
            data.k, data.T, data.iv, data.s, np.zeros(len(data.k), dtype=np.int64), 1, grid
        )
        with torch.no_grad():
            z0 = net.encode(q)
        z = z0.clone()
        if self.refine_iters:
            z.requires_grad_(True)
            _minimize_lbfgs([z], lambda: per_day_loss(net.iv(z, q), q)[0], self.refine_iters)
        with torch.no_grad():
            self.final_loss = float(per_day_loss(net.iv(z, q), q)[0])
            rates, logits = net.decoder(z)
            th, up, um = grid_knots(rates[0], net.grid)
        if not np.isfinite(self.final_loss):
            self.n_failed += 1
        self.z = z.detach().numpy()[0]
        self.knot_T = grid
        self.knots = [(th[j].numpy(), up[j].numpy(), um[j].numpy()) for j in range(th.shape[0])]
        self.weights = torch.softmax(logits[0], -1).numpy()
        return self

    def w(self, k, T) -> np.ndarray:
        k, T = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(T, dtype=float))
        comps = np.stack([essvi_w(k, *interp_knots(T, self.knot_T, *kn)) for kn in self.knots])
        if len(self.knots) == 1:
            return comps[0]
        return mixture_total_variance(k, comps, self.weights)
