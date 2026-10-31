"""Mispricing detection: robust arbitrage-free fits, per-quote residuals and flags.

A least-squares fit absorbs part of a mispriced quote, and more of a mispriced segment, into the
surface. The detector therefore refits each day's `mix4` surface by iteratively reweighted least
squares for a Cauchy loss on spread-normalized residuals: a quote u half-spreads from the surface
gets its half-spread widened by sqrt(1 + (u/c)^2), which is the Cauchy weight, and the surface is
refitted from the previous pass. This mostly protects against isolated mispricings: a mispriced
segment of neighbouring strikes is a smooth bump that one expiry's smile can bend to, so per-day
surfaces absorb much of it whatever the loss. The cross-day conditional model, whose surfaces vary
only through a low-dimensional latent, absorbs very little of either and is scored alongside when a
checkpoint is given. Every quote is scored, including legs of model-free arbitrage, which stay out
of the fit as in the evaluation harness.

Residuals are signed so that positive means the market is rich (quoted above the model). A quote
is flagged when the model price lies outside the bid-ask by more than ``margin`` half-spreads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch

from volsurf.eval.dayset import RESID_FLOOR, FitData
from volsurf.models.conditional import ConditionalSurface
from volsurf.models.neural_essvi import NeuralESSVIConfig, NeuralESSVISurface
from volsurf.pricing import black76

MIX4 = NeuralESSVIConfig(components=4, generator="free", init_scales=(1.0, 1.8, 0.6, 1.3))
QUOTE_COLUMNS = [
    "date", "expiry", "dte", "T", "strike", "option_type", "F", "D", "k",
    "bid", "ask", "mid", "iv_bid", "iv_ask", "iv_mid",
]  # fmt: skip
TRUTH_COLUMNS = ["iv_true", "half_spread_model", "shift", "label", "injection"]


@dataclass
class RobustConfig:
    passes: int = 2  # reweighted refits after the least-squares fit
    scale: float = 1.0  # Cauchy scale c, in half-spreads
    lbfgs_iters: int = 100  # per reweighted refit (each starts from the previous pass)
    refine_iters: int = 100  # latent refinement of the conditional model, when one is scored


@dataclass
class SurfaceState:
    """What a fitted mix4 surface needs to be evaluated or used as a warm start."""

    knot_T: np.ndarray
    knots: list
    weights: np.ndarray

    @classmethod
    def of(cls, model: NeuralESSVISurface) -> SurfaceState:
        return cls(model.knot_T, model.knots, model.weights)

    def model(self) -> NeuralESSVISurface:
        m = NeuralESSVISurface(MIX4)
        m.knot_T, m.knots, m.weights = self.knot_T, self.knots, self.weights
        m.logits = torch.log(torch.as_tensor(self.weights, dtype=torch.float64))
        return m


def cauchy_spreads(s: np.ndarray, u: np.ndarray, scale: float) -> np.ndarray:
    """Half-spreads that turn the squared loss into one IRLS step of a Cauchy loss."""
    return s * np.sqrt(1.0 + (u / scale) ** 2)


def fit_robust(
    data: FitData, cfg: RobustConfig, cold: SurfaceState | None = None
) -> list[SurfaceState]:
    """The least-squares fit (or ``cold``, a cached one), then ``cfg.passes`` reweighted refits,
    each warm-started from the previous pass."""
    states = [cold or SurfaceState.of(NeuralESSVISurface(MIX4).fit(data))]
    refit = replace(MIX4, warm_lbfgs_iters=cfg.lbfgs_iters)
    for _ in range(cfg.passes):
        prev = states[-1].model()
        u = (prev.iv(data.k, data.T) - data.iv) / data.s
        weighted = replace(data, s=cauchy_spreads(data.s, u, cfg.scale))
        states.append(SurfaceState.of(NeuralESSVISurface(refit).fit(weighted, init=prev)))
    return states


def arb_legs(quotes: pd.DataFrame) -> np.ndarray:
    if "arb_vertical" not in quotes:
        return np.zeros(len(quotes), dtype=bool)
    return (quotes["arb_vertical"] | quotes["arb_butterfly"]).to_numpy()


def fit_data(quotes: pd.DataFrame, mask: np.ndarray, resid_floor: float = RESID_FLOOR) -> FitData:
    q = quotes[mask]
    T = q["T"].to_numpy()
    slice_T, slice_id = np.unique(T, return_inverse=True)
    s = np.maximum(q["half_spread_iv"].to_numpy(), resid_floor)
    return FitData(q["k"].to_numpy(), T, q["iv_mid"].to_numpy(), s, slice_id, slice_T)


def score_day(
    quotes: pd.DataFrame,
    cfg: RobustConfig,
    cold: SurfaceState | None = None,
    checkpoint: str | None = None,
    resid_floor: float = RESID_FLOOR,
) -> tuple[pd.DataFrame, list[SurfaceState], float]:
    """Fit one day and return its per-quote table (least-squares and robust model IVs, and the
    refined conditional model's when a checkpoint is given), the mix4 surfaces and fit time."""
    torch.set_num_threads(1)
    q = quotes.sort_values(["T", "k"]).reset_index(drop=True)
    arb = arb_legs(q)
    data = fit_data(q, ~arb, resid_floor)
    start = time.perf_counter()
    states = fit_robust(data, cfg, cold)
    seconds = time.perf_counter() - start
    k, T = q["k"].to_numpy(), q["T"].to_numpy()
    out = q[QUOTE_COLUMNS + [c for c in TRUTH_COLUMNS if c in q]].copy()
    out["s"] = np.maximum(q["half_spread_iv"].to_numpy(), resid_floor)
    out["arb_leg"] = arb
    out["iv_ls"] = states[0].model().iv(k, T)
    out["iv_rob"] = states[-1].model().iv(k, T)
    if checkpoint:  # not reweighted: the encoder reads half-spreads as an input
        cond = ConditionalSurface(checkpoint, refine_iters=cfg.refine_iters).fit(data)
        out["iv_cond"] = cond.iv(k, T)
    return out, states, seconds


def residuals(table: pd.DataFrame, iv_col: str) -> pd.DataFrame:
    """Signed residuals of the market against a model IV column (positive: market rich).

    ``r`` is in price half-spreads, the unit of the flag rule; ``r_iv`` in IV half-spreads;
    ``resid_vp`` in vol points. ``vega`` is per unit of volatility at the market IV."""
    F, K, T, D = (table[c].to_numpy() for c in ("F", "strike", "T", "D"))
    is_call = table["option_type"].to_numpy() == "C"
    model_price = black76.price(F, K, T, table[iv_col].to_numpy(), is_call, D)
    half = 0.5 * (table["ask"].to_numpy() - table["bid"].to_numpy())
    return pd.DataFrame(
        {
            "model_price": model_price,
            "r": (table["mid"].to_numpy() - model_price) / half,
            "r_iv": (table["iv_mid"].to_numpy() - table[iv_col].to_numpy()) / table["s"].to_numpy(),
            "resid_vp": 100.0 * (table["iv_mid"].to_numpy() - table[iv_col].to_numpy()),
            "vega": black76.vega(F, K, T, table["iv_mid"].to_numpy(), D),
        },
        index=table.index,
    )


def flag(r: np.ndarray, margin: float) -> np.ndarray:
    """+1 rich, -1 cheap, 0 unflagged: the model price is outside the bid-ask by more than
    ``margin`` half-spreads."""
    r = np.asarray(r, dtype=float)
    return (np.sign(r) * (np.abs(r) > 1.0 + margin)).astype(np.int8)


def precision_recall(flags: np.ndarray, table: pd.DataFrame) -> dict:
    """Detection quality against injected labels (+1 rich, -1 cheap; |shift| > 1 half-spread).

    Recall is reported on every label and on labels whose true IV lies outside the quoted
    bid-ask (the only ones a quote-based detector can see). Precision is strict, and also with
    flags on decayed but still shifted quotes (0 < |shift| <= 1) set aside."""
    label = table["label"].to_numpy()
    hit = (flags != 0) & (flags == label)
    flagged = flags != 0
    outside = (table["iv_true"] < table["iv_bid"]) | (table["iv_true"] > table["iv_ask"])
    weak = (table["shift"].abs() > 0) & (label == 0)
    out = {
        "flags": int(flagged.sum()),
        "flag_rate": float(flagged.mean()),
        "precision": float(hit.sum() / max(flagged.sum(), 1)),
        "precision_ex_weak": float(hit.sum() / max((flagged & ~weak).sum(), 1)),
        "recall": float(hit.sum() / max((label != 0).sum(), 1)),
        "recall_visible": float((hit & outside).sum() / max(((label != 0) & outside).sum(), 1)),
    }
    for kind in ("point", "segment"):
        is_kind = (table["injection"] == kind).to_numpy() & (label != 0)
        out[f"recall_{kind}"] = float((hit & is_kind).sum() / max(is_kind.sum(), 1))
    return out
