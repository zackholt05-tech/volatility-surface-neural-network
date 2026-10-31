"""Cross-day training of the conditional surface model.

Each step encodes a random *context* subset of every sampled day's quotes and scores the decoded
surface on a random subset of all of that day's quotes (neural-process style), so the encoder learns
to infer a whole surface from a partial chain. Contexts combine the ways quotes go missing in
evaluation: outer wings cut off, interior strikes dropped at random and whole maturities dropped,
with observed shares ranging from sparse to complete chains. Early stopping uses the last months of
the training years, encoded exactly as the evaluation harness does (its fit mask as context).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from volsurf.eval.dayset import RESID_FLOOR, FitData, HoldoutConfig, _holdout_masks, date_seed
from volsurf.eval.runner import EvalConfig, SamplingConfig, load_eval_days
from volsurf.models.conditional import (
    ConditionalConfig,
    ConditionalNet,
    QuoteBatch,
    per_day_loss,
)
from volsurf.models.essvi import interp_knots, rates_from_knots
from volsurf.models.neural_essvi import NeuralESSVIConfig, NeuralESSVISurface


@dataclass
class ContextConfig:
    wing_max: float = 0.2  # per-side share of each slice's outermost strikes cut, ~ U(0, wing_max)
    interior_max: float = 0.9  # share of the remaining strikes dropped, ~ U(0, interior_max)
    slice_prob: float = 0.3  # probability that a day also loses whole maturities
    slice_max: float = 0.5  # share of maturities dropped when it does, ~ U(0, slice_max)
    min_quotes: int = 30


@dataclass
class TrainConfig:
    source: str = "spx"
    years: tuple[int, ...] = (2018, 2019, 2020, 2021)
    val_start: str = "2021-10-01"  # days from here on are held out for early stopping
    model: ConditionalConfig = field(default_factory=ConditionalConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    batch_days: int = 16
    target_quotes: int = 1024
    context_cap: int = 2048
    lr: float = 1e-3
    epochs: int = 300
    patience: int = 30
    grad_clip: float = 10.0
    seed: int = 0
    threads: int = 8
    max_days: int | None = None  # cap on training days, for smoke runs


@dataclass
class TrainDay:
    date: pd.Timestamp
    k: np.ndarray
    T: np.ndarray
    iv: np.ndarray
    s: np.ndarray
    slice_id: np.ndarray
    rank: np.ndarray  # distance in strikes from the nearer end of the slice
    size: np.ndarray  # number of quotes in the quote's slice
    fit: np.ndarray  # the evaluation harness's fit mask for this date


def prepare_days(source: str, years, resid_floor: float = RESID_FLOOR) -> list[TrainDay]:
    """Every screened day, arranged as ``make_day`` does (sorted by T then k), without its
    minimum one-quote wing holdout, which training must not impose."""
    cfg = EvalConfig(sampling=SamplingConfig(every=1, block=1))
    days, _, _ = load_eval_days(source, years, cfg)
    out = []
    for date, quotes in sorted(days.items()):
        q = quotes.sort_values(["T", "k"]).reset_index(drop=True)
        T = q["T"].to_numpy()
        _, slice_id = np.unique(T, return_inverse=True)
        starts = np.flatnonzero(np.r_[True, slice_id[1:] != slice_id[:-1]])
        sizes = np.diff(np.r_[starts, len(T)])
        pos = np.arange(len(T)) - starts[slice_id]
        size = sizes[slice_id]
        rng = np.random.default_rng(date_seed(date))
        fit, _, _ = _holdout_masks(slice_id, HoldoutConfig(), rng)
        out.append(
            TrainDay(
                date=pd.Timestamp(date),
                k=q["k"].to_numpy(),
                T=T,
                iv=q["iv_mid"].to_numpy(),
                s=np.maximum(q["half_spread_iv"].to_numpy(), resid_floor),
                slice_id=slice_id,
                rank=np.minimum(pos, size - 1 - pos),
                size=size,
                fit=fit,
            )
        )
    return out


def sample_context(day: TrainDay, cfg: ContextConfig, rng: np.random.Generator) -> np.ndarray:
    keep = day.rank >= rng.uniform(0.0, cfg.wing_max) * day.size
    keep &= rng.random(len(day.k)) >= rng.uniform(0.0, cfg.interior_max)
    n_slices = day.slice_id.max() + 1
    if rng.random() < cfg.slice_prob and n_slices > 3:
        dropped = rng.random(n_slices) < rng.uniform(0.0, cfg.slice_max)
        if (~dropped).sum() >= 3:
            keep &= ~dropped[day.slice_id]
    if keep.sum() < cfg.min_quotes:
        keep = day.rank >= 0.1 * day.size
    return keep


def make_batch(days, masks, grid: np.ndarray, cap: int | None, rng) -> QuoteBatch:
    parts = []
    for i, (day, mask) in enumerate(zip(days, masks, strict=True)):
        idx = np.flatnonzero(mask)
        if cap is not None and len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        parts.append((day.k[idx], day.T[idx], day.iv[idx], day.s[idx], np.full(len(idx), i)))
    k, T, iv, s, d = (np.concatenate(a) for a in zip(*parts, strict=True))
    return QuoteBatch.from_arrays(k, T, iv, s, d, len(days), grid)


def initial_surface(day: TrainDay, grid: np.ndarray, components: int):
    """Rates on the grid and weights of a per-day free-knot mixture fitted to one day."""
    scales = (1.0, 1.8, 0.6, 1.3) if components == 4 else NeuralESSVIConfig().init_scales
    cfg = NeuralESSVIConfig(components=components, generator="free", init_scales=scales)
    slice_T = np.unique(day.T)
    data = FitData(day.k, day.T, day.iv, day.s, day.slice_id, slice_T)
    fit = NeuralESSVISurface(cfg).fit(data)
    rates = np.stack(
        [rates_from_knots(grid, *interp_knots(grid, fit.knot_T, *kn)) for kn in fit.knots]
    )
    return rates, fit.weights


def feature_stats(days: list[TrainDay], grid: np.ndarray):
    k, T, iv, s = (np.concatenate([getattr(d, a) for d in days]) for a in ("k", "T", "iv", "s"))
    q = QuoteBatch.from_arrays(k, T, iv, s, np.zeros(len(k), dtype=np.int64), 1, grid)
    x = q.features()
    return x.mean(0), x.std(0)


def evaluate(net: ConditionalNet, days: list[TrainDay], batch_days: int) -> dict:
    """Per-day loss and IV RMSE with the harness's fit mask as context and every quote as
    target."""
    grid = net.grid.numpy()
    losses, sq, n = [], 0.0, 0
    with torch.no_grad():
        for i in range(0, len(days), batch_days):
            chunk = days[i : i + batch_days]
            ctx = make_batch(chunk, [d.fit for d in chunk], grid, None, None)
            tgt = make_batch(chunk, [np.ones(len(d.k), bool) for d in chunk], grid, None, None)
            iv = net.iv(net.encode(ctx), tgt)
            losses.append(per_day_loss(iv, tgt).numpy())
            sq += float(torch.sum((iv - tgt.iv) ** 2))
            n += len(tgt.iv)
    losses = np.concatenate(losses)
    finite = np.isfinite(losses)
    return {
        "loss": float(losses[finite].mean()),
        "rmse_vp": 100.0 * np.sqrt(sq / n),
        "nonfinite_days": int((~finite).sum()),
    }


def train(cfg: TrainConfig, log=print, on_epoch=None) -> tuple[ConditionalNet, dict]:
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed)

    days = prepare_days(cfg.source, cfg.years)
    val_start = pd.Timestamp(cfg.val_start)
    train_days = [d for d in days if d.date < val_start]
    val_days = [d for d in days if d.date >= val_start]
    if cfg.max_days:
        train_days = train_days[:: max(1, len(train_days) // cfg.max_days)][: cfg.max_days]
        val_days = val_days[:: max(1, len(val_days) // max(cfg.max_days // 4, 1))]
    log(f"{len(train_days)} training days, {len(val_days)} early-stopping days")

    net = ConditionalNet(cfg.model)
    grid = net.grid.numpy()
    mean, std = feature_stats(train_days, grid)
    net.feat_mean.copy_(mean)
    net.feat_std.copy_(std)
    levels = [np.median(d.iv) for d in train_days]
    typical = train_days[int(np.argsort(levels)[len(levels) // 2])]
    rates, weights = initial_surface(typical, grid, cfg.model.components)
    net.decoder.init_output(rates, weights)
    log(f"decoder initialized from {typical.date.date()}")

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    best = {"loss": np.inf}
    best_state, last_good = None, copy.deepcopy(net.state_dict())
    stale, history = 0, []
    for epoch in range(cfg.epochs):
        start = time.perf_counter()
        net.train()
        order = rng.permutation(len(train_days))
        losses, dropped, rollbacks = [], 0, 0
        for b in range(0, len(order), cfg.batch_days):
            chunk = [train_days[i] for i in order[b : b + cfg.batch_days]]
            ctx = make_batch(
                chunk,
                [sample_context(d, cfg.context, rng) for d in chunk],
                grid,
                cfg.context_cap,
                rng,
            )
            tgt = make_batch(
                chunk, [np.ones(len(d.k), bool) for d in chunk], grid, cfg.target_quotes, rng
            )
            per_day = per_day_loss(net.iv(net.encode(ctx), tgt), tgt)
            finite = torch.isfinite(per_day)
            dropped += int((~finite).sum())
            if not finite.any():
                continue
            loss = per_day[finite].mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            if all(torch.isfinite(p).all() for p in net.parameters()):
                last_good = copy.deepcopy(net.state_dict())
                losses.append(float(loss))
            else:  # one bad step must not poison the shared weights
                net.load_state_dict(last_good)
                rollbacks += 1

        net.eval()
        val = evaluate(net, val_days, cfg.batch_days)
        sched.step(val["loss"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_loss": val["loss"],
            "val_rmse_vp": val["rmse_vp"],
            "val_nonfinite_days": val["nonfinite_days"],
            "dropped_days": dropped,
            "rollbacks": rollbacks,
            "lr": opt.param_groups[0]["lr"],
            "seconds": time.perf_counter() - start,
        }
        history.append(row)
        log(
            " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
        )
        if on_epoch:
            on_epoch(row)
        if val["loss"] < best["loss"]:
            best = {**val, "epoch": epoch}
            best_state = copy.deepcopy(net.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    net.load_state_dict(best_state)
    net.meta = {"best": best, "history": history, "init_day": str(typical.date.date())}
    return net.eval(), best
