"""Detection quality, residual reversion and the delta-hedged backtest for one split.

Reads the per-quote tables of scripts/fit_panel.py and writes, per panel, to
results/<experiment>-<split>/<source>/ (the config's ``experiment``, phase7 by default):

- detection.csv: flag rates for every detector and margin, and on synthetic panels precision and
  recall against the injected labels;
- absorption.csv (synthetic panels): the share of each injected mispricing a detector's surface
  absorbs, by kind, and precision and recall by quarter;
- reversion.csv: Fama-MacBeth reversion slopes of flagged against unflagged quotes;
- reversion_parts.csv: those slopes split into the market's move and the model's;
- backtest.csv (panels with a ``backtest`` entry): the strategy with next-day and same-day
  execution, its matched placebo and the test of their difference; on synthetic panels also the
  true labels traded at the signal's close, a check of the accounting (a perfect detector must make
  money net of costs).

Detection and reversion use the panel's ``margin``, or its backtest margin. A panel is skipped for
a split it has no year for; a split listing several years pools their tables into one panel.
Everything is set in the config and was chosen on the validation years.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from volsurf.config import from_dict, load_yaml
from volsurf.eval.backtest import (
    BacktestConfig,
    Market,
    against_placebo,
    performance,
    run_placebo,
    run_signals,
)
from volsurf.eval.mispricing import flag, precision_recall, residuals
from volsurf.eval.reversion import (
    daily_slopes,
    decomposed_slopes,
    summarize_decomposed,
    summarize_slopes,
)


def detection(table: pd.DataFrame, detectors, margins) -> pd.DataFrame:
    rows = []
    synthetic = "label" in table
    for det in detectors:
        r = residuals(table, det)["r"].to_numpy()
        for m in margins:
            f = flag(r, m)
            row = {"detector": det, "margin": m, "flag_rate": float((f != 0).mean())}
            if synthetic:
                row |= precision_recall(f, table)
            rows.append(row)
    return pd.DataFrame(rows)


def absorption(table: pd.DataFrame, detectors, margin: float) -> pd.DataFrame:
    """Synthetic panels: the median share of each injected shift a detector's surface absorbs,
    (model - truth) / (market - truth) on labelled quotes, by kind; and precision and recall at
    ``margin`` by quarter."""
    labelled = (table["label"] != 0).to_numpy()
    rows = []
    for det in detectors:
        share = ((table[det] - table["iv_true"]) / (table["iv_mid"] - table["iv_true"])).to_numpy()
        row = {"detector": det, "quarter": "all"}
        for kind in ("point", "segment"):
            row[f"absorbed_{kind}"] = float(
                np.median(share[labelled & (table["injection"] == kind).to_numpy()])
            )
        rows.append(row)
        f = flag(residuals(table, det)["r"].to_numpy(), margin)
        quarter = table["date"].dt.quarter.to_numpy()
        for q in np.unique(quarter):
            sel = quarter == q
            pr = precision_recall(f[sel], table[sel])
            rows.append({"detector": det, "quarter": int(q), "precision": pr["precision"],
                         "recall": pr["recall"]})  # fmt: skip
    return pd.DataFrame(rows)


def reversion(table: pd.DataFrame, detectors, margin, horizons) -> pd.DataFrame:
    quotes = table[~table["arb_leg"]].reset_index(drop=True)
    rows = []
    for det in detectors:
        r = residuals(quotes, det)["r"].to_numpy()
        f = flag(r, margin)
        for h in horizons:
            slopes = summarize_slopes(daily_slopes(quotes, r, f, h), h)
            rows.append({"detector": det, "margin": margin, **slopes})
    return pd.DataFrame(rows)


def reversion_parts(table: pd.DataFrame, detectors, margin, horizons) -> pd.DataFrame:
    """The reversion slopes split into the market's move and the model's (`decomposed_slopes`)."""
    quotes = table[~table["arb_leg"]].reset_index(drop=True)
    rows = []
    for det in detectors:
        f = flag(residuals(quotes, det)["r"].to_numpy(), margin)
        for h in horizons:
            parts = summarize_decomposed(decomposed_slopes(quotes, det, f, h), h)
            rows.append({"detector": det, "margin": margin, **parts})
    return pd.DataFrame(rows)


def backtest(table: pd.DataFrame, detector: str, spec: dict) -> pd.DataFrame:
    lags = spec.pop("lags")
    base = from_dict(BacktestConfig, spec)
    mkt = Market(table)
    n_days = len(mkt.dates)
    tradable = ~table["arb_leg"].to_numpy()
    r = residuals(table, detector)["r"].to_numpy()
    runs = [(f"{detector}, lag {lag}", flag(r, base.margin), np.abs(r) < base.exit_r,
             replace(base, lag=lag), np.abs(r) <= 1.0) for lag in lags]  # fmt: skip
    if "label" in table:  # the accounting check: trade the truth at the signal's close
        label = table["label"].to_numpy()
        runs.append(("true labels, lag 0", label, label == 0, replace(base, lag=0), label == 0))
    rows = []
    for name, signal, revert, cfg, unflagged in runs:
        trades = run_signals(table, signal, revert, cfg, tradable=tradable, mkt=mkt)
        placebos = [
            run_placebo(table, trades, unflagged & tradable, cfg, seed=i, mkt=mkt)
            for i in range(cfg.placebo_draws)
        ]
        placebo = performance(pd.concat(placebos, ignore_index=True), n_days)
        rows.append(
            {"run": name, **performance(trades, n_days)}
            | {
                "placebo_gross": placebo["gross"] / cfg.placebo_draws,
                "placebo_sharpe": placebo["sharpe"],
            }  # fmt: skip
            | against_placebo(trades, placebos, n_days, cfg.hold)
        )
        print(pd.DataFrame(rows[-1:]).round(3).to_string(index=False), flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase7.yaml")
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--sources", nargs="*", help="panels to run (default: all)")
    parser.add_argument("--fits", help="per-quote tables (default: results/<experiment>-fits)")
    args = parser.parse_args()

    spec = load_yaml(args.config)
    args.fits = args.fits or f"results/{spec.get('experiment', 'phase7')}-fits"
    for source, panel in spec["panels"].items():
        if (args.sources and source not in args.sources) or args.split not in panel:
            continue
        name = source.replace(":", "_")
        years = panel[args.split] if isinstance(panel[args.split], list) else [panel[args.split]]
        table = pd.concat(
            [pd.read_parquet(Path(args.fits) / name / f"{y}.parquet") for y in years],
            ignore_index=True,
        )
        out = Path("results") / f"{spec.get('experiment', 'phase7')}-{args.split}" / name
        margin = panel.get("margin", panel.get("backtest", {}).get("margin"))
        out.mkdir(parents=True, exist_ok=True)
        print(
            f"== {source} {panel[args.split]}: {table['date'].nunique()} days, {len(table)} quotes"
        )

        det = detection(table, panel["detectors"], spec["detection"]["margins"])
        det.to_csv(out / "detection.csv", index=False)
        print(det.round(3).to_string(index=False))
        if "label" in table:
            ab = absorption(table, panel["detectors"], margin)
            ab.to_csv(out / "absorption.csv", index=False)
            print(ab.round(3).to_string(index=False))
        rev = reversion(table, panel["detectors"], margin, spec["reversion"]["horizons"])
        rev.to_csv(out / "reversion.csv", index=False)
        print(rev.round(3).to_string(index=False))
        parts = reversion_parts(table, panel["detectors"], margin, spec["reversion"]["horizons"])
        parts.to_csv(out / "reversion_parts.csv", index=False)
        print(parts.round(3).to_string(index=False), flush=True)
        if "backtest" in panel:
            bt = backtest(table, panel["detectors"][0], dict(panel["backtest"]))
            bt.to_csv(out / "backtest.csv", index=False)


if __name__ == "__main__":
    main()
