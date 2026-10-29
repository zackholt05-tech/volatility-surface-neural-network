"""Compare models across evaluation experiments that ran on the same sampled days.

Each model is taken from the first listed experiment that contains it. A model found in several
experiments is an anchor: its per-day metrics must agree, which confirms that the experiments share
sampled days, holdouts and screening, so their models can be compared without refitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from volsurf.eval.runner import key_columns, summarize

ANCHOR_COLUMNS = ["fit/rmse_vp", "interp/rmse_vp", "wing/rmse_vp", "arb_ext/calendar_frac"]
SOURCES = ["spx", "synthetic_heston", "synthetic_ssvi", "synthetic_heston_clean"]


def check_anchor(model: str, kept: pd.DataFrame, other: pd.DataFrame, label: str, atol: float):
    a, b = kept.set_index("date"), other.set_index("date")
    if set(a.index) != set(b.index):
        raise SystemExit(f"anchor {model} ({label}): sampled days differ")
    diff = (a[ANCHOR_COLUMNS] - b.loc[a.index, ANCHOR_COLUMNS]).abs().max().max()
    print(f"anchor {model} ({label}): {len(a)} days, max |diff| {diff:.1e}")
    if not diff <= atol:
        raise SystemExit(f"anchor {model} ({label}): per-day metrics differ by {diff:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--models", nargs="*", help="models to keep, in table order")
    parser.add_argument("--sources", nargs="*", default=SOURCES)
    parser.add_argument("--out", help="write the combined tables to results/<OUT>/<source>/")
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    for source in args.sources:
        kept: dict[str, tuple[str, pd.DataFrame, pd.DataFrame]] = {}
        for experiment in args.experiments:
            path = Path("results") / experiment / source
            if not (path / "daily.csv").exists():
                continue
            daily = pd.read_csv(path / "daily.csv")
            stability = pd.read_csv(path / "stability.csv")
            for model, rows in daily.groupby("model"):
                if args.models and model not in args.models:
                    continue
                if model in kept:
                    label = f"{source}: {kept[model][0]} vs {experiment}"
                    check_anchor(model, kept[model][1], rows, label, args.atol)
                    continue
                kept[model] = (experiment, rows, stability[stability["model"] == model])
        if not kept:
            continue

        dates = {m: frozenset(rows["date"]) for m, (_, rows, _) in kept.items()}
        if len(set(dates.values())) > 1:
            raise SystemExit(f"{source}: models were evaluated on different days")
        daily = pd.concat([rows for _, rows, _ in kept.values()], ignore_index=True)
        stability = pd.concat([stab for _, _, stab in kept.values()], ignore_index=True)
        summary = summarize(daily, stability)
        order = [m for m in (args.models or summary.index) if m in summary.index]
        summary = summary.loc[order]

        print(f"\n== {source}: " + ", ".join(f"{m} <- {e}" for m, (e, _, _) in kept.items()))
        print(key_columns(summary).T.to_string())
        if args.out:
            out = Path("results") / args.out / source
            out.mkdir(parents=True, exist_ok=True)
            daily.to_csv(out / "daily.csv", index=False)
            stability.to_csv(out / "stability.csv", index=False)
            summary.to_csv(out / "summary.csv")


if __name__ == "__main__":
    main()
