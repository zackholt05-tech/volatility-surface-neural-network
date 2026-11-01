"""Quarterly walk-forward refits of the cross-day conditional model.

For each quarter of the chosen split, fine-tunes the base checkpoint on every screened day dated
before the quarter starts and saves a checkpoint per snapshot epoch count to
artifacts/conditional/walkforward/<name>_e<epochs>_<quarter>.pt. The evaluation harness fills
``{quarter}`` in a checkpoint path with each day's quarter, so each day is scored by its quarter's
refit. Quarters whose checkpoints all exist are skipped.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from volsurf.config import from_dict, load_yaml
from volsurf.models.conditional import ConditionalNet
from volsurf.training.conditional import TrainConfig, fine_tune, prepare_days


def quarter_start(quarter: str) -> pd.Timestamp:
    year, n = quarter.split("Q")
    return pd.Timestamp(int(year), 3 * int(n) - 2, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/walkforward_spx.yaml")
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--snapshots", type=int, nargs="+", help="override (smoke runs)")
    parser.add_argument("--max-days", type=int, help="cap training days (smoke runs)")
    parser.add_argument("--out", default="artifacts/conditional/walkforward")
    args = parser.parse_args()

    spec = load_yaml(args.config)
    cfg = replace(from_dict(TrainConfig, spec["train"]), source=spec["source"])
    snapshots = tuple(args.snapshots or spec["snapshots"])
    quarters = spec["quarters"][args.split]
    out = Path(args.out)
    last_year = max(int(q[:4]) for q in quarters)
    days = prepare_days(spec["source"], list(range(spec["first_year"], last_year + 1)))

    for quarter in quarters:
        paths = {e: out / f"{spec['name']}_e{e}_{quarter}.pt" for e in snapshots}
        if all(p.exists() for p in paths.values()):
            print(f"{quarter}: checkpoints exist, skipped")
            continue
        train_days = [d for d in days if d.date < quarter_start(quarter)]
        if args.max_days:
            train_days = train_days[-args.max_days :]
        print(f"{quarter}: {len(train_days)} training days, {train_days[0].date.date()} to "
              f"{train_days[-1].date.date()}", flush=True)  # fmt: skip
        net = ConditionalNet.load(spec["base"])
        states, history = fine_tune(net, train_days, cfg, snapshots, log=print)
        for epochs, state in states.items():
            net.load_state_dict(state)
            net.save(paths[epochs], {
                "quarter": quarter, "base": spec["base"], "epochs": epochs,
                "train_days": len(train_days), "train_end": str(train_days[-1].date.date()),
                "history": history[:epochs],
            })  # fmt: skip
            print(f"  -> {paths[epochs]}", flush=True)


if __name__ == "__main__":
    main()
