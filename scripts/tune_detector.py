"""Choose the run-pooled robust detector on the synthetic validation year.

Refits a sample of days for every cell of the grids in configs/detector_tuning.yaml, starting each
from the day's cached least-squares mix4 surface, and scores precision, recall and the share of
each injected mispricing the surface absorbs. Also checks that the Phase 7 robust detector is
reproduced exactly from the cache. Writes results/detector-tuning/<source>_<year>.csv; cells
already in that file are not refitted, so extending a grid fits only the new cells.
"""

from __future__ import annotations

import argparse
import itertools
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from volsurf.config import deep_merge, from_dict, load_yaml
from volsurf.eval.mispricing import (
    RobustConfig,
    SurfaceState,
    flag,
    precision_recall,
    residuals,
    score_day,
)
from volsurf.eval.runner import EvalConfig, SamplingConfig, ScreenConfig, load_eval_days


def cell_name(cell: dict) -> str:
    return "p{pool}_c{scale:g}_n{passes}".format(**cell)


def _job(args):
    quotes, cfg, cold, variants = args
    table, _, _ = score_day(quotes, cfg, cold, variants=variants)
    return table


def scores(table: pd.DataFrame, detector: str, margin: float) -> dict:
    labelled = (table["label"] != 0).to_numpy()
    f = flag(residuals(table, detector)["r"].to_numpy(), margin)
    share = ((table[detector] - table["iv_true"]) / (table["iv_mid"] - table["iv_true"])).to_numpy()
    row = {"detector": detector, **precision_recall(f, table)}
    for kind in ("point", "segment"):
        sel = labelled & (table["injection"] == kind).to_numpy()
        row[f"absorbed_{kind}"] = float(np.median(share[sel]))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/detector_tuning.yaml")
    parser.add_argument("--fits", default="results/phase7-fits")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-days", type=int, help="first n sampled days only (smoke runs)")
    parser.add_argument("--out", default="results/detector-tuning")
    args = parser.parse_args()

    spec = load_yaml(args.config)
    cfg = from_dict(RobustConfig, spec["robust"])
    cells = {
        cell_name(c): c
        for grid in spec["grids"]
        for c in (dict(zip(grid, v, strict=True)) for v in itertools.product(*grid.values()))
    }
    name = spec["source"].replace(":", "_")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_{spec['year']}.csv"
    done = pd.read_csv(path) if path.exists() and not args.max_days else pd.DataFrame()
    known = set(done.get("detector", []))
    todo = {
        n: from_dict(RobustConfig, deep_merge(spec["robust"], c))
        for n, c in cells.items()
        if f"iv_{n}" not in known
    }

    fits = pickle.loads((Path(args.fits) / name / "fits.pkl").read_bytes())
    frozen = pd.read_parquet(Path(args.fits) / name / f"{spec['year']}.parquet")
    screen = EvalConfig(
        sampling=SamplingConfig(every=1, block=1), screen=ScreenConfig(drop_flagged_quotes=False)
    )
    days, _, _ = load_eval_days(spec["source"], [spec["year"]], screen)
    dates = sorted(days)[:: spec["every"]][: args.max_days]
    jobs = [(days[d], cfg, SurfaceState(*fits[d][0]), todo) for d in dates]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        table = pd.concat(list(pool.map(_job, jobs)), ignore_index=True)

    keys = ["date", "expiry", "strike", "option_type"]
    check = table[keys + ["iv_rob"]].merge(frozen[keys + ["iv_rob"]], on=keys, suffixes=("", "_7"))
    gap = float(np.nanmax(np.abs(check["iv_rob"] - check["iv_rob_7"])))
    print(f"{len(dates)} days, {len(table)} quotes, {len(todo)} new cells; iv_rob vs Phase 7: "
          f"{len(check)} matched, max |diff| {gap:.2e}")  # fmt: skip

    fresh = pd.DataFrame(
        [scores(table, d, spec["margin"]) for d in ["iv_ls", "iv_rob", *(f"iv_{n}" for n in todo)]]
    )
    if len(done):  # the same days and fits, so the anchors must match the earlier run exactly
        anchors = done.set_index("detector").loc[["iv_ls", "iv_rob"], fresh.columns[1:]]
        assert np.allclose(anchors, fresh.set_index("detector").loc[["iv_ls", "iv_rob"]])
        done = done[~done["detector"].isin(["iv_ls", "iv_rob"])].drop(columns="eligible")
    result = pd.concat([fresh.iloc[:2], done, fresh.iloc[2:]], ignore_index=True)
    base = result.set_index("detector").loc["iv_rob"]
    result["eligible"] = (result["precision"] >= base["precision"]) & (
        result["recall_point"] >= base["recall_point"]
    )
    result.to_csv(path if not args.max_days else out / f"{name}_{spec['year']}_smoke.csv",
                  index=False)  # fmt: skip
    cols = ["detector", "precision", "recall", "recall_point", "recall_segment",
            "absorbed_point", "absorbed_segment", "eligible"]  # fmt: skip
    print(result[cols].round(3).to_string(index=False))
    eligible = result[result["eligible"] & result["detector"].str.startswith("iv_p")]
    if len(eligible):
        print("selected:", eligible.sort_values("recall").iloc[-1]["detector"])


if __name__ == "__main__":
    main()
