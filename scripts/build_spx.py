"""Build the processed SPX quote and forward tables from OptionsDX archives."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from volsurf.config import load_yaml
from volsurf.data import optionsdx, store
from volsurf.data.spx import SOURCE, SpxConfig, process_month


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_spx.yaml")
    parser.add_argument("--months", nargs="*", help="YYYYMM keys to (re)build (default: all)")
    parser.add_argument("--workers", type=int, help="override the configured worker count")
    args = parser.parse_args()

    cfg = SpxConfig.from_dict(load_yaml(args.config))
    for archive in optionsdx.extract_archives(cfg.raw_dir, cfg.extract_dir):
        print(f"extracted {archive.name}")

    years = list(cfg.years) if cfg.years else None
    files = optionsdx.month_files(cfg.extract_dir, years)
    if args.months:
        files = [f for f in files if optionsdx.month_key(f) in args.months]
    if not files:
        raise SystemExit(f"no monthly files found in {cfg.extract_dir}")

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers or cfg.workers) as pool:
        rows = list(pool.map(process_month, files, [cfg] * len(files)))
    summary = pd.DataFrame(rows).sort_values("month")

    path = store.table_dir(SOURCE, "summary").with_suffix(".csv")
    if args.months and path.exists():
        previous = pd.read_csv(path, dtype={"month": str})
        summary = pd.concat([previous[~previous["month"].isin(summary["month"])], summary])
        summary = summary.sort_values("month")
    summary.to_csv(path, index=False)

    total = summary[["contracts", "slice_size", "days", "slices"]].sum()
    print(
        f"{len(rows)} months in {time.perf_counter() - start:.0f}s: "
        f"{total['slice_size']:,} quotes kept of {total['contracts']:,} contracts, "
        f"{total['days']} days, {total['slices']:,} slices -> {path}"
    )


if __name__ == "__main__":
    main()
