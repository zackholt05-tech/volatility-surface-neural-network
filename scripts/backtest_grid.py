"""Reapply the Phase 7 backtest selection rule to a validation year.

The SPX backtest settings in configs/phase7.yaml were chosen on SPX 2022 by next-day net Sharpe
over margin {2, 4, 6} x hold {1, 5}, exiting once the residual is below half a spread. This runs
the same grid with the config's other settings on the config's fits of the validation year and
reports the cell the rule picks. Writes results/<experiment>-validation/<source>/backtest_grid.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from mispricing_report import backtest

from volsurf.config import load_yaml

MARGINS = (2.0, 4.0, 6.0)
HOLDS = (1, 5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase7.yaml")
    parser.add_argument("--source", default="spx")
    args = parser.parse_args()

    spec = load_yaml(args.config)
    experiment = spec.get("experiment", "phase7")
    panel = spec["panels"][args.source]
    name = args.source.replace(":", "_")
    year = panel["validation"]
    table = pd.read_parquet(Path("results") / f"{experiment}-fits" / name / f"{year}.parquet")
    rows = []
    for margin in MARGINS:
        for hold in HOLDS:
            cell = dict(panel["backtest"], margin=margin, hold=hold, lags=[1])
            out = backtest(table, panel["detectors"][0], cell)
            rows.append({"margin": margin, "hold": hold, **out.iloc[0].to_dict()})
    grid = pd.DataFrame(rows)
    best = grid.loc[grid["sharpe"].idxmax()]
    out_dir = Path("results") / f"{experiment}-validation" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out_dir / "backtest_grid.csv", index=False)
    print(grid[["margin", "hold", "trades", "gross", "cost", "net", "sharpe"]].round(3).to_string())
    print(f"selected: margin {best['margin']:g}, hold {best['hold']:g} (net Sharpe "
          f"{best['sharpe']:.2f}); frozen: margin {panel['backtest']['margin']:g}, hold "
          f"{panel['backtest']['hold']:g}")  # fmt: skip


if __name__ == "__main__":
    main()
