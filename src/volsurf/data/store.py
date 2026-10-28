"""Reading and writing processed datasets.

Real data lives in ``data/processed/<source>/<table>/<YYYYMM>.parquet`` (tables: ``quotes``,
``forwards``). Synthetic panels live in ``data/synthetic/<name>.parquet``. ``load_quotes`` returns
either kind in the shared schema, selected by ``source="spx"`` or ``source="synthetic:<name>"``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_ROOT = Path("data")


def table_dir(source: str, table: str, root: Path = DATA_ROOT) -> Path:
    return root / "processed" / source / table


def write_partition(df: pd.DataFrame, source: str, table: str, key: str, root=DATA_ROOT) -> Path:
    path = table_dir(source, table, root) / f"{key}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_table(
    source: str,
    table: str,
    years: list[int] | None = None,
    columns: list[str] | None = None,
    root: Path = DATA_ROOT,
) -> pd.DataFrame:
    files = sorted(table_dir(source, table, root).glob("*.parquet"))
    if years is not None:
        files = [f for f in files if int(f.stem[:4]) in years]
    if not files:
        raise FileNotFoundError(f"no {table} partitions for {source!r} under {root}")
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)


def load_quotes(
    source: str = "spx",
    years: list[int] | None = None,
    columns: list[str] | None = None,
    root: Path = DATA_ROOT,
) -> pd.DataFrame:
    if source.startswith("synthetic:"):
        quotes = pd.read_parquet(
            root / "synthetic" / f"{source.split(':', 1)[1]}.parquet", columns=columns
        )
        if years is not None:
            quotes = quotes[quotes["date"].dt.year.isin(years)].reset_index(drop=True)
        return quotes
    return load_table(source, "quotes", years, columns, root)
