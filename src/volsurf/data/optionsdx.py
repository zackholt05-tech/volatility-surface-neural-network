"""OptionsDX SPX end-of-day chains: archive extraction and parsing.

OptionsDX ships one comma-separated ``.txt`` per month inside yearly or quarterly ``.7z``
archives. Each row holds the call and put quotes for one (quote date, expiry, strike) at the
16:00 ET snapshot. The SPX (AM-settled) and SPXW (PM-settled) roots are already merged into one row
per strike, and every expiry is stamped 16:00 ET.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import py7zr

SECONDS_PER_YEAR = 365.0 * 86400.0
MONTH_FILE = re.compile(r"spx_eod_(\d{4})(\d{2})\.txt$")

# Vendor column -> internal name. Greeks, last prices and strike distances are not used.
RAW_COLUMNS = {
    "QUOTE_UNIXTIME": "quote_unix",
    "QUOTE_DATE": "date",
    "UNDERLYING_LAST": "spot",
    "EXPIRE_DATE": "expiry",
    "EXPIRE_UNIX": "expire_unix",
    "STRIKE": "strike",
    "C_BID": "c_bid",
    "C_ASK": "c_ask",
    "C_SIZE": "c_size",
    "C_IV": "c_iv_vendor",
    "C_VOLUME": "c_volume",
    "P_BID": "p_bid",
    "P_ASK": "p_ask",
    "P_SIZE": "p_size",
    "P_IV": "p_iv_vendor",
    "P_VOLUME": "p_volume",
}


def extract_archives(raw_dir: str | Path, out_dir: str | Path) -> list[Path]:
    """Extract every ``.7z`` in ``raw_dir`` whose monthly files are not all present yet.

    Whole archives are extracted at once: they are solid, so per-member extraction would
    re-decompress the stream for every file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    for archive in sorted(Path(raw_dir).glob("*.7z")):
        with py7zr.SevenZipFile(archive) as z:
            names = [i.filename for i in z.list() if not i.is_directory]
            if all((out_dir / n).exists() for n in names):
                continue
            z.extractall(path=out_dir)
        extracted.append(archive)
    return extracted


def month_files(extract_dir: str | Path, years: list[int] | None = None) -> list[Path]:
    """Monthly ``spx_eod_YYYYMM.txt`` files in chronological order, optionally by year."""
    files = []
    for path in sorted(Path(extract_dir).glob("spx_eod_*.txt")):
        match = MONTH_FILE.search(path.name)
        if match and (years is None or int(match.group(1)) in years):
            files.append(path)
    return files


def month_key(path: str | Path) -> str:
    """``YYYYMM`` identifier of a monthly file."""
    match = MONTH_FILE.search(Path(path).name)
    if match is None:
        raise ValueError(f"not an OptionsDX monthly file: {path}")
    return match.group(1) + match.group(2)


def _split_size(size: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = size.astype("string").str.split(" x ", n=1, expand=True)
    to_num = lambda s: pd.to_numeric(s.str.strip(), errors="coerce")  # noqa: E731
    return to_num(parts[0]), to_num(parts[1])


def read_month(path: str | Path) -> pd.DataFrame:
    """Parse one monthly file into a wide frame: one row per (date, expiry, strike).

    Adds ``T`` (exact year fraction from the vendor timestamps, so DST shifts are handled), ``dte``
    in calendar days, and ``is_third_friday`` marking expiries where SPX and SPXW coincide.
    """
    df = pd.read_csv(
        path,
        skipinitialspace=True,
        usecols=lambda c: c.strip().strip("[]") in RAW_COLUMNS,
        dtype={c: "string" for c in ("[C_SIZE]", "[P_SIZE]")},
        na_values=[""],
    )
    df.columns = [RAW_COLUMNS[c.strip().strip("[]")] for c in df.columns]

    for col in ("spot", "strike", "c_bid", "c_ask", "c_iv_vendor", "c_volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("p_bid", "p_ask", "p_iv_vendor", "p_volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["c_bid_size"], df["c_ask_size"] = _split_size(df.pop("c_size"))
    df["p_bid_size"], df["p_ask_size"] = _split_size(df.pop("p_size"))

    df["date"] = pd.to_datetime(df["date"].str.strip())
    df["expiry"] = pd.to_datetime(df["expiry"].str.strip())
    df["T"] = (df["expire_unix"] - df["quote_unix"]) / SECONDS_PER_YEAR
    df["dte"] = (df["expiry"] - df["date"]).dt.days
    exp = df["expiry"].dt
    df["is_third_friday"] = (exp.day_of_week == 4) & exp.day.between(15, 21)
    return df.sort_values(["date", "expiry", "strike"], ignore_index=True)


def two_sided(bid, ask):
    """Mask of usable two-sided quotes."""
    bid, ask = np.asarray(bid, dtype=float), np.asarray(ask, dtype=float)
    return np.isfinite(bid) & np.isfinite(ask) & (bid > 0.0) & (ask > bid)
