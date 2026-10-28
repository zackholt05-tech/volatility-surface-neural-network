"""Data-quality diagnostics for the processed SPX tables: figures and summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from volsurf.data import store  # noqa: E402

COLUMNS = ["date", "dte", "T", "k", "z", "iv_mid", "iv_bid", "iv_ask", "iv_vendor",
           "arb_vertical", "arb_butterfly", "expiry"]  # fmt: skip


def rates_and_carry(fwd: pd.DataFrame, out: Path) -> pd.DataFrame:
    sel = fwd[fwd["valid"] & fwd["T"].between(0.25, 1.0)].copy()
    sel["carry"] = np.log(sel["F"] / sel["spot"]) / sel["T"]
    daily = sel.groupby("date")[["r", "carry"]].median()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(daily.index, 100 * daily["r"], lw=1, label="implied rate r (3-12m)")
    ax.plot(daily.index, 100 * daily["carry"], lw=1, label="carry r - q (3-12m)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(ylabel="% per year", title="Put-call-parity rates and carry")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "rates_carry.png", dpi=120)
    plt.close(fig)
    return daily.resample("QE").median()


def funnel(summary: pd.DataFrame) -> pd.DataFrame:
    steps = ["contracts", "dte", "forward", "two_sided", "spread", "k_range", "iv", "z_range",
             "slice_size"]  # fmt: skip
    by_year = summary.assign(year=summary["month"].str[:4]).groupby("year")[steps].sum()
    total = by_year.sum().rename("all")
    table = pd.concat([by_year, total.to_frame().T])
    return table.div(table["contracts"], axis=0).round(3).assign(kept=table["slice_size"])


def vendor_gap(q: pd.DataFrame) -> pd.DataFrame:
    q = q.assign(gap=q["iv_mid"] - q["iv_vendor"], year=q["date"].dt.year)
    q["T_bucket"] = pd.cut(q["dte"], [6, 30, 90, 365, 731])
    atm = q[q["z"].abs() < 0.5]
    return atm.pivot_table("gap", "year", "T_bucket", aggfunc="median", observed=True).round(4)


def surfaces(q: pd.DataFrame, dates: list[str], out: Path) -> None:
    fig, axes = plt.subplots(1, len(dates), figsize=(6 * len(dates), 4.2), squeeze=False)
    for ax, date in zip(axes[0], dates, strict=True):
        day = q[q["date"] == pd.Timestamp(date)]
        expiries = sorted(day["expiry"].unique())
        targets = [10, 30, 90, 180, 365, 700]
        picks = {
            min(expiries, key=lambda e, t=t: abs((e - pd.Timestamp(date)).days - t))
            for t in targets
        }
        for e in sorted(picks):
            s = day[day["expiry"] == e]
            err = [s["iv_mid"] - s["iv_bid"], s["iv_ask"] - s["iv_mid"]]
            ax.errorbar(s["k"], s["iv_mid"], yerr=err, fmt=".", ms=2, lw=0.5,
                        label=f"{s['dte'].iloc[0]}d")  # fmt: skip
        ax.set(xlabel="k = log(K/F)", ylabel="implied vol", title=f"SPX {date}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "surfaces.png", dpi=120)
    plt.close(fig)


def coverage(q: pd.DataFrame, out: Path) -> None:
    daily = q.groupby("date").agg(
        quotes=("k", "size"),
        slices=("expiry", "nunique"),
        arb=("arb_butterfly", "mean"),
    )
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    ax[0].plot(daily.index, daily["quotes"], lw=0.8)
    ax[0].set(title="quotes per day")
    ax2 = ax[0].twinx()
    ax2.plot(daily.index, daily["slices"], lw=0.8, color="C1")
    ax2.set_ylabel("expiries", color="C1")
    ax[1].plot(daily.index, 100 * daily["arb"], lw=0.8)
    ax[1].set(title="quotes in a tradable butterfly violation (%)")
    fig.tight_layout()
    fig.savefig(out / "coverage.png", dpi=120)
    plt.close(fig)


def third_friday(fwd: pd.DataFrame) -> pd.DataFrame:
    near = fwd[fwd["dte"].between(7, 45)]
    return near.groupby("is_third_friday").agg(
        slices=("dispersion", "size"),
        median_dispersion=("dispersion", "median"),
        invalid=("valid", lambda v: 1 - v.mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/processed/spx/diagnostics")
    parser.add_argument("--dates", nargs="*", default=["2019-06-14", "2020-03-16", "2023-06-15"])
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fwd = store.load_table("spx", "forwards")
    q = store.load_quotes("spx", columns=COLUMNS)
    summary = pd.read_csv(
        store.table_dir("spx", "summary").with_suffix(".csv"), dtype={"month": str}
    )

    pd.set_option("display.width", 200)
    print("Quarterly median rate and carry (3-12m):")
    print((100 * rates_and_carry(fwd, out)).round(2).to_string())
    print("\nFilter funnel (share of contracts surviving each step):")
    print(funnel(summary).to_string())
    print("\nMedian iv_mid - vendor IV, |z| < 0.5:")
    print(vendor_gap(q).to_string())
    print("\nForward quality, 7-45 DTE, third-Friday (SPX/SPXW overlap) vs other expiries:")
    print(third_friday(fwd).to_string())
    surfaces(q, args.dates, out)
    coverage(q, out)
    print(f"\nfigures -> {out}")


if __name__ == "__main__":
    main()
