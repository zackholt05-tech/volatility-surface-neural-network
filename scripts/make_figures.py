"""Figures, tables and quoted numbers of the technical report, from saved results.

Reads the outputs that the Makefile targets write under results/ (and the per-day table of
data/processed/spx/days) and writes, under report/:

- figures/*.pdf: vector figures;
- tables/*.tex: booktabs tabulars, each included by a table environment of the report;
- tables/numbers.tex: macros for the figures the text quotes, so that prose and tables come from
  the same files.

Nothing is refitted: every number is read from a results file or recomputed from the saved
per-quote tables. Tables hold summary statistics only, never quotes.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R = Path("results")
WIDTH = 6.5  # inches: the report's text width

# Okabe-Ito, one colour per model family throughout
BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, RED, PURPLE = (
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
)
GREY = "#888888"
COLOR = {
    "mix4": BLUE,
    "pnn": ORANGE,
    "cond": RED,
    "svi": PURPLE,
    "spline": GREEN,
    "ssvi": GREY,
    "essvi": SKY,
    "nmix4": SKY,
    "mlp": BLACK,  # yellow is too light for an open marker on white
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)
METADATA = {"CreationDate": None}  # deterministic PDFs


# --- formatting ------------------------------------------------------------------------------


def num(x: float, d: int = 2, signed: bool = False) -> str:
    """A number in math mode, with a true minus sign."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{x:+.{d}f}" if signed else f"{x:.{d}f}"
    return f"${s}$"


def pct(x: float, d: int = 0) -> str:
    return "--" if np.isnan(x) else f"{100 * x:.{d}f}\\%"


def share(x: float) -> str:
    """A violation share: exactly zero is the guarantee, so it is printed as 0."""
    return "0" if x == 0 else f"{100 * x:.2f}\\%"


def money(x: float) -> str:
    """Dollars: in millions above $1m, in thousands above $1k."""
    sign = "$-$" if x < 0 else "$+$"
    a = abs(x)
    if a >= 1e6:
        return f"{sign}\\${a / 1e6:.2f}m"
    if a >= 1e4:
        return f"{sign}\\${a / 1e3:.0f}k"
    return f"{sign}\\${a / 1e3:.1f}k" if a >= 1e3 else f"{sign}\\${a:.0f}"


def cost(x: float) -> str:
    return money(x)[3:]  # costs are positive: drop the sign


def pval(p: float) -> str:
    if p < 0.001:
        return "$<0.001$"
    return f"${p:.3f}$" if p < 0.01 else f"${p:.2f}$"


def bold(cell: str) -> str:
    """Bold a cell; numbers in math mode need \\mathbf, since \\textbf does not reach them."""
    if "$" in cell:
        return re.sub(r"\$([^$]+)\$", r"$\\mathbf{\1}$", cell)
    return f"\\textbf{{{cell}}}"


def bold_min(cells: list[str], values, best=min) -> list[str]:
    """Bold the cell of the best value."""
    v = np.asarray(values, dtype=float)
    i = int(np.nanargmin(v) if best is min else np.nanargmax(v))
    return [bold(c) if j == i else c for j, c in enumerate(cells)]


def write_tabular(
    path: Path,
    colspec: str,
    header: list[str],
    rows: list[list[str]],
    groups: dict[int, str] | None = None,
    top: str | None = None,
) -> None:
    """A booktabs tabular; ``groups`` maps a row index to a block heading above that row, and
    ``top`` is a header row (with its rules) above the column names."""
    ncol = len(header)
    lines = [
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        *([top] if top else []),
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for i, row in enumerate(rows):
        if groups and i in groups:
            if i:
                lines.append("\\midrule")
            lines.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{{groups[i]}}}}} \\\\")
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    write_text(path, "\n".join(lines))


def write_text(path: Path, text: str) -> None:
    try:
        path.write_text(text)
    except OSError as error:
        raise SystemExit(f"could not write {path}: {error}") from error


def save(fig, path: Path) -> None:
    """Write a figure as a deterministic PDF and close it."""
    try:
        fig.savefig(path, metadata=METADATA)
    except OSError as error:
        raise SystemExit(f"could not write {path}: {error}") from error
    finally:
        plt.close(fig)


def summary(experiment: str, source: str = "spx") -> pd.DataFrame:
    return pd.read_csv(R / experiment / source / "summary.csv", index_col="model")


def plain_log(axis, ticks) -> None:
    """A log axis labelled with plain numbers at ``ticks``."""
    axis.set_major_locator(mticker.FixedLocator(ticks))
    axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(mticker.NullFormatter())


# --- data ------------------------------------------------------------------------------------


def fig_data(out: Path) -> dict:
    days = pd.concat(
        pd.read_parquet(p) for p in sorted(glob.glob("data/processed/spx/days/*.parquet"))
    )
    days["date"] = pd.to_datetime(days["date"])
    monthly = days.set_index("date").resample("MS").median(numeric_only=True)
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 2.6), sharex=True)
    windows = [
        ("2016-01-01", "2018-01-01", GREEN, "replication (2016-2017)"),
        ("2022-01-01", "2023-01-01", ORANGE, "validation (2022)"),
        ("2023-01-01", "2024-01-01", RED, "test (2023)"),
    ]
    for ax in axes:
        for a, b, c, label in windows:
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=c, alpha=0.15, lw=0, label=label)
    axes[0].plot(monthly.index, monthly["quotes"], color=BLACK)
    axes[0].set_yscale("log")
    plain_log(axes[0].yaxis, [500, 1000, 2000, 5000])
    axes[0].set_ylabel("quotes / day")
    axes[1].plot(monthly.index, 100 * monthly["median_half_spread_iv"], color=BLACK)
    axes[1].set_ylabel("half-spread (vp)")
    axes[1].set_ylim(0, 1.25)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.93))
    axes[1].xaxis.set_major_locator(mdates.YearLocator(1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.align_ylabels(axes)
    save(fig, out / "data.pdf")
    by_year = days.groupby(days["date"].dt.year)
    return {
        "spxDays": f"{len(days):,}".replace(",", "{,}"),
        "spxFirstYearQuotes": f"{by_year['quotes'].median().loc[2010]:.0f}",
        "spxValQuotes": f"{by_year['quotes'].median().loc[2022]:,.0f}".replace(",", "{,}"),
    }


# --- surface models --------------------------------------------------------------------------

NAMES_2022 = [
    ("ssvi", "SSVI"),
    ("svi", "SVI (per slice)"),
    ("spline", "smoothing spline"),
    ("mlp", "MLP, no penalty"),
    ("pnn", "penalized network (\\texttt{pnn})"),
    ("essvi", "eSSVI"),
    ("mix4", "eSSVI mixture (\\texttt{mix4})"),
    ("nmix4", "network-generated mixture"),
    ("cond", "conditional, amortized"),
    ("cond_refined", "conditional, refined"),
]
FIT_COLS = [
    "fit/rmse_vp",
    "fit/within_spread",
    "interp/rmse_vp",
    "wing/rmse_vp",
    "arb_ext/calendar_frac",
    "arb_ext/butterfly_frac",
    "fit_seconds",
]
FIT_HEADER = ["model", "fit", "in spread", "strikes", "wings", "calendar", "butterfly", "s/day"]
FIT_TOP = (
    "& \\multicolumn{4}{c}{quotes (RMSE in vp)} & \\multicolumn{2}{c}{violations} & \\\\\n"
    "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}"
)


def fit_rows(table: pd.DataFrame, names: list[tuple[str, str]]) -> list[list[str]]:
    t = table.loc[[m for m, _ in names]]
    cols = {
        "fit": bold_min([num(x, 3) for x in t["fit/rmse_vp"]], t["fit/rmse_vp"]),
        "in": bold_min([pct(x) for x in t["fit/within_spread"]], t["fit/within_spread"], max),
        "interp": bold_min([num(x, 3) for x in t["interp/rmse_vp"]], t["interp/rmse_vp"]),
        "wing": bold_min([num(x, 2) for x in t["wing/rmse_vp"]], t["wing/rmse_vp"]),
        "time": [
            "$<0.01$" if x < 0.01 else num(x, 2) if x < 10 else num(x, 0) for x in t["fit_seconds"]
        ],
    }
    return [
        [
            name,
            cols["fit"][i],
            cols["in"][i],
            cols["interp"][i],
            cols["wing"][i],
            share(r["arb_ext/calendar_frac"]),
            share(r["arb_ext/butterfly_frac"]),
            cols["time"][i],
        ]
        for i, ((_, name), (_, r)) in enumerate(zip(names, t.iterrows(), strict=True))
    ]


def tab_models_2022(out: Path) -> dict:
    t = pd.concat(
        [summary("phase5-comparison"), summary("phase6-comparison").loc[["cond", "cond_refined"]]]
    )
    write_tabular(
        out / "models_2022.tex", "lrrrrrrr", FIT_HEADER, fit_rows(t, NAMES_2022), top=FIT_TOP
    )
    m, p = t.loc["mix4"], t.loc["pnn"]
    return {
        "mixFitVal": num(m["fit/rmse_vp"], 3)[1:-1],
        "mixInVal": pct(m["fit/within_spread"]),
        "mixWingVal": num(m["wing/rmse_vp"])[1:-1],
        "pnnWingVal": num(p["wing/rmse_vp"])[1:-1],
        "pnnCalVal": share(p["arb_ext/calendar_frac"]),
        "mixTimeVal": f"{m['fit_seconds']:.0f}",
        "pnnTimeVal": f"{p['fit_seconds']:.0f}",
        "condTimeVal": f"{t.loc['cond', 'fit_seconds']:.2f}",
        "sviCalVal": share(t.loc["svi", "arb_ext/calendar_frac"]),
    }


def tab_out_of_sample(out: Path) -> dict:
    test = pd.concat(
        [
            summary("walkforward-test").loc[["mix4", "cond", "cond_wf"]],
            summary("walkforward-full-test").loc[
                ["cond_full", "cond_full_wf", "cond_full_wf_refined"]
            ],
        ]
    )
    names_test = [
        ("mix4", "eSSVI mixture (\\texttt{mix4})"),
        ("cond", "conditional, 2018--2021, frozen"),
        ("cond_wf", "conditional, 2018--2021, walk-forward"),
        ("cond_full", "conditional, 2010--2021, frozen"),
        ("cond_full_wf", "conditional, 2010--2021, walk-forward"),
        ("cond_full_wf_refined", "\\quad the same, refined"),
    ]
    rep = summary("replication-test")
    names_rep = [
        ("ssvi", "SSVI"),
        ("svi", "SVI (per slice)"),
        ("spline", "smoothing spline"),
        ("pnn", "penalized network (\\texttt{pnn})"),
        ("mix4", "eSSVI mixture (\\texttt{mix4})"),
        ("cond", "conditional, 2010--2015, frozen"),
        ("cond_wf", "conditional, 2010--2015, walk-forward"),
        ("cond_wf_refined", "\\quad the same, refined"),
    ]
    rows = fit_rows(test, names_test) + fit_rows(rep, names_rep)
    write_tabular(
        out / "out_of_sample.tex",
        "lrrrrrrr",
        FIT_HEADER,
        rows,
        {
            0: f"SPX 2023, test year ({test['days'].iloc[0]:.0f} days)",
            len(names_test): f"SPX 2016--2017, replication ({rep['days'].iloc[0]:.0f} days)",
        },
        top=FIT_TOP,
    )
    return {
        "mixFitTest": num(test.loc["mix4", "fit/rmse_vp"], 3)[1:-1],
        "mixWingTest": num(test.loc["mix4", "wing/rmse_vp"])[1:-1],
        "condWingTest": num(test.loc["cond_full_wf", "wing/rmse_vp"])[1:-1],
        "condFitTest": num(test.loc["cond_full_wf", "fit/rmse_vp"])[1:-1],
        "mixFitRep": num(rep.loc["mix4", "fit/rmse_vp"], 3)[1:-1],
        "mixInRep": pct(rep.loc["mix4", "fit/within_spread"]),
        "mixWingRep": num(rep.loc["mix4", "wing/rmse_vp"])[1:-1],
        "pnnWingRep": num(rep.loc["pnn", "wing/rmse_vp"])[1:-1],
        "condWingRep": num(rep.loc["cond_wf", "wing/rmse_vp"])[1:-1],
    }


def fig_tradeoff(out: Path) -> None:
    t = pd.concat(
        [summary("phase5-comparison"), summary("phase6-comparison").loc[["cond", "cond_refined"]]]
    )
    label = {
        "ssvi": "SSVI",
        "svi": "SVI",
        "spline": "spline",
        "mlp": "MLP",
        "pnn": "pnn",
        "essvi": "eSSVI",
        "mix4": "mix4",
        "nmix4": "nmix4",
        "nmix1": "nmix1",
        "cond": "cond",
        "cond_refined": "cond, refined",
    }
    fig, ax = plt.subplots(figsize=(WIDTH * 0.6, 2.35))
    for m, r in t.iterrows():
        viol = r["arb_ext/calendar_frac"] + r["arb_ext/butterfly_frac"]
        c = COLOR.get(m.split("_")[0], GREY)
        free = viol == 0
        ax.plot(
            r["fit/rmse_vp"],
            r["wing/rmse_vp"],
            "o" if free else "s",
            ms=5,
            color=c,
            mfc=c if free else "white",
            mew=1.1,
        )
        text = label[m] + ("" if free else f" ({100 * viol:.1f}%)")
        ax.annotate(
            text,
            (r["fit/rmse_vp"], r["wing/rmse_vp"]),
            xytext=(4, 2),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log(ax.xaxis, [0.1, 0.2, 0.5, 1.0])
    plain_log(ax.yaxis, [1, 2, 3, 5])
    ax.set_xlim(right=1.6)
    ax.set_xlabel("fit RMSE (vp)")
    ax.set_ylabel("held-out wing RMSE (vp)")
    ax.plot([], [], "o", color=BLACK, label="arbitrage-free")
    ax.plot([], [], "s", color=BLACK, mfc="white", label="violations (share of grid)")
    ax.legend(loc="lower right")
    save(fig, out / "tradeoff.pdf")


def fig_regime(out: Path) -> dict:
    frozen = pd.read_csv(R / "phase6-shift/spx/quarterly.csv")
    wf = pd.concat(
        [
            pd.read_csv(R / "walkforward-shift/spx/quarterly.csv"),
            pd.read_csv(R / "walkforward-shift/spx-test/quarterly.csv"),
        ]
    )
    quarters = list(dict.fromkeys(list(frozen["quarter"]) + list(wf["quarter"])))
    x = {q: i for i, q in enumerate(quarters)}
    allq = pd.concat([frozen, wf]).drop_duplicates("quarter").set_index("quarter").loc[quarters]
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(WIDTH, 2.55), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    ax.plot(
        [x[q] for q in frozen["quarter"]],
        frozen["amortized_vp"],
        "o-",
        color=RED,
        label="conditional, trained 2018-2021Q3, frozen",
    )
    ax.plot(
        [x[q] for q in wf["quarter"]],
        wf["amortized_vp"],
        "o--",
        color=RED,
        mfc="white",
        label="conditional, refitted before each quarter",
    )
    ax.plot(
        [x[q] for q in allq.index],
        allq["mix4_free_vp"],
        "o-",
        color=BLUE,
        label="mix4, fitted per day",
    )
    ax.set_ylabel("fit RMSE (vp)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    bx.bar([x[q] for q in allq.index], allq["atm_skew"], color=GREY, width=0.6)
    bx.set_ylabel("ATM skew")
    bx.set_xticks(range(len(quarters)), quarters, rotation=45)
    for a in (ax, bx):
        for i in range(1, len(quarters)):  # breaks where years were not sampled
            if int(quarters[i][:4]) - int(quarters[i - 1][:4]) > 1:
                a.axvline(i - 0.5, color=GREY, lw=0.5, ls=":")
    fig.align_ylabels((ax, bx))
    save(fig, out / "regime.pdf")
    f = frozen.set_index("quarter")
    in_sample = f.loc[f.index.str[:4] < "2022", "amortized_vp"]  # the training years sampled
    return {
        "condQIn": f"{in_sample.min():.2f}",
        "condQInMax": f"{in_sample.max():.2f}",
        "condQOne": f"{f.loc['2022Q1', 'amortized_vp']:.2f}",
        "condQFour": f"{f.loc['2022Q4', 'amortized_vp']:.2f}",
        "skewQOne": num(f.loc["2022Q1", "atm_skew"])[1:-1],
        "skewQFour": num(f.loc["2022Q4", "atm_skew"])[1:-1],
    }


def fig_gaps(out: Path) -> dict:
    levels = [
        ("phase6-comparison", 0.15),
        ("phase6-gap-0.5", 0.5),
        ("phase6-gap-0.75", 0.75),
        ("phase6-gap-0.9", 0.9),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.05), sharey=True)
    nums = {}
    for ax, (source, title) in zip(
        axes, [("spx", "SPX 2022"), ("synthetic_heston", "synthetic Heston 2021")], strict=True
    ):
        for m, c, ls, lab in [
            ("mix4", BLUE, "-", "mix4"),
            ("cond_refined", RED, "--", "conditional, refined"),
            ("svi", PURPLE, ":", "SVI"),
        ]:
            ys = [summary(e, source)["interp/rmse_vp"].get(m, np.nan) for e, _ in levels]
            ax.plot([100 * f for _, f in levels], ys, "o", ls=ls, color=c, label=lab)
        slices = summary("phase6-gap-slices", source)["interp/rmse_vp"]
        nums[source] = slices
        ax.set_title(title)
        ax.set_xlabel("interior strikes held out (%)")
        ax.set_yscale("log")
        plain_log(ax.yaxis, [0.1, 0.2, 0.5, 1.0])
        ax.set_xticks([15, 50, 75, 90])
    axes[0].set_ylabel("held-out strike RMSE (vp)")
    axes[0].legend(loc="upper left")
    save(fig, out / "gaps.pdf")
    g90 = summary("phase6-gap-0.9")["interp/rmse_vp"]
    h90 = summary("phase6-gap-0.9", "synthetic_heston")
    return {
        "mixGapNinety": num(g90["mix4"], 3)[1:-1],
        "sviGapNinety": num(g90["svi"])[1:-1],
        "condGapNinety": num(g90["cond_refined"])[1:-1],
        "mixGapSlices": num(nums["spx"]["mix4"])[1:-1],
        "condGapSlices": num(nums["spx"]["cond_refined"])[1:-1],
        "sviGapSlices": num(nums["spx"]["svi"])[1:-1],
        "hestonTruthNinetyCond": num(h90.loc["cond_refined", "interp/truth_rmse_vp"], 3)[1:-1],
        "hestonTruthNinetyMix": num(h90.loc["mix4", "interp/truth_rmse_vp"])[1:-1],
    }


def tab_heston(out: Path) -> dict:
    panels = [
        summary("heston-multi-validation", "synthetic_heston"),
        summary("heston-multi-test", "synthetic_heston"),
        summary("heston-multi-test", "synthetic_heston_fresh"),
    ]
    val_mix4 = summary("phase6-comparison", "synthetic_heston").loc["mix4"]
    panels[0] = pd.concat([panels[0], val_mix4.to_frame().T])
    names = [
        ("mix4", "eSSVI mixture (\\texttt{mix4})"),
        ("cond", "conditional, one year"),
        ("cond_refined", "\\quad refined"),
        ("cond_multi", "conditional, nine years"),
        ("cond_multi_refined", "\\quad refined"),
    ]
    rows = []
    for m, name in names:
        cells = [num(p.loc[m, "interp/truth_rmse_vp"], 3) for p in panels]
        cells += [num(p.loc[m, "wing/truth_rmse_vp"], 3) for p in panels]
        rows.append([name, *cells])
    for j in range(1, 7):  # bold the best of each column
        col = "interp/truth_rmse_vp" if j <= 3 else "wing/truth_rmse_vp"
        vals = [panels[(j - 1) % 3].loc[m, col] for m, _ in names]
        best = int(np.argmin(vals))
        rows[best][j] = bold(rows[best][j])
    header = ["model", "2021 (val.)", "2022 (test)", "fresh 2023", "2021", "2022", "fresh 2023"]
    lines_top = (
        "& \\multicolumn{3}{c}{held-out strikes} & \\multicolumn{3}{c}{held-out wings} \\\\"
        "\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}"
    )
    write_tabular(out / "heston.tex", "lrrrrrr", header, rows, top=lines_top)
    fresh = panels[2]
    return {
        "multiTruthFresh": num(fresh.loc["cond_multi", "interp/truth_rmse_vp"], 3)[1:-1],
        "mixTruthFresh": num(fresh.loc["mix4", "interp/truth_rmse_vp"], 3)[1:-1],
        "multiInFresh": pct(fresh.loc["cond_multi", "fit/within_spread"]),
        "multiTime": f"{1e3 * fresh.loc['cond_multi', 'fit_seconds']:.0f}",
        "mixTimeHeston": f"{fresh.loc['mix4', 'fit_seconds']:.1f}",
    }


def tab_forecast(out: Path) -> dict:
    rows, nums = [], {}
    for source, name in [("spx", "SPX 2022"), ("synthetic_heston", "Heston 2021")]:
        s = pd.read_csv(R / "phase6-forecast" / source / "summary.csv")
        s = s.pivot(index="h", columns="model", values="rmse_vp")
        t = pd.read_csv(R / "phase6-forecast" / source / "tests.csv")
        t = t.pivot(index="h", columns="baseline", values="p_value")
        for h in s.index:
            best = min(["var1", "latent_rw", "mix4_rw"], key=lambda c, h=h: s.loc[h, c])
            cells = [
                bold(num(s.loc[h, c])) if c == best else num(s.loc[h, c])
                for c in ["var1", "latent_rw", "mix4_rw"]
            ]
            rows.append(
                [
                    name if h == s.index[0] else "",
                    f"${h}$",
                    *cells,
                    pval(t.loc[h, "mix4_rw"]),
                    num(s.loc[h, "oracle"]),
                ]
            )
        nums[source] = (s, t)
    header = [
        "panel",
        "$h$ (days)",
        "VAR(1)",
        "latent RW",
        "\\texttt{mix4} RW",
        "$p$",
        "decoded target",
    ]
    write_tabular(out / "forecast.tex", "llrrrrr", header, rows)
    s, t = nums["spx"]
    return {
        "fcVarTwenty": num(s.loc[20, "var1"])[1:-1],
        "fcRwTwenty": num(s.loc[20, "mix4_rw"])[1:-1],
        "fcPTwenty": pval(t.loc[20, "mix4_rw"])[1:-1],
    }


# --- mispricing ------------------------------------------------------------------------------

DETECTORS = [
    ("iv_ls", "least-squares \\texttt{mix4}"),
    ("iv_rob", "robust \\texttt{mix4}"),
    ("iv_seg", "pooled robust \\texttt{mix4}"),
    ("iv_cond", "conditional, one year"),
    ("iv_cond_multi", "conditional, nine years"),
]


def tab_detection(out: Path) -> dict:
    panels = ["synthetic_heston", "synthetic_heston_fresh"]
    det = {p: pd.read_csv(R / "phase7-warm-test" / p / "detection.csv") for p in panels}
    ab = {p: pd.read_csv(R / "phase7-warm-test" / p / "absorption.csv") for p in panels}
    det = {p: d[d["margin"] == 0.5].set_index("detector") for p, d in det.items()}
    ab = {p: a[a["quarter"] == "all"].set_index("detector") for p, a in ab.items()}
    cols = [
        ("precision", lambda p, d: num(det[p].loc[d, "precision"], 3)),
        ("recall", lambda p, d: num(det[p].loc[d, "recall"])),
        ("recall_point", lambda p, d: num(det[p].loc[d, "recall_point"])),
        ("recall_segment", lambda p, d: num(det[p].loc[d, "recall_segment"])),
        ("absorbed_segment", lambda p, d: pct(ab[p].loc[d, "absorbed_segment"])),
    ]
    rows = []
    for d, name in DETECTORS:
        rows.append([name] + [" / ".join(f(p, d) for p in panels) for _, f in cols])
    for j, (c, _) in enumerate(cols[:4], start=1):  # bold the best detector on the fresh panel
        best = det[panels[1]][c].loc[[d for d, _ in DETECTORS]].to_numpy().argmax()
        rows[best][j] = bold(rows[best][j])
    header = [
        "detector",
        "precision",
        "recall",
        "isolated",
        "segments",
        "absorbed",
    ]
    top = "& & \\multicolumn{3}{c}{recall} & \\\\\n\\cmidrule(lr){3-5}"
    write_tabular(out / "detection.tex", "lrrrrr", header, rows, top=top)
    f = det[panels[1]]
    return {
        "multiPrecFresh": num(f.loc["iv_cond_multi", "precision"], 3)[1:-1],
        "multiRecFresh": num(f.loc["iv_cond_multi", "recall"])[1:-1],
        "multiSegFresh": pct(f.loc["iv_cond_multi", "recall_segment"]),
        "robPrecFresh": num(f.loc["iv_rob", "precision"])[1:-1],
        "robRecFresh": num(f.loc["iv_rob", "recall"])[1:-1],
        "robSegFresh": pct(f.loc["iv_rob", "recall_segment"]),
        "segPrecFresh": num(f.loc["iv_seg", "precision"])[1:-1],
        "segRecFresh": num(f.loc["iv_seg", "recall"])[1:-1],
        "segSegFresh": pct(f.loc["iv_seg", "recall_segment"]),
        "robAbsorbFresh": pct(ab[panels[1]].loc["iv_rob", "absorbed_segment"]),
        "segAbsorbFresh": pct(ab[panels[1]].loc["iv_seg", "absorbed_segment"]),
    }


def fig_detection(out: Path) -> None:
    det = pd.read_csv(R / "phase7-warm-test/synthetic_heston_fresh/detection.csv")
    style = {
        "iv_ls": (GREY, "s", ":", "least-squares mix4"),
        "iv_rob": (BLUE, "^", "--", "robust mix4"),
        "iv_seg": (SKY, "v", "-.", "pooled robust mix4"),
        "iv_cond": (ORANGE, "D", "--", "conditional, one year"),
        "iv_cond_multi": (RED, "o", "-", "conditional, nine years"),
    }  # marker and dash as well as colour, so the lines read in grey
    fig, ax = plt.subplots(figsize=(WIDTH * 0.55, 2.25))
    for d, (c, marker, ls, lab) in style.items():
        r = det[(det["detector"] == d) & (det["margin"] <= 2)].sort_values("margin")
        ax.plot(r["recall"], r["precision"], marker=marker, ls=ls, color=c, label=lab, ms=3.5)
        at = r[r["margin"] == 0.5]
        ax.plot(at["recall"], at["precision"], marker, color=c, ms=8, mfc="white", mew=1.3)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower left")
    save(fig, out / "detection.pdf")


def ls_starts(path: Path) -> pd.DataFrame:
    s = pd.read_csv(path, parse_dates=["date"])
    s["ratio"] = s["objective"] / s["cold_objective"]
    return s


STARTS = [
    ("SPX 2022 (validation)", R / "phase7-warm-fits/spx/ls_starts_2022.csv"),
    ("SPX 2023 (test)", R / "phase7-warm-fits/spx/ls_starts_2023.csv"),
    ("SPX 2016", R / "replication-warm-fits/spx/ls_starts_2016.csv"),
    ("SPX 2017", R / "replication-warm-fits/spx/ls_starts_2017.csv"),
    ("Heston 2021", R / "phase7-warm-fits/synthetic_heston/ls_starts_2021.csv"),
    ("Heston 2022", R / "phase7-warm-fits/synthetic_heston/ls_starts_2022.csv"),
    ("Heston, fresh 2023", R / "phase7-warm-fits/synthetic_heston_fresh/ls_starts_2023.csv"),
]


def tab_refit(out: Path) -> dict:
    rows, nums = [], {}
    for name, path in STARTS:
        s = ls_starts(path)
        worst = s.loc[s["ratio"].idxmin()]
        far = (s["ratio"] < 0.9).sum()
        rows.append(
            [
                name,
                f"{(s['start'] == 'warm').sum()} / {len(s)}",
                num(s["ratio"].median()),
                f"{far}",
                f"{(s['ratio'] < 0.5).sum()}",
                "--"
                if far == 0
                else f"{worst['cold_objective']:.2f} $\\to$ "
                f"{worst['objective']:.2f} ({worst['date']:%b %-d})",
            ]
        )
        nums[name] = s
    header = [
        "panel",
        "start won",
        "median",
        "$>10\\%$",
        "$>2\\times$",
        "worst day",
    ]
    top = "& & & \\multicolumn{2}{c}{days lower by} & \\\\\n\\cmidrule(lr){4-5}"
    write_tabular(out / "refit.tex", "lrrrrl", header, rows, top=top)
    s = nums["SPX 2023 (test)"]
    worst = s.loc[s["ratio"].idxmin()]
    return {
        "refitTwoX": f"{(s['ratio'] < 0.5).sum()}",
        "refitDays": f"{len(s)}",
        "refitMedian": num(s["ratio"].median())[1:-1],
        "refitWorstDay": f"{worst['date']:%B %-d, %Y}".replace(" ", "~", 1),
        "refitWorstCold": f"{worst['cold_objective']:.2f}",
        "refitWorstWarm": f"{worst['objective']:.2f}",
    }


def refit_day(starts: Path, cached: Path, refit: Path) -> tuple[pd.Timestamp, pd.DataFrame]:
    """The day of ``starts`` whose refit lowered the objective most, with each quote's cached
    and refitted least-squares IVs and squared cached residual ``e`` in IV half-spreads."""
    day = ls_starts(starts).sort_values("ratio")["date"].iloc[0]
    keys = ["date", "expiry", "strike", "option_type"]
    c = pd.read_parquet(cached, filters=[("date", "==", day)])
    w = pd.read_parquet(refit, columns=[*keys, "iv_ls"], filters=[("date", "==", day)])
    d = c.merge(w, on=keys, suffixes=("_cached", ""))
    d["e"] = ((d["iv_mid"] - d["iv_ls_cached"]) / d["s"]) ** 2
    return day, d


def fig_refit(out: Path) -> None:
    fig = plt.figure(figsize=(WIDTH, 3.9))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.55, wspace=0.28)
    top = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2])]
    for ax, names in zip(top, [STARTS[:2], STARTS[2:4]], strict=True):
        for _, path in names:
            s = ls_starts(path)
            ax.plot(
                s["date"],
                s["ratio"],
                ".",
                ms=2.5,
                color=BLUE if s["date"].dt.year.iloc[0] in (2023, 2017) else GREY,
            )
        ax.axhline(0.5, color=RED, lw=0.6, ls="--")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(mdates.MonthLocator((1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)
    plain_log(top[0].yaxis, [0.25, 0.5, 1.0])
    plain_log(top[1].yaxis, [0.01, 0.1, 0.5, 1.0])
    top[0].set_ylabel("refit / cached objective")
    top[0].set_title("(a) SPX 2022 (grey) and 2023 (blue)")
    top[1].set_title("(b) SPX 2016 (grey) and 2017 (blue)")

    # (c) the SPX day whose objective fell most, its expiry with the largest cached-fit error,
    # in implied vol; (d, e) the worst 2023 day, its two such expiries, in IV half-spreads
    for j, (starts, cached_path, refit_path, n, letter) in enumerate(
        [
            (STARTS[3][1], "replication-fits/spx/2017", "replication-warm-fits/spx/2017", 1, "c"),
            (STARTS[1][1], "phase7b-fits/spx/2023", "phase7-warm-fits/spx/2023", 2, "d"),
        ]
    ):
        day, d = refit_day(starts, R / f"{cached_path}.parquet", R / f"{refit_path}.parquet")
        worst = d[~d["arb_leg"]].groupby("expiry")["e"].mean().nlargest(n).index.sort_values()
        for i, expiry in enumerate(worst):
            ax = fig.add_subplot(grid[1, j + i])
            q = d[d["expiry"] == expiry].sort_values("k")
            title = f"({chr(ord(letter) + i)}) {day:%Y-%m-%d}, {q['dte'].iloc[0]:.0f}-day expiry"
            ax.set_title(title)
            ax.set_xlabel("log-moneyness $k$")
            if j == 0:
                ax.vlines(
                    q["k"],
                    100 * q["iv_bid"],
                    100 * q["iv_ask"],
                    color=BLACK,
                    lw=0.8,
                    label="bid-ask",
                )
                ax.plot(q["k"], 100 * q["iv_ls_cached"], "--", color=RED, label="cached fit")
                ax.plot(q["k"], 100 * q["iv_ls"], "-", color=BLUE, label="refit")
                ax.set_ylabel("implied vol (%)")
                ax.legend(loc="upper right")
            else:
                ax.axhspan(-1, 1, color=GREY, alpha=0.2, lw=0, label="inside the spread")
                ax.plot(
                    q["k"],
                    (q["iv_mid"] - q["iv_ls_cached"]) / q["s"],
                    "x",
                    ms=2.5,
                    mew=0.7,
                    color=RED,
                    label="cached fit",
                )
                ax.plot(
                    q["k"],
                    (q["iv_mid"] - q["iv_ls"]) / q["s"],
                    ".",
                    ms=3,
                    color=BLUE,
                    label="refit",
                )
                if i == 0:
                    ax.set_ylabel("residual (IV half-spreads)")
                else:
                    ax.legend(loc="lower left", markerscale=2)
    save(fig, out / "refit.pdf")


def tab_reversion(out: Path) -> dict:
    cached = pd.read_csv(R / "phase7-test/spx/reversion.csv")
    refit = pd.read_csv(R / "phase7-warm-test/spx/reversion.csv")
    parts = pd.read_csv(R / "phase7-warm-test/spx/reversion_parts.csv")

    def pick(t, det):
        return t[(t["detector"] == det) & (t["margin"] == 4.0)].set_index("h")

    rows = []
    for name, t in [
        ("cached fits, flagged $-$ unflagged", pick(cached, "iv_rob")),
        ("refits, flagged $-$ unflagged", pick(refit, "iv_rob")),
    ]:
        rows.append(
            [name]
            + [
                f"{num(t.loc[h, 'difference'], 2, True)} ({pval(t.loc[h, 'p_value'])})"
                for h in (1, 3, 5)
            ]
        )
    for det, label in [("iv_rob", "robust"), ("iv_seg", "pooled robust")]:
        p = pick(parts, det)
        rows.append(
            [f"{label}: flagged, market part"]
            + [
                f"{num(p.loc[h, 'market_flagged'], 2, True)} ({pval(p.loc[h, 'market_flagged_p'])})"
                for h in (1, 3, 5)
            ]
        )
        rows.append(
            [f"{label}: flagged, model part"]
            + [num(p.loc[h, "model_flagged"], 2, True) for h in (1, 3, 5)]
        )
        rows.append(
            [f"{label}: unflagged, market part"]
            + [num(p.loc[h, "market_unflagged"], 2, True) for h in (1, 3, 5)]
        )
        rows.append(
            [f"{label}: market, flagged $-$ unflagged ($p$)"]
            + [pval(p.loc[h, "market_difference_p"]) for h in (1, 3, 5)]
        )
    write_tabular(
        out / "reversion.tex",
        "lrrr",
        ["", "1 day", "3 days", "5 days"],
        rows,
        {
            0: "reversion slope, price residuals, robust \\texttt{mix4}",
            2: "IV residuals split into market and model moves (refits)",
        },
    )
    r, c = pick(refit, "iv_rob"), pick(cached, "iv_rob")
    seg = pick(refit, "iv_seg")
    p = pick(parts, "iv_rob")
    return {
        "revDiffOne": num(r.loc[1, "difference"], 2, True)[1:-1],
        "revPOne": pval(r.loc[1, "p_value"])[1:-1],
        "revCachedDiffOne": num(c.loc[1, "difference"], 2, True)[1:-1],
        "revCachedPOne": pval(c.loc[1, "p_value"])[1:-1],
        "revDays": f"{r.loc[1, 'days']:.0f}",
        "revCachedDays": f"{c.loc[1, 'days']:.0f}",
        "revSegDays": f"{seg.loc[1, 'days']:.0f}",
        "revMarketPOne": pval(p.loc[1, "market_difference_p"])[1:-1],
    }


def tab_backtest(out: Path) -> dict:
    def runs(experiment, source="spx"):
        return pd.read_csv(R / experiment / source / "backtest.csv").set_index("run")

    warm, cached = runs("phase7-warm-test"), runs("phase7-test")
    rows_spec = [
        (
            "SPX 2023 test, refits",
            warm,
            [("iv_rob, lag 1", "next day"), ("iv_rob, lag 0", "same day (bound)")],
        ),
        (
            "SPX 2023 test, cached fits",
            cached,
            [("iv_rob, lag 1", "next day"), ("iv_rob, lag 0", "same day (bound)")],
        ),
        (
            "SPX 2022 validation, refits",
            runs("phase7-warm-validation"),
            [("iv_rob, lag 1", "next day")],
        ),
        (
            "SPX 2016--2017 replication, refits",
            runs("replication-warm-test"),
            [("iv_rob, lag 1", "next day"), ("iv_rob, lag 0", "same day (bound)")],
        ),
        (
            "Heston 2022 test, conditional (one year)",
            runs("phase7-test", "synthetic_heston"),
            [
                ("iv_cond, lag 1", "next day"),
                ("iv_cond, lag 0", "same day (bound)"),
                ("true labels, lag 0", "true labels, same day"),
            ],
        ),
    ]
    rows, groups = [], {}
    for group, t, items in rows_spec:
        groups[len(rows)] = group
        for run, label in items:
            r = t.loc[run]
            rows.append(
                [
                    label,
                    f"{r['trades']:,.0f}".replace(",", "{,}"),
                    money(r["gross"]),
                    cost(r["cost"]),
                    money(r["net"]),
                    num(r["sharpe"]),
                    money(r["placebo_net"]),
                    f"{money(r['excess_net'])} ({pval(r['p_value'])})",
                ]
            )
    header = [
        "execution",
        "trades",
        "gross",
        "costs",
        "net",
        "Sharpe",
        "placebo",
        "excess ($p$)",
    ]
    write_tabular(out / "backtest.tex", "lrrrrrrr", header, rows, groups)
    n, s = warm.loc["iv_rob, lag 1"], warm.loc["iv_rob, lag 0"]
    c = cached.loc["iv_rob, lag 1"]
    return {
        "btNet": money(n["net"]),
        "btPlacebo": money(n["placebo_net"]),
        "btExcess": money(n["excess_net"]),
        "btP": pval(n["p_value"])[1:-1],
        "btTrades": f"{n['trades']:,.0f}".replace(",", "{,}"),
        "btSold": pct(n["share_sold"]),
        "btSharpe": num(n["sharpe"])[1:-1],
        "btSameExcess": money(s["excess_net"]),
        "btSameP": pval(s["p_value"])[1:-1],
        "btCachedNet": money(c["net"]),
        "btCachedP": pval(c["p_value"])[1:-1],
        "btCachedSold": pct(c["share_sold"]),
        "btRepTrades": f"{runs('replication-warm-test').loc['iv_rob, lag 1', 'trades']:.0f}",
    }


def write_numbers(path: Path, numbers: dict) -> None:
    lines = ["% Generated by scripts/make_figures.py; do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(numbers.items())]
    write_text(path, "\n".join(lines) + "\n")


def _inputs() -> dict[str, list[Path]]:
    """The files read here, by the Makefile target that writes them."""
    spx, heston, fresh = "spx", "synthetic_heston", "synthetic_heston_fresh"
    both = (spx, heston)
    summaries = {
        "constrained": [("phase5-comparison", spx)],
        "cross-day": [("phase6-comparison", s) for s in both],
        "gaps": [(f"phase6-gap-{g}", s) for g in ("0.5", "0.75", "0.9", "slices") for s in both],
        "walk-forward-test": [("walkforward-test", spx)],
        "full-history-test": [("walkforward-full-test", spx)],
        "replication": [("replication-test", spx)],
        "heston-multi": [("heston-multi-validation", heston)],
        "heston-multi-test": [("heston-multi-test", s) for s in (heston, fresh)],
    }
    files = {t: [R / e / s / "summary.csv" for e, s in v] for t, v in summaries.items()}
    files["data"] = [Path("data/processed/spx/days")]
    files["cross-day"].append(R / "phase6-shift/spx/quarterly.csv")
    files["walk-forward"] = [R / "walkforward-shift/spx/quarterly.csv"]
    files["walk-forward-test"].append(R / "walkforward-shift/spx-test/quarterly.csv")
    files["forecast"] = [
        R / "phase6-forecast" / s / f for s in both for f in ("summary.csv", "tests.csv")
    ]
    files["mispricing-test"] = [
        R / "phase7-test/spx/reversion.csv",
        R / "phase7-test/spx/backtest.csv",
        R / "phase7-test/synthetic_heston/backtest.csv",
    ]
    files["phase7b-test"] = [R / "phase7b-fits/spx/2023.parquet"]
    files["warm-validation"] = [
        R / "phase7-warm-validation/spx/backtest.csv",
        R / "phase7-warm-fits/spx/ls_starts_2022.csv",
        R / "phase7-warm-fits/synthetic_heston/ls_starts_2021.csv",
    ]
    warm = R / "phase7-warm-test"
    files["warm-test"] = [
        *(warm / s / f for s in (heston, fresh) for f in ("detection.csv", "absorption.csv")),
        *(warm / spx / f for f in ("reversion.csv", "reversion_parts.csv", "backtest.csv")),
        R / "phase7-warm-fits/spx/ls_starts_2023.csv",
        R / "phase7-warm-fits/spx/2023.parquet",
        R / "phase7-warm-fits/synthetic_heston/ls_starts_2022.csv",
        R / "phase7-warm-fits/synthetic_heston_fresh/ls_starts_2023.csv",
    ]
    files["replication-mispricing"] = [R / "replication-fits/spx/2017.parquet"]
    files["replication-warm"] = [
        R / "replication-warm-test/spx/backtest.csv",
        R / "replication-warm-fits/spx/2017.parquet",
        *(R / f"replication-warm-fits/spx/ls_starts_{y}.csv" for y in (2016, 2017)),
    ]
    return files


def check_inputs() -> None:
    """Stop before writing anything if an input is missing, naming the target that writes it."""
    missing = [
        f"  {path}  (make {target})"
        for target, paths in _inputs().items()
        for path in paths
        if not (any(path.glob("*.parquet")) if path.is_dir() else path.exists())
    ]
    if missing:
        raise SystemExit("missing inputs; run the targets named:\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="report")
    args = parser.parse_args()
    check_inputs()
    figures, tables = Path(args.out) / "figures", Path(args.out) / "tables"
    for d in (figures, tables):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SystemExit(f"could not create {d}: {error}") from error

    numbers = {}
    numbers |= fig_data(figures)
    fig_tradeoff(figures)
    numbers |= fig_regime(figures)
    numbers |= fig_gaps(figures)
    fig_detection(figures)
    fig_refit(figures)
    numbers |= tab_models_2022(tables)
    numbers |= tab_out_of_sample(tables)
    numbers |= tab_heston(tables)
    numbers |= tab_forecast(tables)
    numbers |= tab_detection(tables)
    numbers |= tab_refit(tables)
    numbers |= tab_reversion(tables)
    numbers |= tab_backtest(tables)
    write_numbers(tables / "numbers.tex", numbers)
    print(
        f"wrote {len(list(figures.glob('*.pdf')))} figures, "
        f"{len(list(tables.glob('*.tex')))} tables, {len(numbers)} numbers"
    )


if __name__ == "__main__":
    main()
