"""Saved panels as the dashboard reads them: per-quote tables, fitted surfaces, residuals, flags.

A panel is a run of consecutive trading days fitted by scripts/fit_panel.py: one per-quote table a
year (market quote and every detector's model IV) and the cached `mix4` fits. The panels are the
refits with the previous-day start (configs/phase7_warm.yaml), whose least-squares surfaces avoid
the default start's poor local minima. A panel's years are
loaded together, so a residual can be followed into the next year; panels are never joined to
each other. Flags use the rule and frozen margins of the mispricing reports.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from volsurf.config import load_yaml
from volsurf.eval.dayset import FitData
from volsurf.eval.mispricing import SurfaceState, flag, residuals
from volsurf.models.base import SurfaceModel
from volsurf.models.conditional import ConditionalSurface
from volsurf.models.registry import build_model

ROOT = Path(__file__).resolve().parents[3]
KEYS = ["expiry", "strike", "option_type"]
BASELINES = "configs/phase4_final.yaml"  # frozen SVI and spline specs
DETECTORS = {
    "iv_rob": "robust mix4 (Phase 7)",
    "iv_seg": "pooled robust mix4 (Phase 7b)",
    "iv_ls": "least-squares mix4",
    "iv_cond": "conditional, one training year",
    "iv_cond_multi": "conditional, nine training years",
}


@dataclass(frozen=True)
class PanelSpec:
    name: str
    fits: str  # directory written by scripts/fit_panel.py
    years: tuple[int, ...]
    config: str  # frozen config with the panel's flag margin and detectors
    source: str  # the panel's key in that config
    checkpoint: str  # conditional model; {quarter} is the date's quarter (walk-forward refits)
    test: tuple[int, ...]  # the panel's test years, which nothing was tuned on

    @property
    def synthetic(self) -> bool:
        return self.source.startswith("synthetic")


PANELS = [
    PanelSpec(
        "SPX 2022–2023", "results/phase7-warm-fits/spx", (2022, 2023), "configs/phase7_warm.yaml",
        "spx",
        "artifacts/conditional/walkforward/spx_full_e10_{quarter}.pt", (2023,),
    ),
    PanelSpec(
        "SPX 2016–2017", "results/replication-warm-fits/spx", (2016, 2017),
        "configs/replication_warm.yaml", "spx",
        "artifacts/conditional/walkforward/spx_rep_e40_{quarter}.pt", (2016, 2017),
    ),
    PanelSpec(
        "Heston 2021–2022 (synthetic)", "results/phase7-warm-fits/synthetic_heston", (2021, 2022),
        "configs/phase7_warm.yaml", "synthetic:heston", "artifacts/conditional/heston_multi_d8.pt",
        (2022,),
    ),
    PanelSpec(
        "Heston fresh 2023 (synthetic)", "results/phase7-warm-fits/synthetic_heston_fresh", (2023,),
        "configs/phase7_warm.yaml", "synthetic:heston_fresh",
        "artifacts/conditional/heston_multi_d8.pt", (2023,),
    ),
]  # fmt: skip


def missing_files(spec: PanelSpec) -> list[str]:
    paths = [f"{spec.fits}/{y}.parquet" for y in spec.years] + [f"{spec.fits}/fits.pkl"]
    return [p for p in paths if not (ROOT / p).exists()]


def available_panels() -> list[PanelSpec]:
    return [s for s in PANELS if not missing_files(s)]


def quarter(date) -> str:
    date = pd.Timestamp(date)
    return f"{date.year}Q{(date.month - 1) // 3 + 1}"


@dataclass
class Panel:
    spec: PanelSpec
    table: pd.DataFrame  # sorted by date, and by (T, k) within a day
    dates: np.ndarray  # trading days, ascending
    starts: np.ndarray  # day i is rows starts[i]:starts[i + 1] of the table
    fits: dict  # date -> [least-squares state, robust passes...] as (knot_T, knots, weights)
    margin: float  # frozen flag margin, in half-spreads outside the bid-ask
    default_detector: str  # the first detector of the panel's frozen config

    @property
    def detectors(self) -> list[str]:
        return [c for c in DETECTORS if c in self.table]

    def index(self, date) -> int:
        i = int(np.searchsorted(self.dates, np.datetime64(pd.Timestamp(date))))
        if i == len(self.dates) or self.dates[i] != np.datetime64(pd.Timestamp(date)):
            raise KeyError(f"{date} is not a day of {self.spec.name}")
        return i

    def rows(self, i: int) -> slice:
        return slice(int(self.starts[i]), int(self.starts[i + 1]))

    def day(self, i: int) -> pd.DataFrame:
        return self.table.iloc[self.rows(i)]

    def state(self, i: int, robust: bool = True) -> SurfaceState | None:
        fit = self.fits.get(pd.Timestamp(self.dates[i]))
        if not fit:
            return None
        return SurfaceState(*(fit[-1] if robust else fit[0]))

    def checkpoint(self, i: int) -> Path:
        return ROOT / self.spec.checkpoint.format(quarter=quarter(self.dates[i]))


def _panel_settings(spec: PanelSpec) -> tuple[float, str]:
    panel = load_yaml(ROOT / spec.config)["panels"][spec.source]
    margin = panel.get("margin", panel.get("backtest", {}).get("margin"))
    return float(margin), panel["detectors"][0]


def make_panel(spec: PanelSpec, table: pd.DataFrame, fits: dict) -> Panel:
    table = table.sort_values("date", kind="stable").reset_index(drop=True)
    dates, starts = np.unique(table["date"].to_numpy(), return_index=True)
    margin, detector = _panel_settings(spec)
    return Panel(spec, table, dates, np.r_[starts, len(table)], fits, margin, detector)


def load_panel(spec: PanelSpec) -> Panel:
    if missing := missing_files(spec):
        raise FileNotFoundError(f"{spec.name}: missing {', '.join(missing)}")
    root = ROOT / spec.fits
    table = pd.concat([pd.read_parquet(root / f"{y}.parquet") for y in spec.years])
    return make_panel(spec, table, pickle.loads((root / "fits.pkl").read_bytes()))


def panel_residuals(panel: Panel, detector: str) -> np.ndarray:
    """Residuals of every quote in price half-spreads (positive: market rich)."""
    return residuals(panel.table, detector)["r"].to_numpy()


def flags_per_day(panel: Panel, r: np.ndarray, margin: float) -> np.ndarray:
    return np.add.reduceat((flag(r, margin) != 0).astype(int), panel.starts[:-1])


def follow(panel: Panel, r: np.ndarray, i: int, horizons=(1, 3, 5)) -> pd.DataFrame:
    """Residuals of day i's contracts h trading days later in the panel (NaN once unquoted)."""
    day = panel.table.iloc[panel.rows(i)][KEYS]
    out = pd.DataFrame(index=day.index)
    for h in horizons:
        col = f"r_{h}d"
        if i + h >= len(panel.dates):
            out[col] = np.nan
            continue
        later = panel.rows(i + h)
        nxt = panel.table.iloc[later][KEYS].assign(**{col: r[later]})
        out[col] = day.merge(nxt, on=KEYS, how="left")[col].to_numpy()
    return out


def fit_data(day: pd.DataFrame) -> FitData:
    """The quotes a model is fitted on: every quote but the legs of model-free arbitrage, with the
    floored half-spreads, as in the mispricing fits."""
    q = day[~day["arb_leg"].to_numpy()]
    T = q["T"].to_numpy()
    slice_T, slice_id = np.unique(T, return_inverse=True)
    return FitData(
        q["k"].to_numpy(), T, q["iv_mid"].to_numpy(), q["s"].to_numpy(), slice_id, slice_T
    )


def slice_bounds(day: pd.DataFrame) -> pd.DataFrame:
    """Quoted log-moneyness range of each maturity, the data domain of the arbitrage checks."""
    b = day.groupby("T")["k"].agg(["min", "max"]).reset_index()
    return b[["T", "min", "max"]]


LIVE_MODELS = {
    "mix4": "mix4, robust",
    "mix4_ls": "mix4, least squares",
    "cond": "conditional",
    "cond_refined": "conditional, refined",
    "svi": "SVI",
    "spline": "spline",
}  # saved mix4 fits; the conditional model (amortized or refined), SVI and the spline fit live


def surface_model(panel: Panel, i: int, kind: str) -> SurfaceModel:
    """A fitted surface for day i: a saved mix4 fit, or SVI, the spline or the conditional model
    fitted now on the same quotes."""
    if kind in ("mix4", "mix4_ls"):
        state = panel.state(i, robust=kind == "mix4")
        if state is None:
            raise KeyError(f"no saved fit for {pd.Timestamp(panel.dates[i]).date()}")
        return state.model()
    data = fit_data(panel.day(i))
    if kind in ("cond", "cond_refined"):
        path = panel.checkpoint(i)
        if not path.exists():
            raise FileNotFoundError(f"missing {path.relative_to(ROOT)}")
        return ConditionalSurface(str(path), refine_iters=100 if kind == "cond_refined" else 0).fit(
            data
        )
    specs = {m["name"]: m for m in load_yaml(ROOT / BASELINES)["models"]}
    return build_model(specs[kind], seed=0).fit(data)
