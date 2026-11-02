"""Shared state of the dashboard: cached panels, residuals and fits, and the sidebar selection.

Panels and residual arrays are large and read-only, so they are cached once per process as
resources; small per-day results (flag counts, grids) are cached as data. Cache keys are names and
day indices, never frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from volsurf.eval.mispricing import flag
from volsurf.models.base import SurfaceModel
from volsurf.viz.heatmaps import ArbitrageMaps, arbitrage_maps
from volsurf.viz.panels import (
    DETECTORS,
    PANELS,
    Panel,
    available_panels,
    flags_per_day,
    load_panel,
    missing_files,
    panel_residuals,
    slice_bounds,
    surface_model,
)

SOURCES = {
    "SPX 2022–2023": "make phase7b-validation phase7b-test",
    "SPX 2016–2017": "make replication-mispricing",
    "Heston 2021–2022 (synthetic)": "make phase7b-validation phase7b-test",
    "Heston fresh 2023 (synthetic)": "make phase7b-test",
}
MODEL_HELP = (
    "mix4: the saved fits of the mispricing runs (robust, or the least-squares first pass). The "
    "conditional model (the walk-forward checkpoint of the date's quarter; refined adds 100 "
    "L-BFGS steps on its latent), SVI and the spline are fitted on the day's quotes when picked."
)


@st.cache_resource(show_spinner="Loading the panel…")
def get_panel(name: str) -> Panel:
    return load_panel(next(s for s in PANELS if s.name == name))


@st.cache_resource(show_spinner="Scoring every quote of the panel…")
def get_residuals(name: str, detector: str) -> np.ndarray:
    return panel_residuals(get_panel(name), detector)


@st.cache_data(show_spinner=False)
def get_flag_counts(name: str, detector: str, margin: float) -> np.ndarray:
    return flags_per_day(get_panel(name), get_residuals(name, detector), margin)


@st.cache_resource(show_spinner="Fitting the surface…", max_entries=64)
def get_model(name: str, i: int, kind: str) -> SurfaceModel:
    return surface_model(get_panel(name), i, kind)


@st.cache_data(show_spinner="Checking the surface for arbitrage…", max_entries=64)
def get_maps(name: str, i: int, kind: str, widen: float) -> ArbitrageMaps:
    panel = get_panel(name)
    return arbitrage_maps(get_model(name, i, kind), slice_bounds(panel.day(i)), widen=widen)


@dataclass
class Selection:
    name: str
    panel: Panel
    i: int
    detector: str
    margin: float

    @property
    def date(self) -> pd.Timestamp:
        return pd.Timestamp(self.panel.dates[self.i])

    @property
    def day(self) -> pd.DataFrame:
        return self.panel.day(self.i)

    @property
    def r_all(self) -> np.ndarray:
        return get_residuals(self.name, self.detector)

    @property
    def r(self) -> np.ndarray:
        return self.r_all[self.panel.rows(self.i)]

    @property
    def flags(self) -> np.ndarray:
        return flag(self.r, self.margin)


def _default_day(name: str, panel: Panel) -> int:
    """A busy but typical test-year day: of the test days with flags under the panel's frozen
    detector and margin, the first whose count is closest to their 90th percentile (the busiest
    days are outliers)."""
    counts = get_flag_counts(name, panel.default_detector, panel.margin)
    test = np.flatnonzero(np.isin(pd.to_datetime(panel.dates).year, list(panel.spec.test)))
    busy = test[counts[test] > 0]
    if not busy.size:
        return int(test[0])
    return int(busy[np.argmin(np.abs(counts[busy] - np.quantile(counts[busy], 0.9)))])


def sidebar() -> None:
    specs = available_panels()
    if not specs:
        lines = [f"- **{s.name}**: `{SOURCES[s.name]}` (missing {', '.join(missing_files(s))})"
                 for s in PANELS]  # fmt: skip
        st.error("No fitted panels found under `results/`. Each comes from:\n" + "\n".join(lines))
        st.stop()
    side = st.sidebar
    name = side.selectbox("Panel", [s.name for s in specs], key="panel")
    panel = get_panel(name)
    detectors = panel.detectors
    detector = side.selectbox(
        "Detector", detectors, index=detectors.index(panel.default_detector),
        format_func=DETECTORS.get, key=f"detector:{name}",
    )  # fmt: skip
    margin = side.number_input(
        "Flag margin (half-spreads beyond the bid-ask)", min_value=0.0, max_value=10.0,
        value=panel.margin, step=0.5, key=f"margin:{name}",
        help=f"Frozen for this panel: {panel.margin:g} ({panel.spec.config})",
    )  # fmt: skip
    i = side.select_slider(
        "Day", options=range(len(panel.dates)), value=_default_day(name, panel),
        format_func=lambda j: str(pd.Timestamp(panel.dates[j]).date()), key=f"day:{name}",
    )  # fmt: skip
    test = ", ".join(str(y) for y in panel.spec.test)
    side.caption(
        f"{len(panel.dates)} days, {len(panel.table):,} quotes. Test years: {test}. "
        f"It opens on a busy but typical test-year day."
    )
    missing = [s.name for s in PANELS if s not in specs]
    if missing:
        side.caption("Not fitted: " + "; ".join(missing))
    st.session_state["selection"] = Selection(name, panel, int(i), detector, float(margin))


def current() -> Selection:
    return st.session_state["selection"]
