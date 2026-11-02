"""Static-arbitrage maps: where a fitted surface violates the butterfly and calendar conditions,
on the evaluation's grids over the quoted domain or the extended one (the wings)."""

from __future__ import annotations

import numpy as np
import streamlit as st
from common import MODEL_HELP, current, get_maps

from volsurf.eval.metrics import ArbEvalConfig
from volsurf.viz.heatmaps import butterfly_figure, calendar_figure
from volsurf.viz.panels import LIVE_MODELS

DOMAINS = {
    "extended": f"quoted range widened by {ArbEvalConfig().widen:.0%} per side (wings)",
    "quoted": "quoted range",
}


def render() -> None:
    sel = current()
    st.header(f"Static arbitrage, {sel.name}, {sel.date.date()}")
    left, right = st.columns([3, 2])
    kind = left.segmented_control(
        "Surface", list(LIVE_MODELS), default="svi", format_func=LIVE_MODELS.get,
        key="arb_model", help=MODEL_HELP,
    ) or "svi"  # fmt: skip
    where = right.segmented_control(
        "Domain", list(DOMAINS), default="extended", format_func=DOMAINS.get, key="arb_domain"
    ) or "extended"  # fmt: skip
    widen = ArbEvalConfig().widen if where == "extended" else 0.0
    try:
        maps = get_maps(sel.name, sel.i, kind, widen)
    except (KeyError, FileNotFoundError) as err:
        st.warning(str(err))
        return

    c = st.columns(4)
    c[0].metric("butterfly violations", f"{100 * maps.butterfly_frac:.2f}%")
    c[1].metric("calendar violations", f"{100 * maps.calendar_frac:.2f}%")
    c[2].metric("worst calendar drop", f"{maps.calendar_vp.max():.2f} vp")
    c[3].metric("min Durrleman g", f"{maps.g.min():.3g}")
    cols = st.columns(2)
    cols[0].plotly_chart(butterfly_figure(maps), width="stretch")
    cols[1].plotly_chart(calendar_figure(maps), width="stretch")
    legs = int(sel.day["arb_leg"].sum())
    st.caption(
        "Grey is checked and arbitrage-free; red marks violations, darker for larger ones. The "
        "dotted lines bound the quoted strikes. The calendar map compares each grid maturity with "
        "the next one. mix4 and the conditional model are arbitrage-free by construction, so "
        "their maps stay grey; SVI and the spline are fitted slice by slice and are not. "
        f"Model-free check of the quotes themselves: {legs} of {len(sel.day):,} quotes "
        f"({100 * np.mean(sel.day['arb_leg']):.2f}%) are legs of a tradable vertical or "
        "butterfly arbitrage, and are left out of every fit."
    )


render()
