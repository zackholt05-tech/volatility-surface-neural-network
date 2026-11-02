"""Figures of a day's fitted surface and quotes: the implied-volatility surface in 3D, one
expiry's smile against the bid-ask, a residual path and the flags of a panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from volsurf.eval.metrics import domain

COLORS = {
    "iv_rob": "#1f77b4", "iv_seg": "#17becf", "iv_ls": "#7f7f7f", "iv_cond": "#9467bd",
    "iv_cond_multi": "#8c564b", "live": "#ff7f0e", "iv_true": "#2ca02c",
}  # fmt: skip
RICH, CHEAP = "#d62728", "#2ca02c"
DAY_TICKS = [7, 14, 30, 60, 90, 180, 365, 730]


def surface_grid(model, bounds: pd.DataFrame, n_T: int = 40, n_k: int = 61, widen: float = 0.0):
    """IVs on each maturity's quoted k-range (log-spaced and quoted maturities), so the surface is
    not extrapolated where no strike is quoted. Returns (K, T, IV), each of shape (n_T', n_k)."""
    quoted = bounds["T"].to_numpy()
    T = np.union1d(np.geomspace(quoted.min(), quoted.max(), n_T), quoted)
    lo, hi = domain(bounds, T, widen)
    K = lo[:, None] + np.linspace(0.0, 1.0, n_k)[None, :] * (hi - lo)[:, None]
    TT = np.broadcast_to(T[:, None], K.shape)
    return K, TT, np.sqrt(np.maximum(model.w(K, TT), 0.0) / TT)


def surface_figure(K, T, iv, day: pd.DataFrame, flags: np.ndarray | None = None) -> go.Figure:
    fig = go.Figure(
        go.Surface(
            x=K, y=365.0 * T, z=100.0 * iv, colorscale="Viridis", opacity=0.92,
            colorbar={"title": "IV %", "len": 0.6}, name="model",
            hovertemplate="k %{x:.3f}<br>%{y:.0f} days<br>%{z:.2f}%<extra>model</extra>",
        )
    )  # fmt: skip
    fig.add_trace(
        go.Scatter3d(
            x=day["k"], y=day["dte"], z=100.0 * day["iv_mid"], mode="markers",
            marker={"size": 1.6, "color": "#111"}, name="market mid",
            hovertemplate="k %{x:.3f}<br>%{y} days<br>%{z:.2f}%<extra>market</extra>",
        )
    )  # fmt: skip
    if flags is not None and (flags != 0).any():
        f = day[flags != 0]
        fig.add_trace(
            go.Scatter3d(
                x=f["k"], y=f["dte"], z=100.0 * f["iv_mid"], mode="markers",
                name="flagged (red rich, green cheap)",
                marker={"size": 4, "color": np.where(flags[flags != 0] > 0, RICH, CHEAP)},
            )
        )  # fmt: skip
    fig.update_layout(
        height=620, margin={"l": 0, "r": 0, "t": 10, "b": 0}, legend={"y": 0.9},
        scene={
            "xaxis_title": "log-moneyness k", "zaxis_title": "implied vol (%)",
            "yaxis": {"title": "days to expiry", "type": "log", "tickvals": DAY_TICKS},
            "camera": {"eye": {"x": -1.6, "y": -1.5, "z": 0.8}},
        },
    )  # fmt: skip
    return fig


def smile_figure(
    quotes: pd.DataFrame,
    curves: dict[str, np.ndarray],
    flags: np.ndarray | None = None,
    dense: tuple[np.ndarray, np.ndarray, str] | None = None,
    highlight: int | None = None,
) -> go.Figure:
    """One expiry: market mids with bid-ask bars, model IVs at the quotes (``curves``, by column),
    an optional dense model curve, rich (red) and cheap (green) flags, one highlighted quote."""
    q = quotes.reset_index(drop=True)
    hover = "K %{customdata[0]:.0f} %{customdata[1]}<br>k %{x:.3f}<br>%{y:.2f}%"
    custom = np.c_[q["strike"], q["option_type"]]
    fig = go.Figure(
        go.Scatter(
            x=q["k"], y=100.0 * q["iv_mid"], mode="markers", name="market (bid-ask)",
            marker={"size": 4, "color": "#111"}, customdata=custom, hovertemplate=hover,
            error_y={
                "type": "data", "symmetric": False, "thickness": 1, "width": 0, "color": "#999",
                "array": 100.0 * (q["iv_ask"] - q["iv_mid"]),
                "arrayminus": 100.0 * (q["iv_mid"] - q["iv_bid"]),
            },
        )
    )  # fmt: skip
    for name, iv in curves.items():
        fig.add_trace(
            go.Scatter(
                x=q["k"], y=100.0 * np.asarray(iv), mode="lines", name=name.removeprefix("iv_"),
                line={"color": COLORS.get(name, "#444"), "width": 1.6,
                      "dash": "dot" if name == "iv_true" else "solid"},
            )
        )  # fmt: skip
    if dense is not None:
        k, iv, label = dense
        fig.add_trace(
            go.Scatter(
                x=k, y=100.0 * iv, mode="lines", name=label,
                line={"color": COLORS["live"], "width": 2},
            )
        )  # fmt: skip
    if flags is not None and (flags != 0).any():
        f = np.asarray(flags)
        for sign, color, name in ((1, RICH, "flagged rich"), (-1, CHEAP, "flagged cheap")):
            m = f == sign
            if m.any():
                fig.add_trace(
                    go.Scatter(
                        x=q["k"][m], y=100.0 * q["iv_mid"][m], mode="markers", name=name,
                        marker={"size": 10, "color": color, "symbol": "x"},
                        customdata=custom[m], hovertemplate=hover,
                    )
                )  # fmt: skip
    if highlight is not None:
        fig.add_trace(
            go.Scatter(
                x=[q["k"][highlight]], y=[100.0 * q["iv_mid"][highlight]], mode="markers",
                name="selected", marker={"size": 18, "color": "rgba(0,0,0,0)",
                                         "line": {"color": "#000", "width": 2}},
            )
        )  # fmt: skip
    fig.update_layout(
        height=430, xaxis_title="log-moneyness k", yaxis_title="implied vol (%)",
        margin={"t": 20, "b": 40}, legend={"orientation": "h", "y": -0.2},
    )  # fmt: skip
    return fig


def path_figure(days: list, r: np.ndarray, margin: float) -> go.Figure:
    """A contract's residual in half-spreads over the following trading days."""
    fig = go.Figure(
        go.Scatter(x=days, y=r, mode="lines+markers", name="residual", line={"color": "#1f77b4"})
    )
    for level in (1.0 + margin, -(1.0 + margin)):
        fig.add_hline(y=level, line={"dash": "dot", "color": "#d62728"})
    fig.add_hline(y=0.0, line={"color": "#999"})
    fig.update_layout(
        height=300, xaxis_title="trading days after the flag", margin={"t": 20, "b": 40},
        yaxis_title="residual (half-spreads)", showlegend=False,
    )  # fmt: skip
    return fig


def flags_figure(dates: np.ndarray, counts: np.ndarray, selected, test_years) -> go.Figure:
    dates = pd.to_datetime(dates)
    test = np.isin(dates.year, list(test_years))
    fig = go.Figure()
    for mask, name, color in (
        (~test, "validation years", "#9ecae1"),
        (test, "test years", "#3182bd"),
    ):
        if mask.any():
            fig.add_trace(go.Bar(x=dates[mask], y=counts[mask], name=name, marker_color=color))
    fig.add_vline(x=pd.Timestamp(selected), line={"color": "#d62728", "width": 1.5})
    fig.update_layout(
        height=230, bargap=0, margin={"t": 10, "b": 30}, yaxis_title="flagged quotes",
        legend={"orientation": "h", "y": 1.15},
    )  # fmt: skip
    return fig
