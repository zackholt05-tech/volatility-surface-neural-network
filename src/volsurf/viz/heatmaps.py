"""Static-arbitrage maps of a fitted surface, on the grids of `volsurf.eval.metrics`.

`arbitrage_maps` evaluates the butterfly function g and the calendar decrease of total variance
exactly where `arbitrage_metrics` does (the same maturities, k-grids, domain and tolerances), so
the shares of violating points it reports are the evaluation's own numbers. The figures put both
on a common log-moneyness axis, with the quoted domain marked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from volsurf.arbitrage.conditions import durrleman_g, fd_slice_derivatives
from volsurf.eval.metrics import ArbEvalConfig, domain
from volsurf.viz.surfaces import DAY_TICKS


@dataclass
class ArbitrageMaps:
    T: np.ndarray  # grid maturities: log-spaced points and every quoted maturity
    K: np.ndarray  # (n_T, n_k) butterfly k-grid of each maturity, over its domain
    g: np.ndarray  # (n_T, n_k - 2) Durrleman g at the interior points of K
    kc: np.ndarray  # common k-grid of the calendar check
    dW: np.ndarray  # (n_T - 1, n_kc) change of w from T_i to T_i+1
    both: np.ndarray  # where the calendar check applies: inside the domains of T_i and T_i+1
    calendar_vp: np.ndarray  # size of each total-variance decrease in vol points (0 if none)
    quoted: tuple[np.ndarray, np.ndarray]  # quoted k-range at each grid maturity
    cfg: ArbEvalConfig

    @property
    def butterfly(self) -> np.ndarray:
        return self.g < -self.cfg.tol_g

    @property
    def butterfly_frac(self) -> float:
        return float(np.mean(self.butterfly))

    @property
    def calendar_frac(self) -> float:
        dW = self.dW[self.both]
        return float(np.mean(dW < -self.cfg.tol_cal)) if dW.size else 0.0


def arbitrage_maps(
    model, bounds: pd.DataFrame, cfg: ArbEvalConfig | None = None, widen: float | None = None
) -> ArbitrageMaps:
    """Butterfly and calendar checks of ``model`` over its quoted domain widened by ``widen``
    (default: the evaluation's extended domain)."""
    cfg = cfg or ArbEvalConfig()
    widen = cfg.widen if widen is None else widen
    quoted_T = bounds["T"].to_numpy()
    T = np.union1d(np.geomspace(quoted_T.min(), quoted_T.max(), cfg.n_T), quoted_T)
    lo, hi = domain(bounds, T, widen)

    K = lo[:, None] + np.linspace(0.0, 1.0, cfg.n_k)[None, :] * (hi - lo)[:, None]
    W = model.w(K, np.broadcast_to(T[:, None], K.shape))
    g = np.stack([durrleman_g(*fd_slice_derivatives(W[i], K[i])) for i in range(len(T))])

    kc = np.linspace(lo.min(), hi.max(), cfg.n_k_cal)
    shape = (len(T), kc.size)
    Wc = model.w(np.broadcast_to(kc, shape), np.broadcast_to(T[:, None], shape))
    inside = (kc[None, :] >= lo[:, None]) & (kc[None, :] <= hi[:, None])
    both = inside[:-1] & inside[1:]
    dW = np.diff(Wc, axis=0)
    earlier = np.sqrt(np.maximum(Wc[:-1], 1e-12) * T[:-1, None])
    calendar_vp = 100.0 * np.where(both, np.maximum(-dW, 0.0) / (2.0 * earlier), 0.0)
    return ArbitrageMaps(T, K, g, kc, dW, both, calendar_vp, domain(bounds, T), cfg)


def _on_common_grid(maps: ArbitrageMaps) -> np.ndarray:
    """Butterfly violation size (-g where g < -tol, else 0) of each maturity interpolated onto
    the calendar k-grid; NaN outside that maturity's domain."""
    out = np.full((len(maps.T), maps.kc.size), np.nan)
    size = np.where(maps.butterfly, -maps.g, 0.0)
    for i in range(len(maps.T)):
        k = maps.K[i, 1:-1]
        on = (maps.kc >= k[0]) & (maps.kc <= k[-1])
        out[i, on] = np.interp(maps.kc[on], k, size[i])
    return out


def _map_figure(maps: ArbitrageMaps, z: np.ndarray, rows: np.ndarray, title: str, unit: str):
    zmax = float(np.nanmax(z)) if np.isfinite(z).any() and np.nanmax(z) > 0 else 1.0
    fig = go.Figure(
        go.Heatmap(
            x=maps.kc, y=365.0 * rows, z=z, zmin=0.0, zmax=zmax, colorbar={"title": unit},
            colorscale=[[0.0, "#e8ecf0"], [1e-6, "#fcbba1"], [0.5, "#ef3b2c"], [1.0, "#67000d"]],
            hovertemplate="k %{x:.3f}<br>%{y:.0f} days<br>%{z:.3g}<extra></extra>",
        )
    )  # fmt: skip
    lo, hi = maps.quoted
    for edge, name in ((lo, "quoted range"), (hi, None)):
        fig.add_trace(
            go.Scatter(
                x=edge, y=365.0 * maps.T, mode="lines", line={"color": "#222", "dash": "dot"},
                name=name or "", showlegend=name is not None, hoverinfo="skip",
            )
        )  # fmt: skip
    fig.update_layout(
        title=title, xaxis_title="log-moneyness k", plot_bgcolor="white", height=430,
        yaxis={"title": "days to expiry", "type": "log", "tickvals": DAY_TICKS},
        margin={"t": 50, "b": 40},
        legend={"orientation": "h", "y": -0.18},
    )  # fmt: skip
    return fig


def butterfly_figure(maps: ArbitrageMaps) -> go.Figure:
    title = f"Butterfly: Durrleman g < 0 on {100 * maps.butterfly_frac:.2f}% of the grid"
    return _map_figure(maps, _on_common_grid(maps), maps.T, title, "−g")


def calendar_figure(maps: ArbitrageMaps) -> go.Figure:
    z = np.where(maps.both, maps.calendar_vp, np.nan)
    title = f"Calendar: total variance falls with maturity on {100 * maps.calendar_frac:.2f}%"
    return _map_figure(maps, z, maps.T[:-1], title, "vol pts")
