"""Surface forecasting: linear latent dynamics and a test for comparing forecast losses."""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t


def fit_var1(z: np.ndarray, consecutive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares VAR(1), z[t+1] = c + A z[t], over rows t with ``consecutive[t]`` true (row
    t + 1 is the next day). Returns (c, A)."""
    x, y = z[:-1][consecutive[:-1]], z[1:][consecutive[:-1]]
    design = np.c_[np.ones(len(x)), x]
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef[0], coef[1:].T


def var_forecast(c: np.ndarray, A: np.ndarray, z: np.ndarray, h: int) -> np.ndarray:
    """h-step forecast from rows of ``z``."""
    for _ in range(h):
        z = c + z @ A.T
    return z


def diebold_mariano(d: np.ndarray, horizon: int) -> tuple[float, float]:
    """Test of a zero mean loss differential d (e.g. baseline loss minus model loss).

    The long-run variance is Newey-West (Bartlett) with bandwidth max(horizon, 4 (n/100)^(2/9)):
    overlapping h-step errors are autocorrelated up to lag h - 1, and Bartlett weights need a
    wider window than that to cover it. Returns the statistic and a two-sided p-value from
    t(n - 1); a positive statistic favours the model.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    lag = max(horizon, int(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    e = d - d.mean()
    var = e @ e / n
    for j in range(1, min(lag, n - 1) + 1):
        var += 2.0 * (1.0 - j / (lag + 1)) * (e[j:] @ e[:-j]) / n
    stat = d.mean() / np.sqrt(var / n)
    return float(stat), float(2.0 * student_t.sf(abs(stat), n - 1))
