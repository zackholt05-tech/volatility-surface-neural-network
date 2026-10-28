"""Surface SVI (Gatheral & Jacquier 2014) with the power-law curvature function.

The functional forms are written with arithmetic operators only, so they accept NumPy arrays or
torch tensors. ``SSVISurface`` calibrates a full surface to one quote date.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from volsurf.eval.dayset import FitData
from volsurf.models.base import W_FLOOR, SurfaceModel


def ssvi(k, theta, rho, phi):
    """Total implied variance

    w(k, theta) = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2)).
    """
    x = phi * k + rho
    return 0.5 * theta * (1.0 + rho * phi * k + (x * x + 1.0 - rho * rho) ** 0.5)


def power_law_phi(theta, eta, gamma):
    """phi(theta) = eta / (theta^gamma (1 + theta)^(1 - gamma))."""
    return eta / (theta**gamma * (1.0 + theta) ** (1.0 - gamma))


def gj_butterfly_free(theta, rho, phi):
    """Sufficient butterfly conditions (GJ 2014, Thm 4.2): theta phi (1+|rho|) < 4 and
    theta phi^2 (1+|rho|) <= 4."""
    theta, phi = np.asarray(theta), np.asarray(phi)
    s = 1.0 + abs(rho)
    return (theta * phi * s < 4.0) & (theta * phi**2 * s <= 4.0)


def gj_calendar_free(theta, phi, rho) -> bool:
    """Calendar conditions (GJ 2014, Thm 4.1) on a maturity-ordered grid of (theta_t, phi_t):
    theta nondecreasing and 0 <= d(theta phi)/d theta <= (1 + sqrt(1 - rho^2)) phi / rho^2."""
    theta, phi = np.asarray(theta, dtype=float), np.asarray(phi, dtype=float)
    if np.any(np.diff(theta) < 0.0):
        return False
    dtheta = np.diff(theta)
    moving = dtheta > 0.0
    slope = np.diff(theta * phi)[moving] / dtheta[moving]
    upper = np.inf if rho == 0 else (1.0 + np.sqrt(1.0 - rho**2)) / rho**2
    phi_mid = np.minimum(phi[:-1], phi[1:])[moving]
    return bool(np.all(slope >= 0.0) and np.all(slope <= upper * phi_mid))


def _softplus(x):
    return np.logaddexp(0.0, x)


def _softplus_inv(y):
    return np.where(y > 30.0, y, np.log(np.expm1(np.maximum(y, 1e-12))))


class SSVISurface(SurfaceModel):
    """SSVI with power-law phi, free of static arbitrage by construction.

    Parameters are an ATM total variance theta_i per quoted slice, kept nondecreasing in T as a
    cumulative sum of softplus increments, plus global (rho, eta, gamma) mapped so that
    eta (1 + |rho|) <= 2 and gamma in (0, 1/2]. By Gatheral & Jacquier (2014, Thm 4.1 and
    Remark 4.4) the surface then has no calendar or butterfly arbitrage. Between slices theta is
    interpolated linearly in T; outside, ATM implied vol is held flat.
    """

    name = "ssvi"

    def __init__(self, max_nfev: int = 2000) -> None:
        super().__init__()
        self.max_nfev = max_nfev

    @staticmethod
    def _unpack(x, n):
        theta = np.cumsum(_softplus(x[:n]))
        rho = 0.999 * np.tanh(x[n])
        eta = 2.0 / (1.0 + abs(rho)) * expit(x[n + 1])
        gamma = 0.5 * expit(x[n + 2])
        return theta, rho, eta, gamma

    def fit(self, data: FitData) -> SSVISurface:
        n = len(data.slice_T)
        # Initial theta: market ATM total variance per slice, made nondecreasing.
        atm = np.empty(n)
        for i in range(n):
            m = data.slice_id == i
            atm[i] = np.interp(0.0, data.k[m], data.w[m]) if m.any() else np.nan
        atm = np.maximum.accumulate(np.nan_to_num(atm, nan=np.nanmin(atm)))
        increments = np.diff(np.r_[0.0, atm])
        x0 = np.r_[_softplus_inv(np.maximum(increments, 1e-6)), np.arctanh(-0.7 / 0.999), 0.0, 0.0]

        k, iv, s, sid, T = data.k, data.iv, data.s, data.slice_id, data.T

        def residuals(x):
            theta, rho, eta, gamma = self._unpack(x, n)
            th = theta[sid]
            w = ssvi(k, th, rho, power_law_phi(th, eta, gamma))
            return (np.sqrt(np.maximum(w, W_FLOOR) / T) - iv) / s

        sol = least_squares(residuals, x0, method="trf", x_scale="jac", max_nfev=self.max_nfev)
        if not sol.success:
            self.n_failed += 1
        self.slice_T = data.slice_T
        self.theta, self.rho, self.eta, self.gamma = self._unpack(sol.x, n)
        return self

    def theta_of(self, T):
        T = np.asarray(T, dtype=float)
        Ts, th = self.slice_T, self.theta
        out = np.interp(T, Ts, th)
        out = np.where(T < Ts[0], th[0] * T / Ts[0], out)
        return np.where(T > Ts[-1], th[-1] * T / Ts[-1], out)

    def w(self, k, T) -> np.ndarray:
        k, T = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(T, dtype=float))
        theta = self.theta_of(T)
        return ssvi(k, theta, self.rho, power_law_phi(theta, self.eta, self.gamma))
