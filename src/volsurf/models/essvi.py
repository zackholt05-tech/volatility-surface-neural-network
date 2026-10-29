"""Extended SSVI (Hendriks & Martini 2019) with an always-feasible no-arbitrage parametrization.

An eSSVI slice has ATM total variance theta, skew-scale psi = theta phi and correlation rho:

    w(k) = 1/2 (theta + rho psi k + sqrt((psi k + theta rho)^2 + theta^2 (1 - rho^2))).

Writing it through the wing slopes u_+ = psi (1 + rho) and u_- = psi (1 - rho) (w ~ u_+/-|k|/2 as
k -> +/-inf) turns the known sufficient conditions into monotonicity and bound constraints. For
maturities T_1 < T_2, the slices are free of calendar arbitrage if (Hendriks & Martini 2019,
Prop. 3.5, in the form used by Mingone 2022)

    theta_2 > theta_1,   u_+/-(T_2) >= u_+/-(T_1),   psi_2 / theta_2 <= psi_1 / theta_1,

and each slice is free of butterfly arbitrage if (Gatheral & Jacquier 2014, Thm. 4.2)

    psi (1 + |rho|) = max(u_+, u_-) < 4,   psi^2 (1 + |rho|) = psi max(u_+, u_-) <= 4 theta.

Every condition is a monotonicity in T, a bound on the wings, or a lower bound on theta given the
wings, so knots can be generated in maturity order from unconstrained "rates" with no feasibility
coupling to later knots: wings first (nondecreasing, below 4), then theta above its lower bounds.
Between knots, (theta, u_+, u_-) are interpolated linearly in T, which preserves every condition
(Corbetta et al. 2019); outside, implied vol is held flat at fixed k. The surface is therefore free
of static arbitrage for any parameter vector: violations are zero by theorem, not by tuning.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import least_squares

from volsurf.eval.dayset import FitData
from volsurf.models.base import W_FLOOR, SurfaceModel
from volsurf.models.ssvi import SSVISurface, power_law_phi

U_MAX = 4.0  # GJ / Lee bound on psi (1 + |rho|): total-variance wing slope below 2
THETA_FLOOR = 1e-8  # strict increase of theta between knots
T0_DAYS = 1.0  # reference maturity (days) at which both wings start from zero


def essvi_w(k, theta, u_plus, u_minus):
    """eSSVI total variance in wing coordinates; NumPy arrays or torch tensors."""
    psi = 0.5 * (u_plus + u_minus)
    psi_rho = 0.5 * (u_plus - u_minus)
    disc = (psi * k) ** 2 + 2.0 * theta * psi_rho * k + theta * theta
    return 0.5 * (theta + psi_rho * k + disc**0.5)


def _softplus_np(x):
    return np.logaddexp(0.0, x)


def _softplus_inv(y):
    y = np.maximum(y, 1e-12)
    return np.where(y > 30.0, y, np.log(np.expm1(np.minimum(y, 30.0))))


def _steps(knot_T, lib):
    """Maturity steps dT and log-maturity steps from (0, T0) to each knot."""
    t0 = min(T0_DAYS / 365.0, 0.5 * float(knot_T[0]))
    if lib is np:
        T = np.asarray(knot_T, dtype=float)
        dT = np.diff(np.r_[0.0, T])
        dlog = np.diff(np.log(np.r_[t0, T]))
    else:
        T = knot_T
        dT = torch.diff(T, prepend=T.new_zeros(1))
        dlog = torch.diff(torch.log(T), prepend=torch.log(T.new_full((1,), t0)))
    return dT, dlog


def knots_from_rates(knot_T, rates, lib=np):
    """Map unconstrained rates (n_knots, 3) to feasible knots (theta, u_+, u_-).

    Columns: forward-variance excess (theta grows by softplus(a) dT above its lower bound) and the
    two wing rates (U_MAX - u decays by exp(-softplus(a) dlogT)). The map is onto the feasible set:
    every arbitrage-free knot sequence (with strict increase) has a preimage.
    """
    if lib is np:
        softplus, exp, cumsum, maximum = _softplus_np, np.exp, np.cumsum, np.maximum
    else:
        softplus, exp = torch.nn.functional.softplus, torch.exp
        cumsum, maximum = (lambda x: torch.cumsum(x, 0)), torch.maximum
    dT, dlog = _steps(knot_T, lib)
    fv, r_plus, r_minus = softplus(rates[:, 0]), softplus(rates[:, 1]), softplus(rates[:, 2])
    u_plus = U_MAX * (1.0 - exp(-cumsum(r_plus * dlog)))
    u_minus = U_MAX * (1.0 - exp(-cumsum(r_minus * dlog)))
    s = u_plus + u_minus
    floor = s * maximum(u_plus, u_minus) / 8.0
    inc = fv * dT + THETA_FLOOR
    theta = [floor[0] + inc[0]]
    for i in range(1, len(inc)):
        theta.append(maximum(theta[-1] * s[i] / s[i - 1], floor[i]) + inc[i])
    theta = np.array(theta) if lib is np else torch.stack(theta)
    return theta, u_plus, u_minus


def rates_from_knots(knot_T, theta, u_plus, u_minus):
    """Inverse of ``knots_from_rates`` for a target knot sequence, clipped where it is infeasible
    (the returned rates then generate the nearest feasible sequence in maturity order)."""
    dT, dlog = _steps(knot_T, np)

    def wing_rates(u):
        u = np.minimum(np.maximum.accumulate(np.maximum(u, 1e-12)), U_MAX * (1 - 1e-9))
        gap = np.log(U_MAX - np.r_[0.0, u])
        return _softplus_inv(np.maximum(-np.diff(gap), 1e-9) / dlog)

    rates = np.empty((len(theta), 3))
    rates[:, 1], rates[:, 2] = wing_rates(u_plus), wing_rates(u_minus)
    _, up, um = knots_from_rates(knot_T, rates)  # wings as the forward map generates them
    s = up + um
    floor = s * np.maximum(up, um) / 8.0
    prev = None
    for i in range(len(theta)):
        lower = floor[i] if prev is None else max(prev * s[i] / s[i - 1], floor[i])
        inc = max(theta[i] - lower - THETA_FLOOR, 1e-9 * dT[i])
        rates[i, 0] = _softplus_inv(inc / dT[i])
        prev = lower + inc + THETA_FLOOR
    return rates


def interp_knots(T, knot_T, theta, u_plus, u_minus):
    """(theta, u_+, u_-) at maturities T: linear between knots, flat implied vol outside."""
    T = np.asarray(T, dtype=float)
    th = np.interp(T, knot_T, theta)
    up = np.interp(T, knot_T, u_plus)
    um = np.interp(T, knot_T, u_minus)
    # Before the first knot, theta and both wings scale with T/T_1: w is homogeneous of degree
    # one in (theta, u_+, u_-), so implied vol at fixed k is unchanged. After the last knot the
    # wings stay fixed and theta grows in proportion to T (flat ATM vol), so psi/theta decreases.
    short = T < knot_T[0]
    scale = T / knot_T[0]
    th = np.where(short, theta[0] * scale, th)
    up = np.where(short, u_plus[0] * scale, up)
    um = np.where(short, u_minus[0] * scale, um)
    th = np.where(T > knot_T[-1], theta[-1] * T / knot_T[-1], th)
    return th, up, um


def ssvi_knots(data: FitData):
    """Knots of a power-law SSVI fit, which lie inside the eSSVI feasible set (constant rho,
    phi nonincreasing, psi increasing, eta (1 + |rho|) <= 2)."""
    ssvi = SSVISurface().fit(data)
    psi = ssvi.theta * power_law_phi(ssvi.theta, ssvi.eta, ssvi.gamma)
    return ssvi.theta, psi * (1.0 + ssvi.rho), psi * (1.0 - ssvi.rho)


class ESSVISurface(SurfaceModel):
    """eSSVI with one knot per quoted maturity, calibrated jointly (Mingone-style global fit)
    and arbitrage-free by construction. Initialized from a power-law SSVI fit."""

    name = "essvi"

    def __init__(self, max_nfev: int = 300) -> None:
        super().__init__()
        self.max_nfev = max_nfev

    def fit(self, data: FitData) -> ESSVISurface:
        knot_T = np.asarray(data.slice_T, dtype=float)
        n = len(knot_T)
        x0 = rates_from_knots(knot_T, *ssvi_knots(data)).ravel()
        k, iv, s, sid, T = data.k, data.iv, data.s, data.slice_id, data.T

        def residuals(x):
            theta, up, um = knots_from_rates(knot_T, x.reshape(n, 3))
            w = essvi_w(k, theta[sid], up[sid], um[sid])
            return (np.sqrt(np.maximum(w, W_FLOOR) / T) - iv) / s

        sol = least_squares(residuals, x0, method="trf", x_scale="jac", max_nfev=self.max_nfev)
        if sol.status <= 0:
            self.n_failed += 1
        self.knot_T = knot_T
        self.theta, self.u_plus, self.u_minus = knots_from_rates(knot_T, sol.x.reshape(n, 3))
        return self

    def w(self, k, T) -> np.ndarray:
        k, T = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(T, dtype=float))
        th, up, um = interp_knots(T, self.knot_T, self.theta, self.u_plus, self.u_minus)
        return essvi_w(k, th, up, um)
