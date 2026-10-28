"""Surface SVI (Gatheral & Jacquier 2014) with the power-law curvature function.

Functions are written with arithmetic operators only, so they accept NumPy arrays or torch tensors.
"""

from __future__ import annotations

import numpy as np


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
