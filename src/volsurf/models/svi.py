"""Raw SVI slice parameterization (Gatheral 2004).

Functions are written with arithmetic operators only, so they accept NumPy arrays or torch tensors.
"""

from __future__ import annotations


def raw_svi(k, a, b, rho, m, sigma):
    """Total implied variance w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))."""
    x = k - m
    return a + b * (rho * x + (x * x + sigma * sigma) ** 0.5)
