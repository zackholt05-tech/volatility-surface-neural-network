import numpy as np
import pandas as pd

from volsurf.eval import metrics
from volsurf.models.base import SurfaceModel

ARB = metrics.ArbEvalConfig(n_T=8, n_k=41, n_k_cal=81)
BOUNDS = pd.DataFrame({"T": [0.05, 0.25, 1.0], "min": [-0.3, -0.5, -0.8], "max": [0.1, 0.2, 0.3]})


class FlatSurface(SurfaceModel):
    """Flat 20% implied vol, optionally NaN beyond a strike."""

    def __init__(self, nan_above: float = np.inf) -> None:
        super().__init__()
        self.nan_above = nan_above

    def fit(self, data):
        return self

    def w(self, k, T):
        k, T = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(T, dtype=float))
        return np.where(k > self.nan_above, np.nan, 0.04 * T)


def test_nonfinite_surface_is_reported_rather_than_passing_as_arbitrage_free():
    clean = metrics.arbitrage_metrics(FlatSurface(), BOUNDS, ARB, 0.5, "a")
    assert clean["a/nonfinite_frac"] == 0.0 and clean["a/butterfly_frac"] == 0.0

    broken = metrics.arbitrage_metrics(FlatSurface(nan_above=0.0), BOUNDS, ARB, 0.5, "a")
    assert broken["a/nonfinite_frac"] > 0.1
    # the violation fractions alone cannot tell this surface from a clean one
    assert broken["a/calendar_frac"] == 0.0 and broken["a/butterfly_frac"] == 0.0
