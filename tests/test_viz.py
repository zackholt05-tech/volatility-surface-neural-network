import sys

import numpy as np
import pandas as pd
import pytest

from volsurf.eval.metrics import ArbEvalConfig, arbitrage_metrics
from volsurf.eval.mispricing import flag
from volsurf.viz.heatmaps import arbitrage_maps
from volsurf.viz.panels import PANELS, ROOT, available_panels, flags_per_day, follow, make_panel

BOUNDS = pd.DataFrame({"T": [0.05, 0.25, 1.0], "min": [-0.2, -0.4, -0.7], "max": [0.1, 0.2, 0.4]})


class Toy:
    """Total variance that wiggles in T (calendar arbitrage) and has a concave bump in k
    (butterfly arbitrage)."""

    def w(self, k, T):
        k, T = np.broadcast_arrays(np.asarray(k, float), np.asarray(T, float))
        return T * (0.04 + 0.1 * k**2) * (1.0 + 0.6 * np.sin(12.0 * T)) - 0.02 * np.exp(
            -((k / 0.03) ** 2)
        ) * np.sqrt(T)


class Flat:
    def w(self, k, T):
        k, T = np.broadcast_arrays(np.asarray(k, float), np.asarray(T, float))
        return 0.04 * T + 0.0 * k


@pytest.mark.parametrize("widen", [0.0, 0.5])
def test_arbitrage_maps_report_the_evaluation_metrics(widen):
    cfg = ArbEvalConfig()
    maps = arbitrage_maps(Toy(), BOUNDS, cfg, widen)
    metrics = arbitrage_metrics(Toy(), BOUNDS, cfg, widen, "x")
    assert maps.butterfly_frac > 0 and maps.calendar_frac > 0
    assert maps.butterfly_frac == metrics["x/butterfly_frac"]
    assert maps.calendar_frac == metrics["x/calendar_frac"]
    assert maps.calendar_vp.max() == pytest.approx(metrics["x/calendar_max_vp"])
    assert maps.g.min() == metrics["x/butterfly_min_g"]


def test_arbitrage_free_surface_has_empty_maps():
    maps = arbitrage_maps(Flat(), BOUNDS)
    assert maps.butterfly_frac == 0 and maps.calendar_frac == 0
    assert maps.calendar_vp.max() == 0


def _panel():
    """Two contracts over three days spanning a year end; the second is unquoted on day 2."""
    rows = []
    for d, date in enumerate(["2022-12-29", "2022-12-30", "2023-01-03"]):
        for strike in (3800.0, 4000.0):
            if d == 1 and strike == 4000.0:
                continue
            rows.append({"date": pd.Timestamp(date), "expiry": pd.Timestamp("2023-03-17"),
                         "strike": strike, "option_type": "C",
                         "r": 10 * d + strike / 1000})  # fmt: skip
    table = pd.DataFrame(rows).sample(frac=1.0, random_state=0)  # unsorted on purpose
    return make_panel(PANELS[0], table, fits={})


def test_follow_crosses_the_year_end_and_marks_unquoted_contracts():
    panel = _panel()
    r = panel.table["r"].to_numpy()
    assert list(np.diff(panel.starts)) == [2, 1, 2]
    out = follow(panel, r, 0, horizons=(1, 2, 3))
    day0 = panel.day(0).set_index("strike").index
    by_strike = out.set_axis(day0)
    assert by_strike.loc[3800.0, "r_1d"] == pytest.approx(13.8)
    assert np.isnan(by_strike.loc[4000.0, "r_1d"])  # not quoted on 2022-12-30
    assert by_strike.loc[4000.0, "r_2d"] == pytest.approx(24.0)  # 2023-01-03
    assert out["r_3d"].isna().all()  # past the end of the panel


def test_flags_per_day_counts_every_flag():
    panel = _panel()
    r = np.array([0.0, 6.0, -7.0, 0.5, 5.5])
    counts = flags_per_day(panel, r, 4.0)
    assert counts.sum() == (flag(r, 4.0) != 0).sum()
    assert list(counts) == [1, 1, 1]


@pytest.mark.skipif(not available_panels(), reason="no fitted panels under results/")
def test_dashboard_pages_run():
    from streamlit.testing.v1 import AppTest

    sys.path.insert(0, str(ROOT / "dashboard"))
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=300).run()
    assert not at.exception, at.exception
    for page in ("views/arbitrage.py", "views/mispricings.py"):
        at.switch_page(page).run()
        assert not at.exception, at.exception
