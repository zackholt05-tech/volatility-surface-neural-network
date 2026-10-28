import numpy as np
import pandas as pd
import pytest

from volsurf.arbitrage.conditions import check_grid
from volsurf.data.synthetic import (
    QUOTE_COLUMNS,
    InjectionConfig,
    SyntheticConfig,
    generate,
    true_total_variance,
)


@pytest.fixture(scope="module", params=["heston", "ssvi"])
def panel(request):
    cfg = SyntheticConfig(surface=request.param, n_days=8, seed=3)
    quotes, states = generate(cfg)
    return cfg, quotes, states


def test_schema_and_quote_sanity(panel):
    _, q, states = panel
    assert list(q.columns) == QUOTE_COLUMNS
    assert q[["bid", "ask", "mid", "iv_mid", "iv_true", "F", "D"]].notna().all().all()
    assert (q["half_spread_iv"] > 0).all()
    assert (q["bid"] > 0).all() and (q["ask"] > q["bid"]).all()
    assert ((q["iv_bid"] <= q["iv_mid"]) & (q["iv_mid"] <= q["iv_ask"])).all()
    assert ((q["option_type"] == "C") == (q["strike"] >= q["F"])).all()
    assert q["date"].nunique() == len(states) == 8
    assert q["dte"].between(7, 730).all()


def test_unlabelled_quotes_track_truth(panel):
    _, q, _ = panel
    clean = q[q["shift"] == 0.0]
    rel_err = (clean["iv_mid"] - clean["iv_true"]).abs() / clean["half_spread_model"]
    assert rel_err.median() < 0.5


def test_labelled_quotes_are_displaced_in_label_direction(panel):
    _, q, _ = panel
    labelled = q[q["label"] != 0]
    assert len(labelled) > 0
    agree = np.sign(labelled["iv_mid"] - labelled["iv_true"]) == labelled["label"]
    assert agree.mean() > 0.95


def test_mispricings_decay_across_days(panel):
    cfg, q, _ = panel
    key = ["expiry", "strike"]
    dates = sorted(q["date"].unique())
    ratios = []
    for d0, d1 in zip(dates[:-1], dates[1:], strict=True):
        a = q[(q["date"] == d0) & (q["shift"].abs() >= 1.0)][key + ["shift"]]
        b = q[q["date"] == d1][key + ["shift"]]
        ratios.append(a.merge(b, on=key, suffixes=("_0", "_1")).eval("shift_1 / shift_0"))
    ratios = pd.concat(ratios)
    assert len(ratios) > 0
    assert (np.isclose(ratios, cfg.injection.decay)).mean() > 0.9


def test_clean_config_has_no_labels():
    cfg = SyntheticConfig(n_days=3, injection=InjectionConfig(point_prob=0.0, segment_prob=0.0))
    q, _ = generate(cfg)
    assert (q["label"] == 0).all() and (q["shift"] == 0.0).all()
    assert (q["injection"] == "none").all()


def test_generation_is_deterministic():
    cfg = SyntheticConfig(surface="ssvi", n_days=3, seed=11)
    pd.testing.assert_frame_equal(generate(cfg)[0], generate(cfg)[0])


def test_ground_truth_surface_is_arbitrage_free(panel):
    cfg, _, states = panel
    k = np.linspace(-0.3, 0.15, 121)
    T = np.array([30, 60, 120, 240, 480, 730]) / 365.0
    for v in states["v"].iloc[[0, -1]]:
        report = check_grid(true_total_variance(cfg, v, k, T), k, T)
        assert report.calendar_frac == 0.0 and report.butterfly_frac == 0.0, report


def test_config_from_dict_builds_nested_dataclasses():
    heston = {"kappa": 1.0, "theta": 0.05, "sigma": 0.5, "rho": -0.6, "v0": 0.04}
    cfg = SyntheticConfig.from_dict(
        {
            "surface": "ssvi",
            "heston": heston,
            "listing": {"z_range": [-3.0, 2.0]},
            "injection": {"size": [1.5, 3.0]},
        }
    )
    assert cfg.heston.kappa == 1.0
    assert cfg.listing.z_range == (-3.0, 2.0)
    assert cfg.injection.size == (1.5, 3.0)
