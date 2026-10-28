import numpy as np
import pandas as pd
import pytest
import torch

from volsurf.data.synthetic import SyntheticConfig, generate
from volsurf.eval.dayset import make_day
from volsurf.eval.runner import blocks_from_pairs
from volsurf.models.mlp import MLPConfig, MLPSurface
from volsurf.models.ssvi import power_law_phi, ssvi
from volsurf.models.svi import raw_svi
from volsurf.training.collocation import CollocationConfig, build_grid
from volsurf.training.losses import PenaltyConfig, arbitrage_penalties

ON = PenaltyConfig(calendar=1.0, butterfly=1.0, wing=1.0)


@pytest.fixture(scope="module")
def days():
    quotes, _ = generate(SyntheticConfig(surface="ssvi", n_days=2, seed=9))
    return [make_day(g) for _, g in quotes.groupby("date")]


def test_grid_covers_widened_range_and_quoted_maturities(days):
    data = days[0].fit_data()
    cfg = CollocationConfig(n_T=8, n_k=11, widen=1.0)
    grid = build_grid(data, cfg)
    T = grid.T.numpy()
    assert set(np.round(data.slice_T, 12)) <= set(np.round(T, 12))
    assert T.min() < data.slice_T.min() and T.max() > data.slice_T.max()
    assert grid.edge.sum() == 2 * len(np.unique(T))
    first = data.slice_id == 0
    width = data.k[first].max() - data.k[first].min()
    row = grid.k[np.isclose(T, data.slice_T[0])].numpy()
    assert row.min() == pytest.approx(data.k[first].min() - width)


def _grid(k_max=0.5):
    k = torch.linspace(-k_max, k_max, 51, dtype=torch.float64).repeat(4)
    T = torch.tensor([0.1, 0.25, 0.5, 1.0], dtype=torch.float64).repeat_interleave(51)
    edge = torch.zeros_like(k, dtype=torch.bool)
    edge[::51] = True
    edge[50::51] = True
    return type("G", (), {"k": k, "T": T, "edge": edge})()


def test_penalties_vanish_on_arbitrage_free_ssvi():
    def w(k, T):
        theta = 0.04 * T
        return ssvi(k, theta, -0.7, power_law_phi(theta, 1.0, 0.4))

    terms = arbitrage_penalties(w, _grid(), ON)
    assert all(float(v.detach()) == 0.0 for v in terms.values())


def test_penalties_detect_each_violation():
    grid = _grid()
    vogt = dict(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
    wide = _grid(k_max=1.5)  # the Vogt slice violates convexity away from the money
    assert arbitrage_penalties(lambda k, T: raw_svi(k, **vogt) + 0 * T, wide, ON)["butterfly"] > 0
    falling = arbitrage_penalties(lambda k, T: (0.05 - 0.02 * T) * (1 + k**2), grid, ON)
    assert falling["calendar"] > 0
    steep = arbitrage_penalties(lambda k, T: 0.04 * T + 3.0 * k.abs(), grid, ON)
    assert steep["wing"] > 0


def test_penalized_mlp_reports_terms_and_warm_starts(days):
    cfg = MLPConfig(
        hidden=(16, 16), activation="tanh", prior="ssvi", features="k_theta", time_feature="log",
        steps=30, lbfgs_iters=10, warm_steps=5, warm_lbfgs_iters=5,
        penalty=PenaltyConfig(calendar=10.0, butterfly=10.0, wing=10.0),
        collocation=CollocationConfig(n_T=5, n_k=11),
    )  # fmt: skip
    first = MLPSurface(cfg, seed=1).fit(days[0].fit_data())
    assert set(first.penalties) == {"fit", "calendar", "butterfly", "wing"}
    assert all(np.isfinite(v) for v in first.penalties.values())
    second = MLPSurface(cfg, seed=2).fit(days[1].fit_data(), init=first)
    assert second.net is not first.net
    assert torch.equal(second._mu, first._mu)
    assert np.isfinite(second.final_loss) and (second.w(days[1].k, days[1].T) > 0).all()


def test_blocks_from_pairs_groups_consecutive_days():
    d = pd.bdate_range("2022-01-03", periods=6)
    blocks = blocks_from_pairs([d[0], d[1], d[3], d[4], d[5]], [(d[0], d[1]), (d[3], d[4])])
    assert blocks == [[d[0], d[1]], [d[3], d[4]], [d[5]]]
