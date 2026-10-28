import numpy as np
import pandas as pd
import pytest
import torch

from volsurf.data.synthetic import SyntheticConfig, generate
from volsurf.eval import metrics
from volsurf.eval.dayset import FitData, HoldoutConfig, make_day
from volsurf.eval.runner import SamplingConfig, flatten, sample_dates
from volsurf.models.base import SliceSurface
from volsurf.models.mlp import MLPConfig, MLPSurface
from volsurf.models.spline import SplineSurface
from volsurf.models.ssvi import SSVISurface
from volsurf.models.svi import SVISurface, fit_raw_svi, raw_svi

ARB = metrics.ArbEvalConfig(n_T=12, n_k=61, n_k_cal=201)


@pytest.fixture(scope="module")
def ssvi_day():
    cfg = SyntheticConfig(surface="ssvi", n_days=2, seed=5)
    quotes, _ = generate(cfg)
    return cfg, make_day(quotes[quotes["date"] == quotes["date"].iloc[0]])


def test_holdout_masks_partition_and_are_date_seeded(ssvi_day):
    _, day = ssvi_day
    total = day.fit.astype(int) + day.interp.astype(int) + day.wing.astype(int)
    assert (total == 1).all()
    for i in np.unique(day.slice_id):
        m = day.slice_id == i
        if m.sum() >= HoldoutConfig().min_slice:
            k, wing = day.k[m], day.wing[m]
            assert wing[0] and wing[-1]
            assert k[wing].min() == k.min() and k[wing].max() == k.max()
    again = make_day(day.quotes)
    assert (again.interp == day.interp).all()


def test_ssvi_recovers_truth_and_is_arbitrage_free(ssvi_day):
    cfg, day = ssvi_day
    model = SSVISurface().fit(day.fit_data())
    assert model.rho == pytest.approx(cfg.ssvi_rho, abs=0.02)
    assert model.eta == pytest.approx(cfg.ssvi_eta, abs=0.05)
    assert model.gamma == pytest.approx(cfg.ssvi_gamma, abs=0.05)
    truth = metrics.quote_metrics(day, model.iv(day.k, day.T), day.interp, "x")["x/truth_rmse_vp"]
    assert truth < 100 * np.median(day.s)
    bounds = day.slice_bounds()
    for widen in (0.0, 0.5):
        arb = metrics.arbitrage_metrics(model, bounds, ARB, widen, "a")
        assert arb["a/calendar_frac"] == 0.0 and arb["a/butterfly_frac"] == 0.0
        assert arb["a/calendar_frac_material"] == 0.0 and arb["a/calendar_max_vp"] == 0.0


def test_arbitrage_metrics_flag_vogt_butterfly():
    params = dict(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)

    class Vogt:
        def w(self, k, T):
            return raw_svi(np.asarray(k), **params) * np.ones_like(np.asarray(T, dtype=float))

    bounds = pd.DataFrame({"T": [1.0], "min": [-1.5], "max": [1.5]})
    arb = metrics.arbitrage_metrics(Vogt(), bounds, ARB, 0.0, "a")
    assert arb["a/butterfly_frac"] > 0.0 and arb["a/butterfly_min_g"] < 0.0


def test_svi_recovers_exact_slice():
    k = np.linspace(-0.4, 0.2, 40)
    T, true = 0.25, (0.004, 0.08, -0.7, 0.03, 0.12)
    w = raw_svi(k, *true)
    weight = np.full_like(k, 1.0 / (2 * np.sqrt(w / T) * T * 1e-3) ** 2)
    fitted = fit_raw_svi(k, w, weight, T)
    np.testing.assert_allclose(raw_svi(k, *fitted), w, rtol=1e-4)


def test_slice_surface_interpolates_total_variance_linearly():
    class Toy(SliceSurface):
        name = "toy"

        def fit_slice(self, k, w, weight, T):
            return lambda x, T=T: T * 0.04 * (1 + x**2)

    k = np.tile(np.linspace(-0.1, 0.1, 6), 2)
    T = np.repeat([0.1, 0.5], 6)
    data = FitData(
        k, T, np.full(12, 0.2), np.full(12, 0.01), np.repeat([0, 1], 6), np.array([0.1, 0.5])
    )
    model = Toy().fit(data)
    np.testing.assert_allclose(model.w(0.05, 0.3), 0.5 * (0.1 + 0.5) * 0.04 * (1 + 0.05**2))
    np.testing.assert_allclose(model.iv(0.0, [0.05, 1.0]), [0.2, 0.2])  # flat vol outside


def test_spline_fits_within_noise(ssvi_day):
    _, day = ssvi_day
    model = SplineSurface().fit(day.fit_data())
    m = metrics.quote_metrics(day, model.iv(day.k, day.T), day.fit, "fit")
    assert m["fit/rms_norm"] < 1.5


def test_svi_surface_fits_synthetic_day(ssvi_day):
    _, day = ssvi_day
    model = SVISurface().fit(day.fit_data())
    assert model.n_failed == 0
    m = metrics.quote_metrics(day, model.iv(day.k, day.T), day.fit, "fit")
    assert m["fit/within_spread"] > 0.8


@pytest.mark.parametrize("prior", ["none", "ssvi"])
def test_mlp_trains_and_is_deterministic(ssvi_day, prior):
    _, day = ssvi_day
    cfg = MLPConfig(hidden=(16, 16), steps=60, lbfgs_iters=20, prior=prior)
    a = MLPSurface(cfg, seed=7).fit(day.fit_data())
    b = MLPSurface(cfg, seed=7).fit(day.fit_data())
    assert a.final_loss == b.final_loss and np.isfinite(a.final_loss)
    w = a.w(day.k, day.T)
    assert (w > 0).all()
    k_t, T_t = torch.tensor(day.k[:5]), torch.tensor(day.T[:5])
    np.testing.assert_allclose(a.w_torch(k_t, T_t).detach().numpy(), w[:5])
    m = metrics.quote_metrics(day, a.iv(day.k, day.T), day.fit, "fit")
    assert m["fit/rms_norm"] < 5.0


def test_sample_dates_blocks_and_pairs():
    dates = pd.bdate_range("2022-01-03", periods=25)
    chosen, pairs = sample_dates(dates, SamplingConfig(every=10, block=2))
    assert chosen == [dates[0], dates[1], dates[10], dates[11], dates[20], dates[21]]
    assert pairs == [(dates[0], dates[1]), (dates[10], dates[11]), (dates[20], dates[21])]


def test_flatten_nested_config():
    assert flatten({"a": {"b": 1, "c": [1, 2]}, "d": "x"}) == {"a.b": 1, "a.c": "[1, 2]", "d": "x"}
