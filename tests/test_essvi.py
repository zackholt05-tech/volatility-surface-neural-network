import numpy as np
import pytest
import torch

from volsurf.arbitrage.conditions import check_grid
from volsurf.data.synthetic import SyntheticConfig, generate
from volsurf.eval import metrics
from volsurf.eval.dayset import make_day
from volsurf.models.essvi import (
    U_MAX,
    ESSVISurface,
    essvi_w,
    interp_knots,
    knots_from_rates,
    rates_from_knots,
)
from volsurf.models.neural_essvi import (
    NeuralESSVIConfig,
    NeuralESSVISurface,
    _log_otm_price_torch,
    mixture_total_variance,
)
from volsurf.pricing import black76

ARB = metrics.ArbEvalConfig(n_T=12, n_k=61, n_k_cal=201)


def random_knots(rng, n=12):
    knot_T = np.sort(rng.uniform(7 / 365, 2.5, n))
    rates = rng.normal(0.0, 2.0, (n, 3))
    return knot_T, rates


def test_knots_from_rates_always_satisfy_the_sufficient_conditions():
    rng = np.random.default_rng(0)
    for _ in range(200):
        knot_T, rates = random_knots(rng)
        theta, up, um = knots_from_rates(knot_T, rates)
        s, m = up + um, np.maximum(up, um)
        assert np.all(np.diff(theta) > 0) and theta[0] > 0
        assert np.all(np.diff(up) >= 0) and np.all(np.diff(um) >= 0)
        assert np.all(m < U_MAX) and np.all(np.minimum(up, um) > 0)
        assert np.all(np.diff(s / theta) <= 1e-12 * s[1:] / theta[1:])  # psi/theta nonincreasing
        assert np.all(s * m <= 8.0 * theta * (1 + 1e-12))  # GJ: psi^2 (1 + |rho|) <= 4 theta
        torch_knots = knots_from_rates(torch.as_tensor(knot_T), torch.as_tensor(rates), lib=torch)
        for a, b in zip((theta, up, um), torch_knots, strict=True):
            np.testing.assert_allclose(a, b.numpy(), rtol=1e-12)


def test_knots_stay_finite_with_finite_gradients_for_extreme_rates():
    knot_T = np.sort(np.random.default_rng(7).uniform(7 / 365, 2.5, 12))
    for value in (-1000.0, -94.0, 0.0, 5.0):  # very negative rates once underflowed to u = 0
        rates = torch.full((12, 3), value, dtype=torch.float64, requires_grad=True)
        knots = knots_from_rates(torch.as_tensor(knot_T), rates, lib=torch)
        sum(kn.sum() for kn in knots).backward()
        assert all(torch.isfinite(kn).all() for kn in knots) and torch.isfinite(rates.grad).all()
        theta, up, um = knots_from_rates(knot_T, rates.detach().numpy())
        assert np.all(np.isfinite(theta)) and np.all(np.minimum(up, um) > 0)


def test_rates_from_knots_inverts_the_generator():
    rng = np.random.default_rng(1)
    for _ in range(50):
        knot_T, rates = random_knots(rng)
        knots = knots_from_rates(knot_T, rates)
        again = knots_from_rates(knot_T, rates_from_knots(knot_T, *knots))
        for a, b in zip(knots, again, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-8)


def surface_grid(w_fn, knot_T):
    T = np.union1d(np.geomspace(0.3 * knot_T[0], 2.0 * knot_T[-1], 60), knot_T)
    k = np.linspace(-1.5, 1.0, 301)
    return w_fn(k[None, :], T[:, None]), k, T


def test_random_essvi_surfaces_have_no_static_arbitrage():
    rng = np.random.default_rng(2)
    for _ in range(30):
        knot_T, rates = random_knots(rng)
        knots = knots_from_rates(knot_T, rates)

        def w_fn(k, T, knot_T=knot_T, knots=knots):
            return essvi_w(k, *interp_knots(T, knot_T, *knots))

        w, k, T = surface_grid(w_fn, knot_T)
        report = check_grid(w, k, T)
        assert report.calendar_frac == 0.0 and report.butterfly_frac == 0.0, report


def test_price_mixtures_of_essvi_surfaces_have_no_static_arbitrage():
    rng = np.random.default_rng(3)
    for _ in range(15):
        knot_T = np.sort(rng.uniform(7 / 365, 2.5, 10))
        model = NeuralESSVISurface(NeuralESSVIConfig(components=3))
        model.knot_T = knot_T
        model.knots = [knots_from_rates(knot_T, rng.normal(0.0, 2.0, (10, 3))) for _ in range(3)]
        model.weights = rng.dirichlet(np.ones(3))
        w, k, T = surface_grid(model.w, knot_T)
        assert np.isfinite(w).all()
        report = check_grid(w, k, T)
        assert report.calendar_frac == 0.0 and report.butterfly_frac == 0.0, report


def test_log_price_inversion_round_trips_into_the_far_wings():
    rng = np.random.default_rng(4)
    k = rng.uniform(-2.0, 1.5, 5000)
    w = rng.uniform(1e-4, 1.0, 5000)
    v = black76.total_vol_from_log_price(
        black76.log_otm_price(k, w), k, 0.3 * np.sqrt(w), 3 * np.sqrt(w)
    )
    np.testing.assert_allclose(v, np.sqrt(w), rtol=1e-9)
    # agrees with the price-space inversion where prices are comfortably representable
    T = 0.5
    price = np.exp(black76.log_otm_price(k, w))
    ok = price > 1e-8
    iv = black76.implied_vol(price[ok], 1.0, np.exp(k[ok]), T, k[ok] >= 0)
    np.testing.assert_allclose(iv * np.sqrt(T), v[ok], rtol=1e-8)


def test_torch_log_price_matches_numpy_with_finite_gradients_where_prices_underflow():
    rng = np.random.default_rng(6)
    k = np.r_[rng.uniform(-2.0, 1.5, 2000), 0.12, -0.3]
    w = np.r_[10.0 ** rng.uniform(-6, 0, 2000), 9e-6, 1e-5]  # last two: price < 1e-300
    w_t = torch.tensor(w, requires_grad=True)
    log_p = _log_otm_price_torch(torch.tensor(k), w_t)
    np.testing.assert_allclose(log_p.detach().numpy(), black76.log_otm_price(k, w), rtol=1e-10)
    assert np.all(np.exp(log_p.detach().numpy()[-2:]) == 0.0)
    log_p.sum().backward()
    assert torch.isfinite(w_t.grad).all() and torch.all(w_t.grad > 0)


def test_mixture_vol_lies_between_component_vols():
    rng = np.random.default_rng(5)
    k = rng.uniform(-1.0, 0.5, 1000)
    comps = rng.uniform(0.001, 0.2, (3, 1000))
    w_mix = mixture_total_variance(k, comps, np.array([0.5, 0.3, 0.2]))
    assert np.all(w_mix >= comps.min(axis=0) * (1 - 1e-10))
    assert np.all(w_mix <= comps.max(axis=0) * (1 + 1e-10))


@pytest.fixture(scope="module")
def ssvi_days():
    quotes, _ = generate(SyntheticConfig(surface="ssvi", n_days=2, seed=7))
    return [make_day(g) for _, g in quotes.groupby("date")]


def test_essvi_recovers_an_ssvi_surface(ssvi_days):
    day = ssvi_days[0]
    model = ESSVISurface().fit(day.fit_data())
    truth = metrics.quote_metrics(day, model.iv(day.k, day.T), day.interp, "x")["x/truth_rmse_vp"]
    assert truth < 100 * np.median(day.s)


@pytest.mark.parametrize("generator", ["free", "mlp"])
def test_neural_essvi_fits_and_warm_starts_without_arbitrage(ssvi_days, generator):
    cfg = NeuralESSVIConfig(components=2, generator=generator, pretrain_steps=100,
                            pretrain_lbfgs_iters=50, steps=20, lbfgs_iters=60,
                            warm_lbfgs_iters=30)  # fmt: skip
    first = NeuralESSVISurface(cfg).fit(ssvi_days[0].fit_data())
    second = NeuralESSVISurface(cfg).fit(ssvi_days[1].fit_data(), init=first)
    for day, model in zip(ssvi_days, (first, second), strict=True):
        assert np.isfinite(model.final_loss) and model.n_failed == 0
        rms = metrics.quote_metrics(day, model.iv(day.k, day.T), day.interp, "x")["x/rms_norm"]
        assert rms < 2.0
        bounds = day.slice_bounds()
        for widen in (0.0, 0.5):
            arb = metrics.arbitrage_metrics(model, bounds, ARB, widen, "a")
            assert arb["a/calendar_frac"] == 0.0 and arb["a/butterfly_frac"] == 0.0
