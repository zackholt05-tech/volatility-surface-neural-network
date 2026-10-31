import numpy as np
import pytest
import torch

from volsurf.arbitrage.conditions import check_grid
from volsurf.data.synthetic import SyntheticConfig, generate
from volsurf.eval.dayset import make_day
from volsurf.models.conditional import (
    ConditionalConfig,
    ConditionalNet,
    ConditionalSurface,
    QuoteBatch,
    batch_iv,
    grid_knots,
    interp_coefficients,
)
from volsurf.models.essvi import interp_knots, knots_from_rates


def random_net(seed: int, latent_dim: int = 4) -> ConditionalNet:
    torch.manual_seed(seed)
    net = ConditionalNet(ConditionalConfig(latent_dim=latent_dim, enc_hidden=32, dec_hidden=(32,)))
    with torch.no_grad():  # spread the decoded rates over much of their bounded range
        net.decoder.out.weight.mul_(20.0)
    return net


def surface_from_latent(net: ConditionalNet, z: torch.Tensor) -> ConditionalSurface:
    surface = ConditionalSurface("unused")
    with torch.no_grad():
        rates, logits = net.decoder(z)
        th, up, um = grid_knots(rates, net.grid)
    surface.knot_T = net.grid.numpy()
    surface.knots = [(th[j].numpy(), up[j].numpy(), um[j].numpy()) for j in range(th.shape[0])]
    surface.weights = torch.softmax(logits, -1).numpy()
    return surface


def test_batched_grid_knots_match_the_per_surface_map():
    rng = np.random.default_rng(0)
    grid = np.array([5, 9, 14, 30, 60, 120, 365, 900]) / 365.0
    rates = rng.normal(0.0, 4.0, (2, 3, len(grid), 3))
    batched = grid_knots(torch.as_tensor(rates), torch.as_tensor(grid))
    for i in range(2):
        for j in range(3):
            single = knots_from_rates(grid, rates[i, j])
            for a, b in zip(single, batched, strict=True):
                np.testing.assert_allclose(b[i, j].numpy(), a, rtol=1e-12)


def test_interp_coefficients_reproduce_interp_knots_inside_and_outside_the_grid():
    rng = np.random.default_rng(1)
    grid = np.array([5, 9, 14, 30, 60, 120, 365, 900]) / 365.0
    theta, up, um = knots_from_rates(grid, rng.normal(0.0, 2.0, (len(grid), 3)))
    T = np.r_[rng.uniform(1 / 365, 4.0, 500), grid]
    lo, hi, (a_th, b_th, a_u, b_u) = interp_coefficients(T, grid)
    expected = interp_knots(T, grid, theta, up, um)
    got = (
        a_th * theta[lo] + b_th * theta[hi],
        a_u * up[lo] + b_u * up[hi],
        a_u * um[lo] + b_u * um[hi],
    )
    for a, b in zip(expected, got, strict=True):
        np.testing.assert_allclose(b, a, rtol=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_any_latent_decodes_to_a_surface_free_of_static_arbitrage(seed):
    net = random_net(seed)
    z = torch.as_tensor(np.random.default_rng(seed).normal(0.0, 3.0, 4))
    surface = surface_from_latent(net, z)
    T = np.union1d(np.geomspace(2 / 365, 3.0, 60), surface.knot_T)
    k = np.linspace(-1.5, 1.0, 301)
    w = surface.w(k[None, :], T[:, None])
    assert np.isfinite(w).all()
    report = check_grid(w, k, T)
    assert report.calendar_frac == 0.0 and report.butterfly_frac == 0.0, report


def quote_batch(net, n_days=2, n=300, seed=2):
    rng = np.random.default_rng(seed)
    T = rng.choice(np.array([7, 12, 30, 45, 90, 200, 400, 700]) / 365.0, n_days * n)
    k = rng.uniform(-0.6, 0.3, n_days * n) * np.sqrt(T / 0.25)
    iv = rng.uniform(0.1, 0.4, n_days * n)
    s = np.full(n_days * n, 0.005)
    day = np.repeat(np.arange(n_days), n)
    return QuoteBatch.from_arrays(k, T, iv, s, day, n_days, net.grid.numpy())


def test_training_implied_vols_match_the_evaluated_surface():
    net = random_net(3)
    q = quote_batch(net)
    z = torch.as_tensor(np.random.default_rng(3).normal(0.0, 1.0, (q.n_days, 4)))
    with torch.no_grad():
        iv = batch_iv(*net.decoder(z), net.grid, q).numpy()
    for d in range(q.n_days):
        m = q.day.numpy() == d
        expected = surface_from_latent(net, z[d]).iv(q.k.numpy()[m], q.T.numpy()[m])
        np.testing.assert_allclose(iv[m], expected, rtol=1e-8)


def test_encoder_is_permutation_invariant_and_separates_days():
    net = random_net(4)
    q = quote_batch(net)
    with torch.no_grad():
        z = net.encode(q)
        perm = torch.randperm(len(q.k))
        shuffled = QuoteBatch(q.k[perm], q.T[perm], q.iv[perm], q.s[perm], q.day[perm],
                              q.lo[perm], q.hi[perm], q.coef[:, perm], q.n_days)  # fmt: skip
        np.testing.assert_allclose(net.encode(shuffled).numpy(), z.numpy(), rtol=1e-10)
        m = q.day == 1
        alone = QuoteBatch(q.k[m], q.T[m], q.iv[m], q.s[m], q.day[m] * 0, q.lo[m], q.hi[m],
                           q.coef[:, m], 1)  # fmt: skip
        np.testing.assert_allclose(net.encode(alone).numpy()[0], z[1].numpy(), rtol=1e-10)


def test_refined_latent_lowers_the_fit_loss_and_keeps_the_guarantee(tmp_path):
    quotes, _ = generate(SyntheticConfig(surface="ssvi", n_days=1, seed=11))
    day = make_day(quotes)
    path = tmp_path / "net.pt"
    random_net(5).save(path)
    amortized = ConditionalSurface(str(path)).fit(day.fit_data())
    refined = ConditionalSurface(str(path), refine_iters=30).fit(day.fit_data())
    assert np.isfinite(refined.final_loss) and refined.final_loss < amortized.final_loss
    T = np.union1d(np.geomspace(2 / 365, 3.0, 40), day.slice_T)
    k = np.linspace(-1.5, 1.0, 201)
    report = check_grid(refined.w(k[None, :], T[:, None]), k, T)
    assert report.calendar_frac == 0.0 and report.butterfly_frac == 0.0, report


def test_maturity_gaps_hold_out_whole_inner_slices_without_moving_other_holdouts():
    from volsurf.eval.dayset import HoldoutConfig

    quotes, _ = generate(SyntheticConfig(surface="ssvi", n_days=1, seed=12))
    base = make_day(quotes, HoldoutConfig())
    gaps = make_day(quotes, HoldoutConfig(slice_frac=0.3))
    np.testing.assert_array_equal(gaps.wing, base.wing)
    assert np.all(gaps.interp[base.interp])  # the random interior holdout is unchanged
    observed = np.unique(gaps.slice_id[gaps.fit])
    held = np.setdiff1d(np.unique(gaps.slice_id), observed)
    n_inner = len(gaps.slice_T) - 2
    assert len(held) == round(0.3 * n_inner) and 0 not in held and n_inner + 1 not in held

    # models are fitted on the observed maturities only; without gaps every maturity is observed
    data = gaps.fit_data()
    np.testing.assert_array_equal(data.slice_T, gaps.slice_T[observed])
    np.testing.assert_array_equal(data.slice_T[data.slice_id], data.T)
    np.testing.assert_array_equal(base.fit_data().slice_T, base.slice_T)
    np.testing.assert_array_equal(base.fit_data().slice_id, base.slice_id[base.fit])
