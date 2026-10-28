import numpy as np
import pytest
import torch

from volsurf.arbitrage import conditions as arb
from volsurf.models.ssvi import gj_butterfly_free, gj_calendar_free, power_law_phi, ssvi
from volsurf.models.svi import raw_svi
from volsurf.pricing import heston

K_GRID = np.linspace(-1.5, 1.5, 601)
T_GRID = np.array([7, 14, 30, 60, 90, 180, 365, 730]) / 365.0
RHO, ETA, GAMMA = -0.7, 1.0, 0.4
HESTON = heston.HestonParams(kappa=2.0, theta=0.04, sigma=0.6, rho=-0.7, v0=0.03)


def theta_of_T(T):
    return heston.expected_total_variance(T, HESTON)


def ssvi_surface(k, T, theta_fn=theta_of_T, eta=ETA):
    theta = theta_fn(T)
    return ssvi(k, theta, RHO, power_law_phi(theta, eta, GAMMA))


def theta_torch(T):
    p = HESTON
    return p.theta * T + (p.v0 - p.theta) * (1.0 - torch.exp(-p.kappa * T)) / p.kappa


def ssvi_torch(k, T):
    theta = theta_torch(T)
    return ssvi(k, theta, RHO, power_law_phi(theta, ETA, GAMMA))


def test_gj_parameters_satisfy_sufficient_conditions():
    theta = theta_of_T(T_GRID)
    phi = power_law_phi(theta, ETA, GAMMA)
    assert np.all(gj_butterfly_free(theta, RHO, phi))
    assert gj_calendar_free(theta, phi, RHO)


def test_gj_ssvi_is_arbitrage_free_on_grid():
    TT, KK = np.meshgrid(T_GRID, K_GRID, indexing="ij")
    report = arb.check_grid(ssvi_surface(KK, TT), K_GRID, T_GRID)
    assert report.arbitrage_free, report
    assert report.butterfly_min_g > 0.0


def test_gj_ssvi_is_arbitrage_free_with_autograd():
    report = arb.check_torch(ssvi_torch, K_GRID, T_GRID)
    assert report.arbitrage_free, report


def test_autograd_g_matches_finite_differences():
    T = 0.5
    k = np.linspace(-1.5, 1.5, 6001)
    w = ssvi_surface(k, T)
    ki, wi, w_k, w_kk = arb.fd_slice_derivatives(w, k)
    g_fd = arb.durrleman_g(ki, wi, w_k, w_kk)

    k_t = torch.as_tensor(ki, dtype=torch.float64)
    T_t = torch.full_like(k_t, T)
    w_t, wk_t, wkk_t, _ = arb.surface_derivatives(ssvi_torch, k_t, T_t)
    g_ad = arb.durrleman_g(k_t, w_t, wk_t, wkk_t).numpy()
    np.testing.assert_allclose(g_fd, g_ad, atol=1e-5)


def test_vogt_svi_has_butterfly_arbitrage():
    # Gatheral & Jacquier (2014), Example 3.1.
    params = dict(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
    w = raw_svi(K_GRID, **params)[None, :]
    report = arb.check_grid(w, K_GRID, np.array([1.0]))
    assert report.butterfly_frac > 0.0
    assert report.butterfly_min_g < 0.0

    report_ad = arb.check_torch(lambda k, T: raw_svi(k, **params), K_GRID, [1.0])
    assert report_ad.butterfly_min_g < 0.0


def test_decreasing_atm_variance_is_calendar_arbitrage():
    def inverted(T):  # ATM total variance falls between 30 and 90 days
        return np.interp(T, [0.0, 30 / 365, 90 / 365, 3.0], [0.0, 0.01, 0.006, 0.12])

    TT, KK = np.meshgrid(T_GRID, K_GRID, indexing="ij")
    report = arb.check_grid(ssvi_surface(KK, TT, inverted), K_GRID, T_GRID)
    assert report.calendar_frac > 0.0
    assert report.calendar_max > 0.0


def test_steep_wings_violate_lee_bound():
    k = np.linspace(-20.0, 20.0, 801)
    theta = 0.2
    phi = 30.0  # theta * phi * (1 + |rho|) = 10.2 > 4
    assert not gj_butterfly_free(theta, RHO, phi)
    report = arb.check_grid(ssvi(k, theta, RHO, phi)[None, :], k, np.array([1.0]))
    assert report.wing_max_ratio > 2.0
    assert not report.arbitrage_free


def test_heston_surface_is_arbitrage_free_on_grid():
    F = 4000.0
    k = np.linspace(-0.3, 0.15, 181)  # within ~4 ATM std devs of the shortest slice
    T = np.array([30, 60, 90, 180, 365, 730]) / 365.0
    iv = np.stack([heston.implied_vol_slice(F, F * np.exp(k), t, HESTON) for t in T])
    assert np.all(np.isfinite(iv))
    report = arb.check_grid(iv**2 * T[:, None], k, T)
    assert report.calendar_frac == 0.0
    assert report.butterfly_frac == 0.0


@pytest.mark.parametrize("create_graph", [False, True])
def test_surface_derivatives_graph_mode(create_graph):
    a = torch.tensor(0.04, dtype=torch.float64, requires_grad=True)
    k = torch.linspace(-0.5, 0.5, 11, dtype=torch.float64)
    T = torch.full_like(k, 0.5)
    w, w_k, w_kk, w_T = arb.surface_derivatives(lambda k, T: a * T + 0.1 * k**2, k, T, create_graph)
    torch.testing.assert_close(w_kk, torch.full_like(k, 0.2))
    torch.testing.assert_close(w_T, torch.full_like(k, 0.04))
    assert w_T.requires_grad == create_graph
