import numpy as np

from volsurf.eval.forecast import diebold_mariano, fit_var1, var_forecast


def test_var1_recovers_simulated_dynamics_and_skips_non_consecutive_rows():
    rng = np.random.default_rng(0)
    A = np.array([[0.9, 0.05], [-0.1, 0.7]])
    c = np.array([0.1, -0.2])
    z = np.zeros((5000, 2))
    for t in range(1, len(z)):
        z[t] = c + A @ z[t - 1] + 0.1 * rng.standard_normal(2)
    consecutive = np.ones(len(z), dtype=bool)
    z[2500] += 50.0  # a jump across a break in the sample must not enter the fit
    consecutive[2499] = consecutive[2500] = False
    c_hat, A_hat = fit_var1(z, consecutive)
    np.testing.assert_allclose(A_hat, A, atol=0.03)
    np.testing.assert_allclose(c_hat, c, atol=0.03)
    two = var_forecast(c, A, z[:3], 2)
    np.testing.assert_allclose(two, c + (c + z[:3] @ A.T) @ A.T)


def test_diebold_mariano_size_and_power_under_autocorrelation():
    rng = np.random.default_rng(1)
    rejections = 0
    for _ in range(400):
        e = rng.standard_normal(260)
        d = e + 0.8 * np.r_[0.0, e[:-1]]  # MA(1), as with overlapping two-step errors
        rejections += diebold_mariano(d, horizon=2)[1] < 0.05
    assert 0.02 < rejections / 400 < 0.09
    stat, p = diebold_mariano(rng.standard_normal(260) + 0.5, horizon=1)
    assert stat > 0 and p < 1e-4
