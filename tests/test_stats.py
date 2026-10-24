"""Tests of the statistical tools on synthetic data with known answers."""

import numpy as np
import pytest
from scipy import special

from bksoc import stats


@pytest.mark.parametrize("tau", [1.3, 2.0, 2.7])
def test_continuous_mle_recovers_exponent(tau):
    x = stats.sample_power_law(100_000, tau, xmin=2.0, rng=1)
    fit = stats.fit_power_law(x, 2.0)
    assert abs(fit["tau"] - tau) < 4 * fit["err"]


def test_mle_error_bar_is_calibrated():
    """About 68% of independent estimates fall within one quoted error of the true value."""
    rng = np.random.default_rng(2)
    tau, hits, n_rep = 1.8, 0, 400
    for _ in range(n_rep):
        fit = stats.fit_power_law(stats.sample_power_law(500, tau, rng=rng), 1.0)
        hits += abs(fit["tau"] - tau) < fit["err"]
    assert abs(hits / n_rep - 0.683) < 0.07


def test_discrete_sampler_matches_exact_probabilities():
    tau, xmin = 2.2, 3
    x = stats.sample_power_law(400_000, tau, xmin=xmin, discrete=True, rng=3)
    Z = special.zeta(tau, xmin)
    for k in (3, 4, 7, 20):
        p = k ** -tau / Z
        assert abs(np.mean(x == k) - p) < 5 * np.sqrt(p / x.size)


@pytest.mark.parametrize("xmin", [1, 5])
def test_discrete_mle_recovers_exponent(xmin):
    x = stats.sample_power_law(100_000, 1.7, xmin=xmin, discrete=True, rng=4)
    fit = stats.fit_power_law(x, xmin)
    assert abs(fit["tau"] - 1.7) < 4 * fit["err"]


def test_continuous_mle_is_biased_on_integer_data():
    """Treating integers as continuous gives a wrong exponent: the reason a discrete fit is needed."""
    x = stats.sample_power_law(100_000, 1.7, xmin=1, discrete=True, rng=5)
    naive = 1 + x.size / np.log(x / 1.0).sum()
    assert abs(naive - 1.7) > 0.1


@pytest.mark.parametrize("discrete", [False, True])
def test_truncated_mle_recovers_exponent(discrete):
    x = stats.sample_power_law(100_000, 1.25, xmin=1, xmax=10**4, discrete=discrete, rng=6)
    fit = stats.fit_power_law(x, 1, 10**4, discrete=discrete)
    assert abs(fit["tau"] - 1.25) < 4 * fit["err"]


def test_untruncated_fit_of_truncated_data_is_biased():
    x = stats.sample_power_law(100_000, 1.25, xmin=1, xmax=1e3, rng=7)
    assert stats.fit_power_law(x, 1.0)["tau"] > 1.35


def test_scan_xmin_finds_onset_of_power_law():
    """Exponential body below 10, power law tail above it."""
    rng = np.random.default_rng(8)
    body = 1 + rng.exponential(2.0, 30_000)
    body = body[body < 10]
    tail = stats.sample_power_law(20_000, 2.5, xmin=10, rng=rng)
    res = stats.scan_xmin(np.concatenate([body, tail]), np.logspace(0, 2, 41))
    assert 7 <= res["xmin"][res["best"]] <= 16
    assert abs(res["tau"][res["best"]] - 2.5) < 0.1


def test_gof_accepts_power_law_and_rejects_exponential():
    rng = np.random.default_rng(9)
    x = stats.sample_power_law(2000, 2.0, xmin=1, rng=rng)
    fit = stats.fit_power_law(x, 1.0)
    assert stats.gof_pvalue(x, fit["tau"], 1.0, n_boot=100, rng=rng) > 0.1
    y = 1 + rng.exponential(3.0, 2000)
    fy = stats.fit_power_law(y, 1.0)
    assert stats.gof_pvalue(y, fy["tau"], 1.0, n_boot=100, rng=rng) < 0.05


def test_blocking_error_for_ar1_process():
    """AR(1) with coefficient phi: Var(mean) = sigma_x^2 (1+phi) / ((1-phi) n)."""
    rng = np.random.default_rng(10)
    phi, n = 0.9, 400_000
    eps = rng.normal(size=n)
    x = np.empty(n)
    x[0] = eps[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    exact = np.sqrt((1 / (1 - phi**2)) * (1 + phi) / (1 - phi) / n)
    sizes, errs = stats.blocking(x)
    plateau = errs[(sizes >= 256) & (sizes <= 4096)].mean()
    assert abs(plateau / exact - 1) < 0.15
    assert errs[0] < 0.5 * exact          # the naive error is far too small
    assert abs(stats.batch_means_error(x, 50) / exact - 1) < 0.3


def _synthetic_fss(tau, D, Ls, n, rng):
    """P(s; L) = s^-tau exp(-s / L^D) on s >= 1 (continuous), sampled by thinning."""
    out = []
    for L in Ls:
        sc = L**D
        x = np.empty(0)
        while x.size < n:
            y = stats.sample_power_law(2 * n, tau, xmin=1.0, xmax=50 * sc, rng=rng)
            x = np.concatenate([x, y[rng.random(y.size) < np.exp(-y / sc)]])
        out.append(x[:n])
    return out


def _exact_moment(q, tau, sc):
    """<s^q> for the density proportional to s^-tau exp(-s/sc) on [1, 50 sc]."""
    from scipy import integrate
    f = lambda p: integrate.quad(lambda s: s ** (p - tau) * np.exp(-s / sc), 1, 50 * sc,
                                 limit=500, points=[sc])[0]
    return f(q) / f(0)


def test_moment_exponents_on_synthetic_fss():
    """The estimator reproduces the exact finite-size slopes (which include corrections to scaling)."""
    rng = np.random.default_rng(11)
    tau, D, Ls = 1.3, 2.0, [16, 32, 64, 128]
    samples = _synthetic_fss(tau, D, Ls, 200_000, rng)
    qs = np.array([1.0, 1.5, 2.0, 3.0])
    sig, err = stats.moment_exponents(samples, Ls, qs, n_boot=50, rng=rng)
    exact = [np.polyfit(np.log(Ls), np.log([_exact_moment(q, tau, L**D) for L in Ls]), 1)[0] for q in qs]
    assert np.all(np.abs(sig - exact) < 4 * err)
    # the slope between consecutive q approaches D; the offset does not yet equal D(q+1-tau)
    assert abs((sig[-1] - sig[0]) / (qs[-1] - qs[0]) - D) < 0.05


def test_collapse_fit_on_synthetic_fss():
    rng = np.random.default_rng(12)
    tau, D, Ls = 1.6, 2.0, [32, 64, 128, 256]
    samples = _synthetic_fss(tau, D, Ls, 200_000, rng)
    t, d, cost = stats.fit_collapse(samples, Ls, 1.5, 2.2, xmin_scaled=1e-3)
    assert abs(t - tau) < 0.03 and abs(d - D) < 0.06
