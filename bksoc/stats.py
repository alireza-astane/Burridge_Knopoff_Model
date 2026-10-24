"""Statistical tools for avalanche-size data: binning, power-law fits, errors, scaling.

All fits follow the maximum-likelihood approach of Clauset, Shalizi & Newman (2009),
extended to a window [xmin, xmax] (truncated power law) where needed.
"""

import numpy as np
from scipy import optimize, special


# --------------------------------------------------------------------- histograms

def log_binned_density(x, bins_per_decade=10, xmin=None, discrete=None):
    """Probability density from logarithmically spaced bins.

    Returns (centres, density, error). The density is normalised by the total number of
    samples (including those below xmin), so it can be compared across data sets. The
    error is the Poisson error sqrt(count) propagated to the density. For integer data,
    bin edges are integers and each bin [e_k, e_{k+1}) contains e_{k+1} - e_k integers.
    """
    x = np.asarray(x)
    n_total = x.size
    if discrete is None:
        discrete = np.issubdtype(x.dtype, np.integer)
    if xmin is None:
        xmin = x[x > 0].min()
    x = x[x >= xmin]
    xmax = x.max()
    n_bins = max(1, int(np.ceil(np.log10(xmax / xmin) * bins_per_decade)))
    edges = np.logspace(np.log10(xmin), np.log10(xmax), n_bins + 1)
    if discrete:
        edges = np.unique(np.round(edges).astype(np.int64))
        edges = np.append(edges[:-1], xmax + 1)
        edges = np.unique(edges)
        counts, _ = np.histogram(x, bins=edges)
        width = np.diff(edges).astype(float)
        centres = np.sqrt(edges[:-1] * (edges[1:] - 1.0))
    else:
        edges[-1] *= 1 + 1e-12
        counts, _ = np.histogram(x, bins=edges)
        width = np.diff(edges)
        centres = np.sqrt(edges[:-1] * edges[1:])
    dens = counts / (n_total * width)
    err = np.sqrt(counts) / (n_total * width)
    keep = counts > 0
    return centres[keep], dens[keep], err[keep]


def linear_binned_density(x, n_bins=100):
    x = np.asarray(x, dtype=float)
    counts, edges = np.histogram(x, bins=n_bins)
    width = np.diff(edges)
    keep = counts > 0
    return 0.5 * (edges[:-1] + edges[1:])[keep], (counts / (x.size * width))[keep]


def ccdf(x):
    """Empirical complementary cumulative distribution: returns (sorted x, P(X >= x))."""
    xs = np.sort(np.asarray(x))
    n = xs.size
    return xs, 1.0 - np.arange(n) / n


# --------------------------------------------------------------------- sampling

def sample_power_law(n, tau, xmin=1.0, xmax=np.inf, rng=None, discrete=False):
    """Draw n samples from P(x) ~ x^-tau on [xmin, xmax] by inverting the cumulative distribution.

    For discrete=True, samples follow the discrete law P(k) ~ k^-tau for integers
    xmin <= k <= xmax exactly (inverse transform on the exact discrete CDF, using the
    Hurwitz zeta function for the normalisation and bisection for the inversion).
    """
    rng = np.random.default_rng(rng)
    u = rng.random(n)
    if not discrete:
        a = xmin ** (1 - tau)
        b = 0.0 if np.isinf(xmax) else xmax ** (1 - tau)
        return (a - u * (a - b)) ** (1 / (1 - tau))
    xmin = int(xmin)
    upper = special.zeta(tau, xmin)
    tail_end = 0.0 if np.isinf(xmax) else special.zeta(tau, int(xmax) + 1)
    Z = upper - tail_end
    # P(X >= k) = (zeta(tau, k) - tail_end) / Z. X is the largest k with P(X >= k) > u,
    # i.e. with zeta(tau, k) > target; find it by bisection between lo (zeta > target)
    # and hi (zeta <= target).
    target = u * Z + tail_end
    lo = np.full(n, xmin, dtype=np.int64)
    if np.isinf(xmax):
        hi = np.full(n, 2 * xmin, dtype=np.int64)
        while True:
            bad = special.zeta(tau, hi) > target
            if not bad.any():
                break
            hi[bad] *= 2
    else:
        hi = np.full(n, int(xmax) + 1, dtype=np.int64)
    while np.any(hi - lo > 1):
        mid = (lo + hi) // 2
        go_right = special.zeta(tau, mid) > target
        lo = np.where(go_right, mid, lo)
        hi = np.where(go_right, hi, mid)
    return lo


# --------------------------------------------------------------------- maximum likelihood

def _loglike(tau, logsum, n, xmin, xmax, discrete):
    """Log-likelihood of n samples with sum(log x) = logsum under a power law on [xmin, xmax]."""
    if discrete:
        Z = special.zeta(tau, xmin)
        if np.isfinite(xmax):
            Z -= special.zeta(tau, xmax + 1)
        return -n * np.log(Z) - tau * logsum
    if np.isfinite(xmax):
        Z = (xmin ** (1 - tau) - xmax ** (1 - tau)) / (tau - 1)
    else:
        Z = xmin ** (1 - tau) / (tau - 1)
    return -n * np.log(Z) - tau * logsum


def fit_power_law(x, xmin, xmax=np.inf, discrete=None):
    """Maximum-likelihood exponent of a power law fitted to the data in [xmin, xmax].

    Returns dict(tau, err, n). For a continuous, untruncated power law the estimate is the
    closed form tau = 1 + n / sum(ln(x/xmin)) with error (tau-1)/sqrt(n). Otherwise the
    likelihood is maximised numerically and the error comes from its curvature.
    """
    x = np.asarray(x)
    if discrete is None:
        discrete = np.issubdtype(x.dtype, np.integer)
    sel = x[(x >= xmin) & (x <= xmax)].astype(float)
    n = sel.size
    if n < 2:
        return dict(tau=np.nan, err=np.nan, n=n)
    if not discrete and np.isinf(xmax):
        tau = 1 + n / np.log(sel / xmin).sum()
        return dict(tau=tau, err=(tau - 1) / np.sqrt(n), n=n)
    logsum = np.log(sel).sum()
    f = lambda t: -_loglike(t, logsum, n, xmin, xmax, discrete)
    lo_bound = 1.0001 if (discrete or np.isinf(xmax)) else -3.0
    res = optimize.minimize_scalar(f, bounds=(lo_bound, 6.0), method="bounded",
                                   options=dict(xatol=1e-7))
    tau = res.x
    h = 1e-4
    curv = (f(tau + h) - 2 * f(tau) + f(tau - h)) / h**2
    return dict(tau=tau, err=1 / np.sqrt(curv) if curv > 0 else np.nan, n=n)


def power_law_cdf(x, tau, xmin, xmax=np.inf, discrete=False):
    """Model probability P(X <= x) for the (possibly truncated) power law on [xmin, xmax]."""
    x = np.asarray(x, dtype=float)
    if discrete:
        Z = special.zeta(tau, xmin) - (special.zeta(tau, xmax + 1) if np.isfinite(xmax) else 0.0)
        return (special.zeta(tau, xmin) - special.zeta(tau, np.floor(x) + 1)) / Z
    a = xmin ** (1 - tau)
    b = 0.0 if np.isinf(xmax) else xmax ** (1 - tau)
    return (a - x ** (1 - tau)) / (a - b)


def ks_distance(x, tau, xmin, xmax=np.inf, discrete=None):
    """Kolmogorov-Smirnov distance between the data in [xmin, xmax] and the fitted power law."""
    x = np.asarray(x)
    if discrete is None:
        discrete = np.issubdtype(x.dtype, np.integer)
    sel = np.sort(x[(x >= xmin) & (x <= xmax)])
    n = sel.size
    if discrete:
        vals, counts = np.unique(sel, return_counts=True)
        emp = np.cumsum(counts) / n
        model = power_law_cdf(vals, tau, xmin, xmax, True)
        return np.max(np.abs(emp - model))
    model = power_law_cdf(sel, tau, xmin, xmax, False)
    emp_hi = np.arange(1, n + 1) / n
    emp_lo = np.arange(0, n) / n
    return max(np.max(np.abs(emp_hi - model)), np.max(np.abs(emp_lo - model)))


def scan_xmin(x, candidates, xmax=np.inf, discrete=None, min_tail=50):
    """Fit tau for each candidate xmin and return the one minimising the KS distance.

    Returns dict(best=index, xmin, tau, err, ks) with arrays over the candidates.
    """
    x = np.asarray(x)
    taus, errs, kss, ns = [], [], [], []
    for xm in candidates:
        fit = fit_power_law(x, xm, xmax, discrete)
        ns.append(fit["n"])
        taus.append(fit["tau"])
        errs.append(fit["err"])
        kss.append(ks_distance(x, fit["tau"], xm, xmax, discrete) if fit["n"] >= min_tail else np.inf)
    kss = np.array(kss)
    best = int(np.argmin(kss))
    return dict(best=best, xmin=np.asarray(candidates), tau=np.array(taus), err=np.array(errs),
                ks=kss, n=np.array(ns))


def gof_pvalue(x, tau, xmin, xmax=np.inf, discrete=None, n_boot=200, rng=None):
    """Parametric-bootstrap p-value for the power-law hypothesis in [xmin, xmax].

    Synthetic samples of the same size are drawn from the fitted law, refitted, and their
    KS distances compared with the observed one. A small p-value (< 0.1) rejects the power law.
    """
    x = np.asarray(x)
    if discrete is None:
        discrete = np.issubdtype(x.dtype, np.integer)
    rng = np.random.default_rng(rng)
    n = int(((x >= xmin) & (x <= xmax)).sum())
    d_obs = ks_distance(x, tau, xmin, xmax, discrete)
    worse = 0
    for _ in range(n_boot):
        y = sample_power_law(n, tau, xmin, xmax, rng, discrete)
        fy = fit_power_law(y, xmin, xmax, discrete)
        if ks_distance(y, fy["tau"], xmin, xmax, discrete) >= d_obs:
            worse += 1
    return worse / n_boot


# --------------------------------------------------------------------- correlated data

def blocking(x, min_blocks=16):
    """Standard error of the mean for increasing block sizes (Flyvbjerg-Petersen blocking).

    Returns (block sizes, standard errors). For correlated data the error rises with the
    block size and levels off once blocks are longer than the correlation time; the
    plateau is the correct error.
    """
    x = np.asarray(x, dtype=float)
    sizes, errs = [], []
    b = 1
    while x.size // b >= min_blocks:
        nb = x.size // b
        means = x[: nb * b].reshape(nb, b).mean(axis=1)
        sizes.append(b)
        errs.append(means.std(ddof=1) / np.sqrt(nb))
        b *= 2
    return np.array(sizes), np.array(errs)


def batch_means_error(x, n_batches=50):
    """Standard error of the mean from n_batches contiguous batches."""
    x = np.asarray(x, dtype=float)
    b = x.size // n_batches
    means = x[: b * n_batches].reshape(n_batches, b).mean(axis=1)
    return means.std(ddof=1) / np.sqrt(n_batches)


def autocorrelation(x, max_lag):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    var = x.var()
    return np.array([1.0] + [np.mean(x[:-k] * x[k:]) / var for k in range(1, max_lag + 1)])


# --------------------------------------------------------------------- finite-size scaling

def moment_exponents(samples, Ls, qs, n_boot=100, n_blocks=50, rng=None, fit_last=None):
    """Moment exponents sigma(q) from <s^q> ~ L^sigma(q), with block-bootstrap errors.

    samples: list of 1D arrays (one per L, time-ordered). Errors are obtained by
    resampling contiguous blocks of each time series, which respects time correlations.
    fit_last: use only the largest `fit_last` sizes in the log-log fit.
    """
    rng = np.random.default_rng(rng)
    Ls = np.asarray(Ls, dtype=float)
    sl = slice(-fit_last, None) if fit_last else slice(None)
    qs = np.asarray(qs, dtype=float)

    def block_moments(s):
        b = s.size // n_blocks
        blocks = s[: b * n_blocks].astype(float).reshape(n_blocks, b)
        return np.stack([np.mean(blocks ** q, axis=1) for q in qs])   # (nq, n_blocks)

    bm = [block_moments(s) for s in samples]

    def sigmas(idx_list):
        logm = np.log(np.stack([m[:, idx].mean(axis=1) for m, idx in zip(bm, idx_list)], axis=1))
        return np.array([np.polyfit(np.log(Ls[sl]), lm[sl], 1)[0] for lm in logm])

    full = sigmas([np.arange(n_blocks)] * len(samples))
    boots = np.array([sigmas([rng.integers(0, n_blocks, n_blocks) for _ in samples])
                      for _ in range(n_boot)])
    return full, boots.std(axis=0, ddof=1)


def collapse_curves(samples, Ls, tau, D, bins_per_decade=8):
    """Rescaled curves (s / L^D, s^tau P(s), relative error) for a data-collapse plot."""
    out = []
    for s, L in zip(samples, Ls):
        c, p, e = log_binned_density(s[s > 0], bins_per_decade)
        frac = np.count_nonzero(s > 0) / s.size
        out.append((c / L**D, c**tau * p * frac, e / p))
    return out


def collapse_cost(samples, Ls, tau, D, bins_per_decade=8, xmin_scaled=None, grid_points=40):
    """Chi-square per degree of freedom of the spread between rescaled curves.

    On a common grid of log(s/L^D), the curves log(s^tau P) of the different L are compared
    with their error-weighted mean, each weighted by its statistical (Poisson) error. A
    collapse that is perfect within the statistical noise gives a value close to 1. Only the
    part of the curves above xmin_scaled (in units of s/L^D) is used, to exclude the
    lattice-scale region where no collapse is expected.
    """
    curves = collapse_curves(samples, Ls, tau, D, bins_per_decade)
    logs = [(np.log(x), np.log(y), r) for x, y, r in curves]
    lo = max(lx.min() for lx, _, _ in logs)
    if xmin_scaled is not None:
        lo = max(lo, np.log(xmin_scaled))
    hi = min(lx.max() for lx, _, _ in logs)
    if hi <= lo:
        return np.inf
    grid = np.linspace(lo, hi, grid_points)
    ys = np.array([np.interp(grid, lx, ly) for lx, ly, _ in logs])
    sig = np.array([np.interp(grid, lx, r) for lx, _, r in logs])
    w = 1 / sig**2
    mean = (w * ys).sum(axis=0) / w.sum(axis=0)
    chi2 = (w * (ys - mean) ** 2).sum(axis=0) / (len(samples) - 1)
    return float(chi2.mean())


def fit_collapse(samples, Ls, tau0, D0, **kw):
    """Find (tau, D) minimising collapse_cost, starting from (tau0, D0)."""
    f = lambda p: collapse_cost(samples, Ls, p[0], p[1], **kw)
    res = optimize.minimize(f, x0=[tau0, D0], method="Nelder-Mead",
                            options=dict(xatol=1e-4, fatol=1e-8))
    return res.x[0], res.x[1], res.fun
