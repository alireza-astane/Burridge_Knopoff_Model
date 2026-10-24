"""Figures and numbers for the thesis chapter on measuring exponents.

Uses synthetic data (known answers) and the BTW data produced by soc_figures.py.

    python scripts/thesis/stats_figures.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bksoc import stats
from bksoc import plotstyle as ps

ROOT = Path(__file__).resolve().parents[2]
SOC = ROOT / "data" / "soc"
FIG = ROOT / "thesis" / "figures" / "stats"
L2D = [32, 64, 128, 256, 512]

numbers = {}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def style_ax(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ------------------------------------------------------------------ 1. binning

def fig_binning():
    tau = 2.5
    x = stats.sample_power_law(1_000_000, tau, xmin=1.0, rng=1)
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.4), sharex=True)
    edges = np.arange(1, x.max() + 2)
    counts, _ = np.histogram(x, bins=edges)
    keep = counts > 0
    axes[0].loglog(edges[:-1][keep] + 0.5, counts[keep] / x.size, ".", ms=2, color=ps.C[0])
    axes[0].set_title("linear bins (width 1)")
    c, p, e = stats.log_binned_density(x, 5)
    axes[1].errorbar(c, p, yerr=e, fmt="o", ms=3, color=ps.C[0], elinewidth=0.8, capsize=0)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("logarithmic bins")
    xs, F = stats.ccdf(x)
    axes[2].loglog(xs, F, "-", color=ps.C[0], lw=1.2)
    axes[2].set_title(r"cumulative $\bar F(x)$")
    grid = np.logspace(0, np.log10(x.max()), 50)
    for ax in axes[:2]:
        ax.loglog(grid, (tau - 1) * grid**-tau, color=ps.C[1], lw=1.2, label=r"exact $1.5\,x^{-5/2}$")
    axes[2].loglog(grid, grid ** (1 - tau), color=ps.C[1], lw=1.2, label=r"exact $x^{-3/2}$")
    axes[0].axhline(1 / x.size, color=ps.INK_MUTED, lw=0.8, ls=":")
    axes[0].text(1.2, 3e-7, "one event per bin", fontsize=7, color=ps.INK_2, va="top")
    axes[1].legend(loc="lower left")
    axes[2].legend(loc="lower left")
    for ax in axes:
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$P(x)$")
    fig.tight_layout()
    save(fig, "binning")


# ------------------------------------------------------------------ 2. least squares vs MLE

def estimators(x, xmin=1.0):
    out = {}
    counts, edges = np.histogram(x, bins=np.arange(xmin, x.max() + 2))
    keep = counts > 0
    out["LS, linear bins"] = -np.polyfit(np.log(edges[:-1][keep] + 0.5), np.log(counts[keep]), 1)[0]
    c, p, _ = stats.log_binned_density(x, 5, xmin=xmin)
    out["LS, log bins"] = -np.polyfit(np.log(c), np.log(p), 1)[0]
    xs, F = stats.ccdf(x[x >= xmin])
    out["LS, cumulative"] = 1 - np.polyfit(np.log(xs), np.log(F), 1)[0]
    out["maximum likelihood"] = stats.fit_power_law(x, xmin)["tau"]
    return out


def fig_estimators():
    rng = np.random.default_rng(2)
    tau, n, reps = 2.0, 2000, 1000
    res = {}
    for _ in range(reps):
        for k, v in estimators(stats.sample_power_law(n, tau, rng=rng)).items():
            res.setdefault(k, []).append(v)
    fig, axes = plt.subplots(4, 1, figsize=(6.2, 4.6), sharex=True)
    bins = np.linspace(0.3, 2.4, 211)
    for ax, (k, v) in zip(axes, res.items()):
        v = np.array(v)
        ax.hist(v, bins=bins, color=ps.C[0], alpha=0.9)
        ax.axvline(tau, color=ps.C[1], lw=1.2)
        ax.text(0.01, 0.75, f"{k}:  mean {v.mean():.3f},  std {v.std():.3f}",
                transform=ax.transAxes, fontsize=8, color=ps.INK)
        ax.set_yticks([])
        key = {"LS, linear bins": "LSlin", "LS, log bins": "LSlog", "LS, cumulative": "LScum",
               "maximum likelihood": "MLE"}[k]
        numbers[f"est{key}Mean"] = f"{v.mean():.3f}"
        numbers[f"est{key}Std"] = f"{v.std():.3f}"
    axes[-1].set_xlabel(r"estimated exponent $\hat\tau$ (true value $\tau = 2$)")
    fig.tight_layout(h_pad=0.3)
    save(fig, "estimators")


# ------------------------------------------------------------------ 3. choosing xmin

def fig_xmin():
    rng = np.random.default_rng(8)
    body = 1 + rng.exponential(2.0, 30_000)
    body = body[body < 10]
    tail = stats.sample_power_law(20_000, 2.5, xmin=10, rng=rng)
    x = np.concatenate([body, tail])
    cands = np.logspace(0, 2, 41)
    res = stats.scan_xmin(x, cands)
    b = res["best"]
    numbers["xminBest"] = f"{res['xmin'][b]:.1f}"
    numbers["xminTau"] = f"{res['tau'][b]:.3f}"
    numbers["xminErr"] = f"{res['err'][b]:.3f}"
    numbers["xminTauOne"] = f"{res['tau'][0]:.2f}"
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.4))
    xs, F = stats.ccdf(x)
    axes[0].loglog(xs, F, color=ps.C[0], lw=1.2, label="data")
    xm = res["xmin"][b]
    frac = np.mean(x >= xm)
    g = np.logspace(np.log10(xm), np.log10(x.max()), 30)
    axes[0].loglog(g, frac * (g / xm) ** (1 - res["tau"][b]), "--", color=ps.C[1], lw=1.2, label="fitted tail")
    axes[0].axvline(xm, color=ps.INK_MUTED, lw=0.8, ls=":")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel(r"$\bar F(x)$")
    axes[0].legend(loc="lower left")
    axes[1].semilogx(cands, res["ks"], "o-", ms=2.5, color=ps.C[0])
    axes[1].axvline(xm, color=ps.INK_MUTED, lw=0.8, ls=":")
    axes[1].set_xlabel(r"$x_{\min}$")
    axes[1].set_ylabel("KS distance")
    axes[2].fill_between(cands, res["tau"] - res["err"], res["tau"] + res["err"], color=ps.BLUES[0], lw=0)
    axes[2].semilogx(cands, res["tau"], "-", color=ps.C[0], lw=1.2)
    axes[2].axhline(2.5, color=ps.C[1], lw=1.0)
    axes[2].axvline(xm, color=ps.INK_MUTED, lw=0.8, ls=":")
    axes[2].set_xlabel(r"$x_{\min}$")
    axes[2].set_ylabel(r"$\hat\tau(x_{\min})$")
    axes[2].set_ylim(1.4, 3.0)
    fig.tight_layout()
    save(fig, "xmin")


# ------------------------------------------------------------------ 4. effective exponents in BTW 2D

def load2d():
    return {L: np.load(SOC / f"btw2d_L{L}.npz")["size"] for L in L2D}


def fig_effective_exponent(d2):
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6))
    cols = ps.ordinal(len(L2D))
    for L, c in zip(L2D, cols):
        s = d2[L]
        starts = np.logspace(0, np.log10(s.max() / 10), 25).round().astype(int)
        starts = np.unique(starts)
        taus, errs = [], []
        for a in starts:
            f = stats.fit_power_law(s, a, 10 * a)
            taus.append(f["tau"])
            errs.append(f["err"])
        taus, errs = np.array(taus), np.array(errs)
        ok = errs < 0.05
        axes[0].plot(np.sqrt(10) * starts[ok], taus[ok], "-", color=c, lw=1.2, label=f"$L={L}$")
        # untruncated fits from xmin
        tu = [stats.fit_power_law(s, a)["tau"] for a in starts]
        axes[1].plot(starts, tu, "-", color=c, lw=1.2, label=f"$L={L}$")
    for ax in axes:
        ax.set_xscale("log")
    axes[0].set_xlabel("centre of the window $[s, 10s]$")
    axes[0].set_ylabel(r"$\tau_{\rm eff}$")
    axes[0].set_title("fit in one-decade windows")
    axes[0].set_ylim(0.9, 1.6)
    axes[1].set_xlabel(r"$s_{\min}$")
    axes[1].set_title(r"fit of all $s\geq s_{\min}$, no upper cutoff")
    axes[1].set_ylim(0.9, 1.6)
    axes[1].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    save(fig, "effective_exponent")
    # numbers: window fit at L=512 in [10^2, 10^4]
    f = stats.fit_power_law(d2[512], 100, 10_000)
    numbers["btwWinTau"] = f"{f['tau']:.3f}"
    numbers["btwWinErr"] = f"{f['err']:.3f}"
    p = stats.gof_pvalue(d2[512], f["tau"], 100, 10_000, n_boot=100, rng=3)
    numbers["btwWinP"] = f"{p:.2f}"


# ------------------------------------------------------------------ 5. time correlations

def fig_correlations(d2):
    s = d2[64].astype(float)
    acf = stats.autocorrelation(s, 2000)
    sizes, errs = stats.blocking(s)
    naive = s.std(ddof=1) / np.sqrt(s.size)
    numbers["corrRatio"] = f"{errs[-4:].mean() / naive:.1f}"
    numbers["corrAcfOne"] = f"{acf[1]:.3f}"
    from bksoc import open_lattice, theory
    g = theory.mean_topplings_per_site(open_lattice((64, 64)))
    numbers["corrSeed"] = f"{np.sqrt(g.var() / s.var()):.2f}"
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.4))
    lags = np.arange(acf.size)
    axes[0].semilogx(lags[1:], acf[1:], color=ps.C[0], lw=1.2)
    axes[0].axhline(0, color=ps.INK_MUTED, lw=0.6)
    axes[0].set_xlabel(r"lag $\Delta$ (number of avalanches)")
    axes[0].set_ylabel(r"autocorrelation $C(\Delta)$")
    axes[0].set_title("avalanche sizes, 2D BTW, $L=64$")
    axes[1].semilogx(sizes, errs / naive, "o-", ms=3, color=ps.C[0])
    axes[1].axhline(1, color=ps.INK_MUTED, lw=0.6, ls=":")
    axes[1].axhline(np.sqrt(g.var() / s.var()), color=ps.C[1], lw=1.0, ls="--",
                    label=r"$\sqrt{\mathrm{Var}\,g_j/\mathrm{Var}\,s}$")
    axes[1].legend(loc="upper right", fontsize=7)
    axes[1].set_xlabel("block length $b$")
    axes[1].set_ylabel("error of $\\langle s\\rangle$ / naive error")
    axes[1].set_title("blocking analysis")
    fig.tight_layout()
    save(fig, "correlations")


# ------------------------------------------------------------------ 6. collapse and corrections

def synthetic_fss(tau, D, Ls, n, rng):
    """Samples of P(s; L) = s^-tau exp(-s / L^D), s >= 1, by thinning a power law."""
    out = []
    for L in Ls:
        sc = L**D
        x = np.empty(0)
        while x.size < n:
            y = stats.sample_power_law(2 * n, tau, xmin=1.0, xmax=50 * sc, rng=rng)
            x = np.concatenate([x, y[rng.random(y.size) < np.exp(-y / sc)]])
        out.append(x[:n])
    return out


def fig_collapse(d2):
    rng = np.random.default_rng(12)
    Lsyn = [32, 64, 128, 256]
    syn = synthetic_fss(1.6, 2.0, Lsyn, 200_000, rng)
    ts, Ds, cs = stats.fit_collapse(syn, Lsyn, 1.5, 2.2, xmin_scaled=1e-3)
    numbers.update(synTau=f"{ts:.3f}", synD=f"{Ds:.3f}", synCost=f"{cs:.1f}")

    samples = [d2[L] for L in L2D]
    qs = np.linspace(1.0, 4.0, 13)
    sig, err = stats.moment_exponents(samples, L2D, qs, n_boot=100, rng=4, fit_last=3)
    D_mom = np.polyfit(qs[qs >= 2], sig[qs >= 2], 1)[0]
    tau_mom = 2 - 2 / D_mom
    numbers.update(momD=f"{D_mom:.2f}", momTau=f"{tau_mom:.2f}",
                   momCost=f"{stats.collapse_cost(samples, L2D, tau_mom, D_mom, xmin_scaled=1e-4):.0f}")
    for tag, xs in (("A", 1e-4), ("B", 1e-2)):
        t, d, c = stats.fit_collapse(samples, L2D, tau_mom, D_mom, xmin_scaled=xs)
        numbers.update({f"coll{tag}Tau": f"{t:.2f}", f"coll{tag}D": f"{d:.2f}", f"coll{tag}Cost": f"{c:.0f}"})
    # local D from consecutive pairs of L, using the slope of sigma(q) for q in [2, 4]
    Dloc, Derr = [], []
    hi = qs >= 2
    for i in range(len(L2D) - 1):
        rng_b = np.random.default_rng(5)
        pair, Lp = samples[i:i + 2], L2D[i:i + 2]
        sg, _ = stats.moment_exponents(pair, Lp, qs, n_boot=2, rng=rng_b)
        Dloc.append(np.polyfit(qs[hi], sg[hi], 1)[0])
        # bootstrap the slope by resampling blocks of both time series
        boots = []
        for _ in range(40):
            res = [x[:x.size // 50 * 50].reshape(50, -1)[rng_b.integers(0, 50, 50)].ravel() for x in pair]
            sb, _ = stats.moment_exponents(res, Lp, qs, n_boot=2, rng=rng_b)
            boots.append(np.polyfit(qs[hi], sb[hi], 1)[0])
        Derr.append(np.std(boots, ddof=1))
    numbers["Dloc"] = ", ".join(f"{d:.2f}" for d in Dloc)
    numbers["Derr"] = ", ".join(f"{d:.2f}" for d in Derr)

    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.5))
    for (x, y, _), L, c in zip(stats.collapse_curves(syn, Lsyn, ts, Ds), Lsyn, ps.ordinal(len(Lsyn))):
        axes[0].loglog(x, y, "-", color=c, lw=1.2, label=f"$L={L}$")
    axes[0].set_title("synthetic data", fontsize=8.5)
    axes[0].legend(loc="lower left", fontsize=6.5)
    for (x, y, _), L, c in zip(stats.collapse_curves(samples, L2D, tau_mom, D_mom), L2D, ps.ordinal(len(L2D))):
        axes[1].loglog(x, y, "-", color=c, lw=1.2, label=f"$L={L}$")
    axes[1].set_title(fr"2D BTW, $\tau={tau_mom:.2f}$, $D={D_mom:.2f}$", fontsize=8.5)
    axes[1].legend(loc="lower left", fontsize=6.5)
    for ax in axes[:2]:
        ax.set_xlabel(r"$s/L^{D}$")
    axes[0].set_ylabel(r"$s^{\tau}P(s)$")
    Lmid = [np.sqrt(L2D[i] * L2D[i + 1]) for i in range(len(L2D) - 1)]
    axes[2].errorbar(Lmid, Dloc, yerr=Derr, fmt="o-", color=ps.C[0], ms=3.5, capsize=2, label="pairs of $L$")
    axes[2].set_xscale("log")
    axes[2].axhline(2.73, color=ps.C[1], lw=1.0, label="Lübeck & Usadel")
    axes[2].set_xticks(Lmid, [f"{L2D[i]}\n{L2D[i+1]}" for i in range(len(L2D) - 1)], fontsize=6.5)
    axes[2].minorticks_off()
    axes[2].set_xlabel("pair of system sizes")
    axes[2].set_ylabel(r"effective $D$")
    axes[2].set_title("2D BTW: drift of $D$", fontsize=8.5)
    axes[2].legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    save(fig, "collapse")


def fig_corrections():
    """Synthetic data with exactly known asymptotics: tau=1.3, D=2, exponential cutoff."""
    from scipy import integrate
    tau, D = 1.3, 2.0
    Ls = np.array([16, 32, 64, 128, 256, 512, 1024, 2048])

    def moment(q, sc):
        # substitute s = e^u: integrand e^{(p - tau + 1) u} exp(-e^u / sc) is smooth in u
        f = lambda p: integrate.quad(lambda u: np.exp((p - tau + 1) * u - np.exp(u) / sc),
                                     0, np.log(60 * sc), limit=400)[0]
        return f(q) / f(0)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for q, c in zip((1.0, 2.0), (ps.C[0], ps.C[2])):
        m = np.array([moment(q, L**D) for L in Ls])
        loc = np.diff(np.log(m)) / np.diff(np.log(Ls))
        Lm = np.sqrt(Ls[1:] * Ls[:-1])
        ax.semilogx(Lm, loc, "o-", color=c, ms=3.5, label=fr"$q={q:.0f}$: local slope")
        ax.axhline(D * (q + 1 - tau), color=c, lw=0.8, ls="--")
    ax.set_xlabel("$L$")
    ax.set_ylabel(r"$\mathrm{d}\ln\langle s^q\rangle/\mathrm{d}\ln L$")
    ax.set_title(r"exact moments, $\tau=1.3$, $D=2$", fontsize=8.5)
    ax.legend(loc="center right", fontsize=7)
    fig.tight_layout()
    save(fig, "corrections")


def write_numbers():
    lines = []
    for k, v in numbers.items():
        name = "st" + "".join(ch for ch in k if ch.isalpha())
        lines.append(f"\\newcommand{{\\{name}}}{{{v}}}")
    (FIG / "numbers.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ps.use()
    fig_binning()
    fig_estimators()
    fig_xmin()
    fig_corrections()
    d2 = load2d()
    fig_effective_exponent(d2)
    fig_correlations(d2)
    fig_collapse(d2)
    write_numbers()


if __name__ == "__main__":
    main()
