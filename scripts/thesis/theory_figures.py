"""Figures and numbers for the theory chapter.

    python scripts/thesis/theory_figures.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bksoc import bktheory as T
from bksoc import plotstyle as ps
from bksoc import stats
from bksoc.mapped import MappedChain, RULE_FIXED, bk_one_block_drop

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "results"
MAP = ROOT / "data" / "mapping"
FIG = ROOT / "thesis" / "figures" / "theory"
ALPHAS = (1.0, 1.25, 1.35, 1.45, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
numbers = {}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def tag(x):
    return str(x).replace(".", "p").translate(str.maketrans("0123456789", "ABCDEFGHIJ"))


def blocks_density(hb, n_edges=160, nmax=8192):
    edges = np.unique(np.logspace(0, np.log10(nmax), n_edges).astype(int))
    dens = np.array([hb[a:b].sum() / (b - a) for a, b in zip(edges[:-1], edges[1:])])
    return np.sqrt(edges[:-1] * edges[1:]), dens


def crossover_size(hb, l, min_dip=0.2):
    """Minimum of the smoothed, log-binned block-number distribution between 3 l and the peak of
    delocalized events (searched above 5 l). Returns nan if there is no clear dip (one population)."""
    c, d = blocks_density(hb, n_edges=80, nmax=hb.size - 1)
    k = d > 0
    c, d = c[k], d[k]
    sm = np.convolve(np.log(d), np.ones(5) / 5, mode="same")
    upper = (c > 5 * l) & (c < c.max() / 3)
    if not np.any(upper):
        return np.nan
    peak = c[upper][np.argmax(sm[upper])]
    win = (c > 3 * l) & (c < peak)
    if not np.any(win):
        return np.nan
    i = np.argmin(sm[win])
    if sm[upper].max() - sm[win][i] < min_dip:
        return np.nan
    return c[win][i]


# ------------------------------------------------------------------ 1. branching process

def fig_branching():
    s = np.arange(1, 200001, 2)
    fig, ax = plt.subplots(figsize=(3.8, 2.7))
    for p, c in zip((0.5, 0.495, 0.48, 0.45), ps.ordinal(4)[::-1]):
        P = T.binary_branching_pmf(s, p)
        k = P > 1e-14
        ax.loglog(s[k], P[k], color=c, lw=1.2, label=fr"$m={2 * p:.2f}$")
    ax.loglog(s[100:], np.sqrt(2 / np.pi) * s[100:] ** -1.5, "--", color=ps.INK_MUTED, lw=0.8)
    ax.text(3e3, 2e-5, r"$s^{-3/2}$", color=ps.INK_2, fontsize=8)
    ax.set_xlabel("avalanche size $s$")
    ax.set_ylabel("$P(s)$")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    save(fig, "branching")


# ------------------------------------------------------------------ 2. amplification: slip per block vs group size

def fig_amplification():
    l, alpha, sigma = 6.0, 3.0, 0.01
    ns = np.arange(1, 13)
    d = np.load(MAP / "bk_a3_l6.npz")
    M, B = d["M"].astype(float), d["blocks"]
    med = np.array([np.median(M[B == n] / n) for n in ns])
    q1 = np.array([np.percentile(M[B == n] / n, 25) for n in ns])
    q3 = np.array([np.percentile(M[B == n] / n, 75) for n in ns])
    rigid = np.array([T.group_slip(n, l, alpha, sigma) for n in ns])
    drop = bk_one_block_drop(l, alpha, sigma)
    c = MappedChain(4096, l, RULE_FIXED, d=drop, seed=2, noise=0.2)
    while c.P < 30:
        c.run(500_000)
    r = c.run(2_000_000)
    sand = np.array([np.median(r["moment"][r["area"] == n] / n) for n in ns])
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.fill_between(ns, q1, q3, color=ps.BLUES[0], lw=0, alpha=0.7)
    ax.semilogy(ns, med, "o-", color=ps.C[0], ms=3.5, label="BK model (median, quartiles)")
    ax.semilogy(ns, rigid, "s--", color=ps.C[1], ms=3, lw=1, label="rigid group (upper bound)")
    ax.semilogy(ns, sand, "^-", color=ps.C[2], ms=3, lw=1, label="sandpile limit with noise")
    ax.axhline(drop / (2 * l * l + 1), color=ps.INK_MUTED, lw=0.8, ls=":", label="deterministic sandpile")
    ax.set_xlabel("number of blocks in the event $n$")
    ax.set_ylabel("slip per block $M/n$")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, "amplification")
    numbers["ampBK"] = f"{med[7] / med[0]:.0f}"
    numbers["ampRigid"] = f"{rigid[7] / rigid[0]:.0f}"
    numbers["ampSand"] = f"{sand[7] / sand[0]:.1f}"


# ------------------------------------------------------------------ 3. alpha-scan and the crossover

def fig_alpha():
    l = 6.0
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    cols = ps.ordinal(len(ALPHAS))
    rows = []
    for a, c in zip(ALPHAS, cols):
        f = RES / f"alpha_{a}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        h, e = d["h_mu"], d["mu_edges"]
        n = h.size // 5 * 5
        hh = h[:n].reshape(-1, 5).sum(axis=1)
        ee = e[: n + 1: 5]
        D = hh / (np.diff(ee) * float(d["dP"]) * float(d["N"]))
        k = hh > 0
        axes[0].plot(0.5 * (ee[1:] + ee[:-1])[k] - np.log(l), np.log(l * D[k]), color=c, lw=1.0, label=fr"$\alpha={a}$")
        cc, dd = blocks_density(d["h_blocks"], n_edges=60)
        kk = dd > 0
        axes[1].loglog(cc[kk], dd[kk] / float(d["dP"]) / float(d["N"]), color=c, lw=1.0)
        M = d["M"].astype(float)
        M1 = one_block(l, a, 0.01)
        tau = stats.fit_power_law(M, max(10 * M1, 1e-4 * l), 1e-2 * l, discrete=False)["tau"]
        rows.append((a, tau, crossover_moment(M, M1) / l, T.runaway_size(l, a),
                     T.carlson_localization_length(l, a, 0.01)))
    axes[0].set_xlabel(r"$\ln(M/\xi)$")
    axes[0].set_ylabel(r"$\ln[\xi\,\mathcal{D}(\mu)]$")
    axes[0].legend(fontsize=5.5, loc="lower left", ncol=2)
    axes[1].set_xlabel(r"blocks per event $\Delta$")
    axes[1].set_ylabel(r"$K(\Delta)$")
    fig.tight_layout()
    save(fig, "alpha_scan")
    lines = [f"{a} & {tau:.2f} & " + ("---" if not np.isfinite(n_) else f"{n_:.2f}") + " & " +
             ("$\\infty$" if not np.isfinite(ns_) else f"{ns_:.0f}") + f" & {xt:.0f} \\\\" for a, tau, n_, ns_, xt in rows]
    (FIG / "alpha_table.tex").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ 4. noisy sandpile: diffusive scaling

def fig_sandpile_scaling():
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6))
    for l, c in ((6, ps.C[0]), (12, ps.C[1])):
        d = np.load(MAP / f"noise_fixed_w0.2_l{l}.npz")
        hb = d["h_blocks"].astype(float)
        a = np.arange(hb.size)
        k = hb > 0
        axes[0].semilogy(a[k] / l, l * hb[k] / hb.sum(), ".", ms=2.5, color=c, label=fr"$\ell={l}$")
        ht = d["h_topplings"].astype(float)
        s = np.arange(ht.size)
        k = ht > 0
        axes[1].loglog(s[k] / l**2, l**2 * ht[k] / ht.sum(), ".", ms=2, color=c, label=fr"$\ell={l}$")
        a_mean = (a * hb).sum() / hb.sum()
        s_mean = (s * ht).sum() / ht.sum()
        k = (hb > 0) & (a > 4 * l) & (a < 12 * l)
        decay = -1 / np.polyfit(a[k] / l, np.log(hb[k]), 1)[0]
        k = (ht > 0) & (s > l * l / 30) & (s < l * l / 1.5)
        tau = -np.polyfit(np.log(s[k]), np.log(ht[k]), 1)[0]
        w = {6: "six", 12: "twelve"}[l]
        numbers[f"sandA{w}"] = f"{a_mean / l:.2f}"
        numbers[f"sandS{w}"] = f"{s_mean / l**2:.2f}"
        numbers[f"sandDecay{w}"] = f"{decay:.2f}"
        numbers[f"sandTau{w}"] = f"{tau:.2f}"
    axes[0].set_xlabel(r"area / $\ell$")
    axes[0].set_ylabel(r"$\ell\,P(a)$")
    axes[1].set_xlabel(r"size / $\ell^2$")
    axes[1].set_ylabel(r"$\ell^2\,P(s)$")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "sandpile_scaling")


def one_block(l, a, s):
    w1 = 2 * l * l + 1
    return s / w1 * (1 + np.exp(np.pi * a / np.sqrt(w1 - a * a)))


def crossover_moment(M, M1):
    """Moment at which the effective exponent (fits in windows [M, 10 M]) first drops below 1, i.e. where
    the localized power law (tau ~ 2) gives way to the rising delocalized branch (tau ~ 0)."""
    lo = np.logspace(np.log10(100 * M1), np.log10(M.max()) - 1.2, 60)
    te = np.array([stats.fit_power_law(M, x, 10 * x, discrete=False)["tau"] for x in lo])
    cen = np.sqrt(10) * lo
    below = np.nonzero(te < 1)[0]
    if below.size == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    x0, x1 = np.log(cen[i - 1]), np.log(cen[i])
    return np.exp(x0 + (1 - te[i - 1]) * (x1 - x0) / (te[i] - te[i - 1]))


def dip_depth(d, l):
    """Depth (in ln) of the minimum of D(mu) below the peak of delocalized events; <= 0: no separate population."""
    h, e = d["h_mu"], d["mu_edges"]
    n = h.size // 10 * 10
    hh = h[:n].reshape(-1, 10).sum(axis=1)
    ee = e[: n + 1: 10]
    x = 0.5 * (ee[1:] + ee[:-1]) - np.log(l)
    k = hh > 0
    y = np.log(np.where(k, hh / np.diff(ee), 1e-300))
    sel = k & (x > np.log(1e-2)) & (x < np.log(30))
    i = np.argmin(np.where(sel, y, 1e300))
    right = k & (x > x[i])
    return y[right].max() - y[i]


DIP_RUNS = ([(6.0, a, RES / f"alpha_{a}.npz") for a in ALPHAS] +
            [(12.0, a, RES / f"alpha_{a}_l12.npz") for a in (1.25, 1.5, 1.75, 2.0)] +
            [(8.0, 3.0, MAP / "bk_a3_l8.npz"), (12.0, 3.0, RES / "ell_12_N16384.npz"), (16.0, 3.0, RES / "ell_16.npz"),
             (24.0, 3.0, RES / "ell_24.npz"), (10.0, 1.0, MAP / "bk_a1_l10.npz"), (14.0, 1.0, MAP / "bk_a1_l14.npz")])
L_RUNS = ((6, MAP / "bk_a3_l6.npz"), (8, MAP / "bk_a3_l8.npz"), (12, RES / "ell_12_N16384.npz"),
          (16, RES / "ell_16.npz"), (24, RES / "ell_24.npz"))


def onset_alpha(l, sigma=0.01):
    """alpha at which the runaway size equals the localization length, n* = xi_tilde."""
    from scipy.optimize import brentq
    return brentq(lambda a: T.runaway_size(l, a) - T.carlson_localization_length(l, a, sigma), 1.0001, 10)


def fig_crossover():
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))
    A, Mc = [], []
    for a in ALPHAS:
        f = RES / f"alpha_{a}.npz"
        if f.exists():
            A.append(a)
            Mc.append(crossover_moment(np.load(f)["M"].astype(float), one_block(6, a, 0.01)) / 6)
    A, Mc = np.array(A), np.array(Mc)
    ag = np.linspace(1.0, 5.2, 100)
    axes[0].plot(A, Mc, "o", color=ps.C[0], label="measured $M_c/\\xi$")
    axes[0].plot(ag, 2 / ag, "--", color=ps.C[2], lw=1, label=r"$\tilde M/\xi = 2/\alpha$")
    axes[0].set_xlabel(r"$\alpha$ ($\ell = 6$)")
    axes[0].set_ylabel(r"crossover moment $M_c/\xi$")
    axes[0].set_ylim(0, 2.2)
    axes[0].legend(fontsize=6.5)

    # bimodality criterion n* < xi_tilde: depth of the minimum of D(mu) against alpha, with the predicted onsets
    rows = []
    for l, a, f in DIP_RUNS:
        if f.exists():
            rows.append((l, a, T.runaway_size(l, a), T.carlson_localization_length(l, a, 0.01), dip_depth(np.load(f), l)))
    for l, col, mk in ((6.0, ps.C[0], "o"), (12.0, ps.C[1], "s")):
        sel = sorted(r for r in rows if r[0] == l)
        axes[1].plot([r[1] for r in sel], [r[4] for r in sel], mk + "-", color=col, ms=3.5, lw=1, label=fr"$\ell={l:.0f}$")
        a_on = onset_alpha(l)
        axes[1].axvline(a_on, color=col, lw=0.8, ls="--")
        numbers[f"onset{'six' if l == 6 else 'twelve'}"] = f"{a_on:.2f}"
    axes[1].axhline(0, color=ps.INK_MUTED, lw=0.6)
    axes[1].set_xscale("log")
    axes[1].set_xticks([1, 1.5, 2, 3, 5], ["1", "1.5", "2", "3", "5"])
    axes[1].minorticks_off()
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel("depth of the minimum (ln)")
    axes[1].legend(fontsize=6.5, loc="lower right")

    Ls, McL = [], []
    for l, f in L_RUNS:
        if f.exists():
            Ls.append(l)
            McL.append(crossover_moment(np.load(f)["M"].astype(float), one_block(l, 3.0, 0.01)))
    Ls, McL = np.array(Ls, dtype=float), np.array(McL)
    axes[2].loglog(Ls, McL / Ls, "o", color=ps.C[0], label=r"measured $M_c/\xi$")
    axes[2].axhline(2 / 3, color=ps.C[2], ls="--", lw=1, label=r"$\tilde M/\xi = 2/\alpha$")
    axes[2].loglog(Ls, McL[0] / Ls, ":", color=ps.INK_MUTED, lw=1, label=r"$M_c=$ const")
    axes[2].set_xlabel(r"$\ell$ ($\alpha=3$)")
    axes[2].set_ylabel(r"$M_c/\xi$")
    axes[2].set_xticks(Ls, [f"{int(v)}" for v in Ls])
    axes[2].set_yticks([0.1, 0.2, 0.5, 1], ["0.1", "0.2", "0.5", "1"])
    axes[2].minorticks_off()
    axes[2].legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, "crossover")
    numbers["McMin"] = f"{np.nanmin(Mc):.2f}"
    numbers["McMax"] = f"{np.nanmax(Mc):.2f}"
    numbers["McLmin"] = f"{McL.min():.1f}"
    numbers["McLmax"] = f"{McL.max():.1f}"
    numbers["McXiLsix"] = f"{McL[0] / Ls[0]:.2f}"
    numbers["McXiLtwentyfour"] = f"{McL[-1] / Ls[-1]:.2f}"
    lines = []
    for l, a, n_, xt, dd in sorted(rows, key=lambda r: (r[0], r[1])):
        ns_ = "$\\infty$" if not np.isfinite(n_) else f"{n_:.0f}"
        rr = "$\\infty$" if not np.isfinite(n_) else f"{n_ / xt:.2f}"
        lines.append(f"{l:.0f} & {a:g} & {ns_} & {xt:.0f} & {rr} & {dd:.1f} \\\\")
    (FIG / "dip_table.tex").write_text("\n".join(lines) + "\n")


def write_numbers():
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\th{k}}}{{{v}}}" for k, v in numbers.items()) + "\n")
    print(numbers)


def main():
    ps.use()
    fig_branching()
    fig_amplification()
    fig_alpha()
    fig_sandpile_scaling()
    fig_crossover()
    write_numbers()


if __name__ == "__main__":
    main()
