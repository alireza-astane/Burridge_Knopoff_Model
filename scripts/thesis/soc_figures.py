"""Data and figures for the thesis chapters on earthquakes, SOC and the BTW sandpile.

    python scripts/thesis/soc_figures.py            # simulate (cached in data/soc) and plot
    python scripts/thesis/soc_figures.py --replot   # plot only, from cached data
"""

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bksoc import BTW, open_lattice, theory
from bksoc import plotstyle as ps
from bksoc import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "soc"
FIG = ROOT / "thesis" / "figures" / "soc"

L1D = [32, 128, 512]
N1D = {32: 1_000_000, 128: 1_000_000, 512: 300_000}
L2D = [32, 64, 128, 256, 512]
N2D = 1_000_000


# ------------------------------------------------------------------ simulations

def run_1d(L):
    pile = BTW(open_lattice((L,)), seed=100 + L)
    pile.fill_max()
    pile.drive(20 * L)
    av = pile.drive(N1D[L])
    np.savez_compressed(DATA / f"btw1d_L{L}.npz", size=av["size"], duration=av["duration"])
    return L


def run_2d(L):
    nbr = open_lattice((L, L))
    pile = BTW(nbr, seed=200 + L)
    pile.fill_max()
    pile.drive(6 * L * L)                       # drains the excess of the full state
    t0 = time.time()
    chunks = {k: [] for k in ("size", "area", "duration")}
    heights = np.zeros(4)
    bulk = np.zeros((L, L), dtype=bool)
    bulk[L // 4: 3 * L // 4, L // 4: 3 * L // 4] = True
    bulk = bulk.ravel()
    n_chunks = 100
    for _ in range(n_chunks):
        av = pile.drive(N2D // n_chunks)
        for k in chunks:
            chunks[k].append(av[k])
        heights += np.bincount(pile.z[bulk], minlength=4)
    out = {k: np.concatenate(v) for k, v in chunks.items()}
    np.savez_compressed(DATA / f"btw2d_L{L}.npz", heights=heights, **out)
    # one large avalanche for the picture
    if L == 256:
        while True:
            site = pile.rng.integers(pile.N)
            z_before = pile.z.copy()
            rec = pile.add([site], record=True)
            if rec["size"][0] > 40_000:
                np.savez_compressed(DATA / "btw2d_snapshot.npz", topplings=rec["topplings"],
                                    z_before=z_before, site=site)
                break
    return L, time.time() - t0


def simulate():
    DATA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(len(L1D) + len(L2D)) as pool:
        r1 = pool.map_async(run_1d, L1D)
        r2 = pool.map_async(run_2d, L2D)
        r1.get()
        r2.get()
    meta = {"L1D": L1D, "N1D": N1D, "L2D": L2D, "N2D": N2D, "wall_seconds": time.time() - t0}
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2))
    print("simulations done in %.0f s" % meta["wall_seconds"])


# ------------------------------------------------------------------ figures

def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def fig_scale_invariance():
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.4), sharey=True)
    u = np.linspace(1, 10, 200)
    cols = ps.ordinal(3)
    for x0, c in zip([1, 10, 100], cols):
        x = x0 * u
        axes[0].plot(u, (x / x0) ** -1.5, color=c, label=fr"$x_0={x0}$")
        axes[1].plot(u, np.exp(-(x - x0) / 10), color=c, label=fr"$x_0={x0}$")
    axes[0].set_title(r"power law $p(x)\propto x^{-3/2}$")
    axes[1].set_title(r"exponential $p(x)\propto e^{-x/\xi}$, $\xi=10$")
    for ax in axes:
        ax.set_xlabel(r"$x/x_0$")
        ax.set_xlim(1, 10)
    axes[0].set_ylabel(r"$p(x)/p(x_0)$")
    axes[1].legend(loc="upper right")
    axes[0].text(5.5, 0.35, "all three windows\ncoincide", color=ps.INK_2, fontsize=8, ha="center")
    save(fig, "scale_invariance")


def fig_btw1d_profile():
    L, l, j = 40, 30, 12
    z0 = np.ones(L, dtype=np.int64)
    z0[l - 1] = 0
    pile = BTW(open_lattice((L,)), z=z0)
    rec = pile.add([j - 1], record=True)
    _, lp = theory.btw1d_size(L, l, j)
    i = np.arange(1, L + 1)
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    ax.bar(i, rec["topplings"], width=0.8, color=ps.C[0], label="simulation: topplings $n_i$")
    G = lambda k: np.minimum(i, k) - i * k / (L + 1)
    ax.plot(i, G(j) + G(lp) - G(l), color=ps.C[1], lw=1.5,
            label=r"exact: $G_{ij}+G_{il'}-G_{il}$")
    for x, lab in ((j, f"grain added\n$j={j}$"), (l, f"old zero\n$l={l}$"), (lp, f"new zero\n$l'={lp}$")):
        ax.annotate(lab, (x, 0), (x, -3.2), ha="center", va="top", fontsize=7.5, color=ps.INK_2,
                    arrowprops=dict(arrowstyle="-", color=ps.INK_MUTED, lw=0.6))
    ax.set_ylim(-0.5, None)
    ax.set_xlabel("site $i$", labelpad=30)
    ax.set_ylabel("topplings at site $i$")
    ax.set_title(f"one avalanche in the 1D BTW chain, $L={L}$: $s={rec['size'][0]}$ topplings")
    ax.legend(loc="upper right")
    save(fig, "btw1d_profile")


def fig_btw1d_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6))
    y = np.linspace(1e-4, 0.25, 600)
    cols = ps.ordinal(len(L1D))
    for L, c in zip(L1D, cols):
        d = np.load(DATA / f"btw1d_L{L}.npz")
        s = d["size"]
        Y = s[s > 0] / L**2
        h, e = np.histogram(Y, bins=40, range=(0, 0.25))
        dens = h / (s.size * np.diff(e))
        axes[0].plot(0.5 * (e[1:] + e[:-1]), dens, "o", ms=3, color=c, label=f"$L={L}$")
        ds = d["duration"]
        h, e = np.histogram(ds[ds > 0] / L, bins=25, range=(0, 1))
        axes[1].plot(0.5 * (e[1:] + e[:-1]), h / (ds.size * np.diff(e)), "o", ms=3, color=c,
                     label=f"$L={L}$")
    axes[0].plot(y, theory.btw1d_scaled_density(y), color=ps.C[1], lw=1.5, label="exact, $L\\to\\infty$")
    axes[0].set_xlabel(r"$y=s/L^2$")
    axes[0].set_ylabel("probability density")
    axes[0].set_title("avalanche size")
    axes[0].set_ylim(0, 8)
    axes[0].legend()
    t = np.linspace(0, 1, 50)
    axes[1].plot(t, 2 * t, color=ps.C[1], lw=1.5, label="exact: $2t$")
    axes[1].set_xlabel(r"$t=T/L$")
    axes[1].set_title("avalanche duration")
    axes[1].legend()
    fig.tight_layout()
    save(fig, "btw1d_distribution")


def fig_btw2d_snapshot():
    d = np.load(DATA / "btw2d_snapshot.npz")
    L = int(np.sqrt(d["topplings"].size))
    n = d["topplings"].reshape(L, L).astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.0))
    from matplotlib.colors import ListedColormap
    seq = ListedColormap(["#ffffff"] + ps.BLUES)
    im = axes[0].imshow(np.where(n > 0, n, np.nan), cmap="Blues", interpolation="nearest")
    axes[0].set_title(f"topplings per site (total $s={int(n.sum())}$)")
    fig.colorbar(im, ax=axes[0], shrink=0.75)
    zb = d["z_before"].reshape(L, L)
    axes[1].imshow(zb, cmap=ListedColormap(["#0d366b", "#2a78d6", "#86b6ef", "#ffffff"]),
                   interpolation="nearest", vmin=-0.5, vmax=3.5)
    axes[1].set_title("heights before the avalanche (0 dark … 3 white)")
    r, c = divmod(int(d["site"]), L)
    for ax in axes:
        ax.plot(c, r, "x", color=ps.C[1], ms=6, mew=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(ps.INK_MUTED)
            sp.set_linewidth(0.6)
    fig.tight_layout()
    save(fig, "btw2d_snapshot")


def load2d():
    return {L: np.load(DATA / f"btw2d_L{L}.npz") for L in L2D}


def fig_btw2d_distributions(d2):
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.4))
    cols = ps.ordinal(len(L2D))
    for L, c in zip(L2D, cols):
        for ax, key in zip(axes, ("size", "area", "duration")):
            x = d2[L][key]
            n = x.size
            x = x[x > 0]
            xc, p, _ = stats.log_binned_density(x, 6)
            ax.loglog(xc, p * x.size / n, "-", color=c, lw=1.2, label=f"$L={L}$")
    for ax, key, sym in zip(axes, ("size", "area", "duration"), ("s", "a", "T")):
        ax.set_xlabel(f"${sym}$")
        ax.set_title(f"avalanche {key}")
    axes[0].set_ylabel("probability density")
    # guide lines
    s = np.logspace(0.5, 4, 10)
    axes[0].loglog(s, 0.4 * s ** -1.2, "--", color=ps.INK_MUTED, lw=0.9)
    axes[0].text(1e3, 2e-4, r"$s^{-1.2}$", color=ps.INK_2, fontsize=8)
    axes[2].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    save(fig, "btw2d_distributions")


def moment_exponents(d2, qs):
    Ls = np.array(L2D, dtype=float)
    sig = []
    for q in qs:
        m = [np.mean(d2[L]["size"].astype(float) ** q) for L in L2D]
        # use the three largest sizes for the slope (closest to the asymptotic regime)
        sig.append(np.polyfit(np.log(Ls[-3:]), np.log(m[-3:]), 1)[0])
    return np.array(sig)


def fig_btw2d_moments(d2):
    qs = np.linspace(0.25, 4, 16)
    sig = moment_exponents(d2, qs)
    D_est = np.polyfit(qs[qs >= 2], sig[qs >= 2], 1)[0]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5))
    Ls = np.array(L2D)
    exact = [theory.mean_topplings_per_site(open_lattice((L, L))).mean() for L in L2D]
    sim = [d2[L]["size"].mean() for L in L2D]
    axes[0].loglog(Ls, exact, "-", color=ps.C[1], label=r"exact $\frac{1}{N}\sum_{ij}G_{ij}$")
    axes[0].loglog(Ls, sim, "o", color=ps.C[0], label="simulation")
    axes[0].set_xlabel("$L$")
    axes[0].set_xticks(L2D, [str(L) for L in L2D])
    axes[0].minorticks_off()
    axes[0].set_ylabel(r"$\langle s\rangle$")
    axes[0].set_title(r"mean size $\propto L^2$ (exact)")
    axes[0].legend()
    axes[1].plot(qs, sig, "o-", color=ps.C[0], label=r"measured $\sigma(q)$")
    axes[1].plot(qs, D_est * (qs - 1) + 2, "--", color=ps.INK_MUTED, lw=0.9,
                 label=fr"slope $D\approx{D_est:.2f}$")
    axes[1].plot([1], [2], "s", color=ps.C[1], ms=6, label=r"exact $\sigma(1)=2$")
    axes[1].set_xlabel("$q$")
    axes[1].set_ylabel(r"$\sigma(q)$ in $\langle s^q\rangle\propto L^{\sigma(q)}$")
    axes[1].set_title("moment analysis")
    axes[1].legend()
    fig.tight_layout()
    save(fig, "btw2d_moments")
    return D_est, dict(zip([float(q) for q in qs], [float(x) for x in sig]))


def numbers(d2, D_est):
    """Numbers quoted in the chapter text, written to a small LaTeX macro file."""
    L = max(L2D)
    p = d2[L]["heights"] / d2[L]["heights"].sum()
    exact_mean = {L: theory.mean_topplings_per_site(open_lattice((L, L))).mean() for L in L2D}
    tau_from_relation = 2 - 2 / D_est
    lines = [
        f"\\newcommand{{\\socHeightSim}}[1]{{\\ifcase#1 {p[0]:.4f}\\or {p[1]:.4f}\\or {p[2]:.4f}\\or {p[3]:.4f}\\fi}}",
        f"\\newcommand{{\\socMeanHeightSim}}{{{(p * np.arange(4)).sum():.4f}}}",
        f"\\newcommand{{\\socLmax}}{{{L}}}",
        f"\\newcommand{{\\socNav}}{{{N2D:,}}}".replace(",", "{,}"),
        f"\\newcommand{{\\socDest}}{{{D_est:.2f}}}",
        f"\\newcommand{{\\socTauRel}}{{{tau_from_relation:.2f}}}",
        f"\\newcommand{{\\socMeanSim}}{{{d2[L]['size'].mean():.0f}}}",
        f"\\newcommand{{\\socMeanExact}}{{{exact_mean[L]:.0f}}}",
    ]
    (FIG / "numbers.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def fig_gutenberg_richter():
    """Synthetic catalogue obeying Gutenberg-Richter with b = 1 (illustration only)."""
    rng = np.random.default_rng(11)
    b, m_min, n = 1.0, 2.0, 20_000
    mags = m_min + rng.exponential(1 / (b * np.log(10)), n)      # P(M_w >= m) = 10^{-b (m - m_min)}
    M0 = 10 ** (1.5 * mags + 9.1)                                  # Hanks-Kanamori, N m
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6))
    m_grid = np.arange(m_min, mags.max(), 0.1)
    Ncum = np.array([(mags >= m).sum() for m in m_grid])
    axes[0].semilogy(m_grid, Ncum, "o", ms=2.5, color=ps.C[0], label="synthetic catalogue")
    axes[0].semilogy(m_grid, n * 10 ** (-b * (m_grid - m_min)), color=ps.C[1], lw=1.2,
                     label=r"$\log_{10}N = a - b\,M_w$, $b=1$")
    axes[0].set_xlabel("moment magnitude $M_w$")
    axes[0].set_ylabel(r"$N(\geq M_w)$")
    axes[0].set_title("cumulative count (Gutenberg-Richter plot)")
    axes[0].legend()
    xc, p, _ = stats.log_binned_density(M0, 4)
    axes[1].loglog(xc, p, "o", ms=3, color=ps.C[0], label="synthetic catalogue")
    tau = 1 + 2 * b / 3
    axes[1].loglog(xc, p[0] * (xc / xc[0]) ** -tau, color=ps.C[1], lw=1.2,
                   label=r"$P(M_0)\propto M_0^{-5/3}$")
    axes[1].set_xlabel(r"seismic moment $M_0$ [N m]")
    axes[1].set_ylabel(r"$P(M_0)$ [1/(N m)]")
    axes[1].set_title("probability density of the moment")
    axes[1].legend()
    fig.tight_layout()
    save(fig, "gutenberg_richter")


def fig_percolation():
    from scipy import ndimage
    from matplotlib.colors import ListedColormap
    rng = np.random.default_rng(3)
    L = 150
    u = rng.random((L, L))
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.4))
    for ax, p in zip(axes, (0.45, 0.5927, 0.70)):
        occ = u < p
        lab, n = ndimage.label(occ)
        img = np.zeros((L, L))
        img[occ] = 1
        if n:
            sizes = np.bincount(lab.ravel())[1:]
            img[lab == 1 + sizes.argmax()] = 2
        ax.imshow(img, cmap=ListedColormap(["#ffffff", "#b7d3f6", ps.C[1]]), interpolation="nearest",
                  vmin=-0.5, vmax=2.5)
        ax.set_title(fr"$p={p}$" + (r"$\;\approx p_c$" if abs(p - 0.5927) < 1e-3 else ""))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(ps.INK_MUTED)
            sp.set_linewidth(0.6)
    fig.tight_layout()
    save(fig, "percolation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    if not args.replot:
        simulate()
    ps.use()
    fig_gutenberg_richter()
    fig_percolation()
    fig_scale_invariance()
    fig_btw1d_profile()
    fig_btw1d_distribution()
    fig_btw2d_snapshot()
    d2 = load2d()
    fig_btw2d_distributions(d2)
    D_est, _ = fig_btw2d_moments(d2)
    numbers(d2, D_est)


if __name__ == "__main__":
    main()
