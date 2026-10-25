"""Figures and numbers for the results chapter (data from results_data.py and mapping_data.py)."""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bksoc import plotstyle as ps
from bksoc import stats

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "results"
MAP = ROOT / "data" / "mapping"
FIG = ROOT / "thesis" / "figures" / "results"
numbers = {}
SIGMAS = (0.1, 0.03, 0.01, 0.003, 0.001, 0.0003)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def load(path):
    return np.load(path), json.loads(path.with_suffix(".json").read_text())


def D_mu(d, rebin=5, per_time=False):
    """Events per unit magnitude and unit length, per unit plate displacement (or per unit time)."""
    h, e = d["h_mu"], d["mu_edges"]
    n = h.size // rebin * rebin
    h = h[:n].reshape(-1, rebin).sum(axis=1)
    e = e[: n + 1: rebin]
    norm = np.diff(e) * float(d["N"]) * (float(d["dtau"]) if per_time else float(d["dP"]))
    cen = 0.5 * (e[1:] + e[:-1])
    k = h > 0
    return cen[k], (h / norm)[k], (np.sqrt(h) / norm)[k]


def one_block_moment(l, alpha, sigma):
    w1 = 2 * l * l + 1
    return sigma / w1 * (1 + np.exp(np.pi * alpha / np.sqrt(w1 - alpha**2)))


def tag(x):
    return str(x).replace(".", "p").replace("-", "m").translate(str.maketrans("0123456789", "ABCDEFGHIJ"))


# ------------------------------------------------------------------ normalisation (1991 Fig. 3)

# landmark values read from Fig. 3 of Carlson et al. (1991): (ln(M/xi), ln(xi D))
FIG3 = {6: dict(slope=(-5.0, -5.5), minimum=(-0.8, -8.5), peak=(3.7, -5.9)),
        12: dict(slope=(-5.0, -6.6), minimum=(-0.5, -10.0), peak=(4.0, -6.6))}


def fig_normalisation():
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for l, c in ((6, ps.C[0]), (12, ps.C[2])):
        d = np.load(MAP / f"bk_a3_l{l}.npz")
        h, e = d["h_mu"], d["mu_edges"]
        n = h.size // 10 * 10
        hh = h[:n].reshape(-1, 10).sum(axis=1)
        ee = e[: n + 1: 10]
        D = hh / (np.diff(ee) * float(d["dP"]) * float(d["N"]))
        k = hh > 0
        x = 0.5 * (ee[1:] + ee[:-1])[k] - np.log(l)
        y = np.log(l * D[k])
        ax.plot(x, y, "-", color=c, lw=1.3, label=fr"this work, $\ell={l}$")
        pts = np.array(list(FIG3[l].values()))
        ax.plot(pts[:, 0], pts[:, 1], "s", ms=5, mfc="none", mec=c, mew=1.2, label=fr"Carlson et al., $\ell={l}$")
        offs = [np.interp(px, x, y) - py for px, py in FIG3[l].values()]
        numbers[f"offMean{tag(l)}"] = f"{np.mean(offs):.1f}"
        numbers[f"sumOurs{tag(l)}"] = f"{np.trapz(np.exp(x + y), x):.2f}"
        # the paper's curves, approximated by our curve shifted down by the mean offset
        numbers[f"sumTheirs{tag(l)}"] = f"{np.trapz(np.exp(x + y - np.mean(offs)), x):.2f}"
    ax.set_xlabel(r"$\ln(M/\xi)$")
    ax.set_ylabel(r"$\ln[\xi\,\mathcal{D}(\mu)]$")
    ax.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    save(fig, "normalisation")


# ------------------------------------------------------------------ sigma-scan

def blocks_density(hb, n_edges=48, nmax=8192):
    """Log-binned density of the number of blocks per event."""
    edges = np.unique(np.logspace(0, np.log10(nmax), n_edges).astype(int))
    dens = np.array([hb[a:b].sum() / (b - a) for a, b in zip(edges[:-1], edges[1:])])
    return np.sqrt(edges[:-1] * edges[1:]), dens


def fig_sigma():
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.6))
    cols = ps.ordinal(len(SIGMAS))
    rows = []
    l, alpha = 6.0, 3.0
    for sig, c in zip(SIGMAS, cols):
        f = RES / f"sig_{sig}.npz"
        if not f.exists():
            continue
        d, meta = load(f)
        mu, D, _ = D_mu(d)
        axes[0].plot(mu - np.log(l), np.log(l * D), "-", color=c, lw=1.1, label=fr"$\sigma={sig}$")
        M = d["M"]
        M1 = one_block_moment(l, alpha, sig)
        lo = np.logspace(np.log10(10 * M1), np.log10(M.max()) - 1.3, 36)
        te = []
        for x in lo:
            fit = stats.fit_power_law(M, x, 10 * x, discrete=False)
            te.append((fit["tau"], fit["err"]))
        te = np.array(te)
        ok = te[:, 1] < 0.05
        axes[1].semilogx(np.sqrt(10) * lo[ok] / l, te[ok, 0], "-", color=c, lw=1.1)
        cen, dens = blocks_density(d["h_blocks"])
        k = dens > 0
        axes[2].loglog(cen[k], dens[k] / float(d["dP"]) / float(d["N"]), "-", color=c, lw=1.1)
        # localized power law, in the same window [1e-3 xi, 1e-2 xi] for every sigma
        if 10 * M1 < 1e-3 * l:
            fit = stats.fit_power_law(M, 1e-3 * l, 1e-2 * l, discrete=False)
            tau_s = f"${fit['tau']:.2f}\\pm{fit['err']:.2f}$"
        else:
            tau_s = "---"
        # crossover in moment: minimum of D(mu) between 1e-2 xi and 1e2 xi
        sel = (mu - np.log(l) > np.log(1e-2)) & (mu - np.log(l) < np.log(1e2))
        mu_min = mu[sel][np.argmin(D[sel])] if sig <= 0.03 else np.nan
        # crossover in size: minimum of K(Delta) between 3 l and the peak of delocalized events
        cf, df = blocks_density(d["h_blocks"], n_edges=160)
        kf = df > 0
        sm = np.convolve(np.log(np.where(kf, df, 1e-300)), np.ones(5) / 5, mode="same")
        win = (cf > 3 * l) & (cf < 1000) & kf
        peak = cf[win][np.argmax(sm[win] * 0 + np.where(cf[win] > 60, sm[win], -1e9))]
        win2 = (cf > 3 * l) & (cf < peak) & kf
        n_min = cf[win2][np.argmin(sm[win2])]
        xi_t = 2 * l / alpha * np.log(4 * l * l / sig)
        Mstar = np.percentile(M[M > 1.0], 99) if np.any(M > 1.0) else np.nan
        hb = d["h_blocks"]
        rate10 = hb[10:].sum() / (float(d["dP"]) * float(d["N"]))
        rows.append((sig, M1, rate10, np.exp(mu_min) / l, n_min, xi_t, Mstar / (2 * l / alpha)))
    axes[0].set_xlabel(r"$\ln(M/\xi)$")
    axes[0].set_ylabel(r"$\ln[\xi\,\mathcal{D}(\mu)]$")
    axes[0].legend(fontsize=6, loc="lower left")
    axes[1].axhline(2, color=ps.INK_MUTED, lw=0.7, ls=":")
    axes[1].set_xlabel(r"window centre $M/\xi$")
    axes[1].set_ylabel(r"$\tau_{\rm eff}$")
    axes[1].set_ylim(-0.5, 3)
    axes[2].set_xlabel(r"blocks per event $\Delta$")
    axes[2].set_ylabel(r"$K(\Delta)$")
    fig.tight_layout()
    save(fig, "sigma_scan")
    lines = [f"{sig} & ${M1:.1e}$ & {r10:.3f} & " + ("---" if np.isnan(mm) else f"{mm:.2f}") +
             f" & {nmin:.0f} & {xt:.0f} & {ms:.0f} \\\\"
             for sig, M1, r10, mm, nmin, xt, ms in rows]
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / "sigma_table.tex").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ l-scan

def fig_ell():
    runs = [(6, MAP / "bk_a3_l6.npz", "N=4096"), (8, MAP / "bk_a3_l8.npz", "N=4096"),
            (12, MAP / "bk_a3_l12.npz", "N=8192"), (12, RES / "ell_12_N16384.npz", "N=16384"),
            (16, RES / "ell_16.npz", "N=16384"), (24, RES / "ell_24.npz", "N=32768")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    cols = ps.ordinal(5)
    cmap = {6: cols[0], 8: cols[1], 12: cols[2], 16: cols[3], 24: cols[4]}
    pts = []
    for l, f, lab in runs:
        if not f.exists():
            continue
        d = np.load(f)
        mu, D, _ = D_mu(d)
        ls_ = "--" if lab == "N=16384" and l == 12 else "-"
        axes[0].plot(mu - np.log(l), np.log(l * D), ls_, color=cmap[l], lw=1.1, label=fr"$\ell={l}$, ${lab}$")
        M = d["M"]
        Mt = 2 * l / 3.0
        Mstar = np.percentile(M[M > Mt], 99) if np.any(M > Mt) else np.nan
        hb = d["h_blocks"]
        nmax = np.nonzero(hb)[0].max()
        pts.append((l, lab, Mstar / Mt, nmax, int(d["N"])))
    axes[0].set_xlabel(r"$\ln(M/\xi)$")
    axes[0].set_ylabel(r"$\ln[\xi\,\mathcal{D}(\mu)]$")
    axes[0].legend(fontsize=6, loc="lower left")
    good = [p for p in pts if not (p[0] == 12 and p[1] == "N=8192")]
    if good:
        L_ = np.array([p[0] for p in good], dtype=float)
        R_ = np.array([p[2] for p in good])
        axes[1].loglog(L_, R_, "o-", color=ps.C[0], label=r"$M^*/\tilde M$ (99th percentile)")
        k = np.polyfit(np.log(L_), np.log(R_), 1)[0]
        numbers["MstarExp"] = f"{k:.2f}"
        axes[1].set_xlabel(r"$\ell$")
        axes[1].set_ylabel(r"$M^*/\tilde M$")
        axes[1].set_xticks(L_, [f"{int(v)}" for v in L_])
        yt = [200, 300, 400, 600]
        axes[1].set_yticks(yt, [str(v) for v in yt])
        axes[1].minorticks_off()
        axes[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "ell_scan")
    # localized exponent in [max(10 M1, 1e-4 xi), 1e-2 xi]
    taus = {}
    for l, f, lab in runs:
        if not f.exists() or (l == 12 and lab == "N=16384"):
            continue
        M = np.load(f)["M"].astype(float)
        lo = max(10 * one_block_moment(l, 3.0, 0.01), 1e-4 * l)
        fit = stats.fit_power_law(M, lo, 1e-2 * l, discrete=False)
        taus[l] = (fit["tau"], fit["err"])
        numbers[f"tauEll{tag(l)}"] = f"{fit['tau']:.2f}"
        numbers[f"errEll{tag(l)}"] = f"{fit['err']:.2f}"
    lines = [f"{l} & {N} & {nmax} & {r:.0f} & " + (f"${taus[l][0]:.3f}\\pm{taus[l][1]:.3f}$" if (l in taus and not
             (l == 12 and N == 16384)) else "---") + " \\\\" for l, lab, r, nmax, N in pts]
    (FIG / "ell_table.tex").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ 1989 model

FIG6 = {"a": (2.5, 0.01, 100, 1000), "b": (2.5, 0.001, 100, 320), "c": (2.5, 0.001, 400, 225),
        "d": (1.0, 0.001, 100, 45), "e": (4.0, 0.01, 100, 50)}
# windows of the scaling region of localized events, in M (see Fig. 6 of Carlson & Langer 1989)
B_WINDOW = {"a": (np.exp(-5), np.exp(-1)), "b": (np.exp(-5), np.exp(-1)), "c": (np.exp(-5), np.exp(-1)),
            "d": (np.exp(-7), np.exp(2)), "e": (np.exp(-5), np.exp(-1))}


def fig_1989():
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.6), sharex=True, sharey=True)
    rows = []
    for ax, (key, (alpha, nu, N, T)) in zip(axes.ravel(), FIG6.items()):
        lo, hi = B_WINDOW[key]
        bs = []
        for s_ in range(1, 6):
            f = RES / f"cl89_{key}_seed{s_}.npz"
            if f.exists():
                d = np.load(f)
                mu, R, _ = D_mu(d, rebin=6, per_time=True)
                ax.plot(mu, np.log(R), ".", ms=1.6, color=ps.BLUES[1], label="their run length" if s_ == 1 else None)
                M = d["M"]
                fit = stats.fit_power_law(M[M > 0], lo, hi, discrete=False)
                bs.append(fit["tau"] - 1)
        f = RES / f"cl89_{key}_long.npz"
        b_long, b_err = np.nan, np.nan
        if f.exists():
            d = np.load(f)
            mu, R, _ = D_mu(d, rebin=6, per_time=True)
            ax.plot(mu, np.log(R), "-", color=ps.C[1], lw=1.1, label="long run")
            M = d["M"]
            fit = stats.fit_power_law(M[M > 0], lo, hi, discrete=False)
            b_long, b_err = fit["tau"] - 1, fit["err"]
            numbers[f"lenLong{key.upper()}"] = f"{float(d['dtau']) * nu / 2:.0f}"
        ax.set_title(fr"({key}) $\alpha={alpha}$, $\nu={nu}$, $N={N}$", fontsize=8)
        ax.axvline(np.log(2 * N), color=ps.INK_MUTED, lw=0.7, ls=":")
        bs = np.array(bs)
        rows.append((key, alpha, nu, N, T, b_long, b_err, bs.min() if bs.size else np.nan,
                     bs.max() if bs.size else np.nan))
    axes[1, 2].axis("off")
    axes[0, 0].legend(fontsize=6, loc="lower left", markerscale=4)
    for ax in axes[1]:
        ax.set_xlabel(r"$\mu=\ln M$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\ln\mathcal{R}(\mu)$")
    fig.tight_layout()
    save(fig, "cl1989")
    lines = [f"({k}) & {a} & {n} & {N_} & {T_} & " + ("---" if np.isnan(bl) else f"${bl:.2f}\\pm{be:.2f}$") +
             " & " + ("---" if np.isnan(bmin) else f"{bmin:.2f}--{bmax:.2f}") + " \\\\"
             for k, a, n, N_, T_, bl, be, bmin, bmax in rows]
    (FIG / "cl1989_table.tex").write_text("\n".join(lines) + "\n")


def write_numbers():
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\res{k}}}{{{v}}}" for k, v in numbers.items()) + "\n")
    print(numbers)


def main():
    ps.use()
    fig_normalisation()
    fig_sigma()
    fig_ell()
    fig_1989()
    write_numbers()


if __name__ == "__main__":
    main()
