"""Figures and numbers for the mapping chapter, from data/mapping (see mapping_data.py)."""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bksoc import plotstyle as ps
from bksoc import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "mapping"
FIG = ROOT / "thesis" / "figures" / "mapping"
numbers = {}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def load(name):
    d = np.load(DATA / f"{name}.npz")
    meta = json.loads((DATA / f"{name}.json").read_text())
    return d, meta


def D_mu(d, rebin=5):
    """Frequency per unit magnitude, per unit length and per unit plate displacement."""
    h = d["h_mu"]
    e = d["mu_edges"]
    n = h.size // rebin * rebin
    h = h[:n].reshape(-1, rebin).sum(axis=1)
    e = e[: n + 1: rebin]
    width = np.diff(e)
    cen = 0.5 * (e[1:] + e[:-1])
    norm = width * float(d["dP"]) * float(d["N"])
    keep = h > 0
    return cen[keep], (h / norm)[keep], (np.sqrt(h) / norm)[keep]


def fig_compare_alpha3():
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.7), sharey=True)
    for ax, l in zip(axes, (6, 8, 12)):
        for model, c, lab in (("bk", ps.C[0], "BK model"), ("fixed", ps.C[1], "sandpile, fixed drop"),
                              ("reset", ps.C[2], "sandpile, reset (OFC)")):
            name = f"{model}_a3_l{l}"
            if not (DATA / f"{name}.npz").exists():
                continue
            d, _ = load(name)
            mu, D, _ = D_mu(d)
            ax.plot(mu - np.log(l), np.log(l * D), "-", color=c, lw=1.2, label=lab)
        ax.set_title(fr"$\ell={l}$", fontsize=9)
        ax.set_xlabel(r"$\ln(M/\xi)$")
        ax.set_xlim(-12, 7)
    axes[0].set_ylabel(r"$\ln[\xi\,\mathcal{D}(\mu)]$")
    axes[0].legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "compare_alpha3")


def divergence(make, n_steps=3000, per=20):
    """Distance between two copies started 1e-12 apart, sampled every `per` events."""
    a, b = make(), make()
    b.perturb()
    out_P, out_d = [], []
    for _ in range(n_steps // per):
        a.run(per)
        b.run(per)
        out_P.append(a.P)
        out_d.append(max(np.max(np.abs(a.state_vector() - b.state_vector())), 1e-17))
    return np.array(out_P), np.array(out_d)


def fig_chaos():
    from bksoc.bk import BKChain
    from bksoc.mapped import MappedChain, RULE_FIXED, bk_one_block_drop

    class BKw(BKChain):
        def perturb(self):
            self.U[self.N // 2] += 1e-12
        def state_vector(self):
            return self.U - self.P

    class MPw(MappedChain):
        def perturb(self):
            self.g[self.N // 2] += 1e-12
        def state_vector(self):
            return self.g

    bk = lambda: BKw(256, 6.0, 3.0, 0.01, seed=11)
    warm = bk()
    warm.run(20000)
    U0, P0 = warm.U.copy(), warm.state.copy()

    def bk_from_warm():
        c = BKw(256, 6.0, 3.0, 0.01, U=U0.copy())
        c.state[:] = P0
        return c

    d = bk_one_block_drop(6.0, 3.0, 0.01)
    warm_m = MPw(256, 6.0, RULE_FIXED, d=d, seed=11)
    while warm_m.P < 50:
        warm_m.run(100000)
    g0, Pm0 = warm_m.g.copy(), warm_m.state.copy()

    def mp_from_warm():
        c = MPw(256, 6.0, RULE_FIXED, d=d, f=g0.copy())
        c.state[:] = Pm0
        return c

    Pb, db = divergence(bk_from_warm, 6000, 10)
    Pm, dm = divergence(mp_from_warm, 3_000_000, 20000)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.semilogy(Pb - Pb[0], db, color=ps.C[0], lw=1.2, label="BK model")
    ax.semilogy(Pm - Pm[0], dm, color=ps.C[1], lw=2.0, ls="--", label="sandpile, fixed drop")
    ax.set_xlabel("plate displacement")
    ax.set_ylabel("distance between copies")
    ax.set_xlim(0, 5)
    ax.set_ylim(1e-14, 10)
    numbers["SandDistEnd"] = f"{dm[-1]:.1e}"
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "chaos")
    grow = db > 1e-9
    if grow.sum() > 3:
        k = np.polyfit((Pb - Pb[0])[grow & (db < 1e-2)], np.log(db[grow & (db < 1e-2)]), 1)[0]
        numbers["LyapPlate"] = f"{k:.0f}"


def bk_exponent(name, lo, hi):
    """Truncated power-law MLE of the moment density P(M) in [lo, hi]; returns tau, err, b = tau - 1."""
    d, _ = load(name)
    M = d["M"].astype(float)
    f = stats.fit_power_law(M, lo, hi, discrete=False)
    return f["tau"], f["err"]


def fig_alpha_test():
    from bksoc.mapped import bk_one_block_drop
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=True)
    for a, c in ((3.0, ps.C[0]), (1.0, ps.C[2])):
        d, _ = load(f"bk_a{a:.0f}_l6")
        mu, D, _ = D_mu(d)
        axes[0].plot(mu, np.log(D), color=c, lw=1.2, label=fr"$\alpha={a:.0f}$")
        d, _ = load(f"fixed_a{a:.0f}_l6")
        mu, D, _ = D_mu(d)
        shift = np.log(bk_one_block_drop(6.0, a, 0.01))
        axes[1].plot(mu - shift, np.log(D), color=c, lw=1.2 if a == 3 else 2.4, ls="-" if a == 3 else ":",
                     label=fr"$\alpha={a:.0f}$")
    axes[0].set_title(r"BK model, $\ell=6$", fontsize=9)
    axes[1].set_title(r"sandpile (fixed drop), $\ell=6$", fontsize=9)
    axes[0].set_xlabel(r"$\mu=\ln M$")
    axes[1].set_xlabel(r"$\ln(M/\Delta f)$")
    axes[0].set_ylabel(r"$\ln\mathcal{D}(\mu)$")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "alpha_test")


def fig_blocks():
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    for name, c, lab in (("bk_a3_l6", ps.C[0], "BK model"), ("fixed_a3_l6", ps.C[1], "sandpile, fixed drop"),
                         ("noise_fixed_w0.2_l6", ps.C[2], r"sandpile, fixed drop, noise $w=0.2$")):
        if not (DATA / f"{name}.npz").exists():
            continue
        d, _ = load(name)
        h = d["h_blocks"].astype(float)
        n = np.arange(h.size)
        k = h > 0
        ax.loglog(n[k], h[k] / (float(d["dP"]) * float(d["N"])), ".", ms=2.5, color=c, label=lab)
    ax.set_xlabel("number of blocks that slipped")
    ax.set_ylabel("events per block per unit plate displ.")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, "blocks")


def fig_noise():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    for ax, l in zip(axes, (6, 12)):
        d, _ = load(f"bk_a3_l{l}")
        mu, D, _ = D_mu(d)
        ax.plot(mu - np.log(l), np.log(l * D), color=ps.INK, lw=1.6, label="BK model")
        for w, c in zip((0.01, 0.05, 0.2), ps.ordinal(3)):
            for model, ls in (("fixed", "-"), ("reset", "--")):
                name = f"noise_{model}_w{w}_l{l}"
                if not (DATA / f"{name}.npz").exists():
                    continue
                dd, _ = load(name)
                mu, D, _ = D_mu(dd)
                ax.plot(mu - np.log(l), np.log(l * D), ls, color=c, lw=1.1,
                        label=f"{model}, $w={w}$" if l == 6 else None)
        ax.set_title(fr"$\ell={l}$", fontsize=9)
        ax.set_xlabel(r"$\ln(M/\xi)$")
        ax.set_xlim(-12, 7)
    axes[0].set_ylabel(r"$\ln[\xi\,\mathcal{D}(\mu)]$")
    axes[0].legend(fontsize=6, loc="lower left", ncol=1)
    fig.tight_layout()
    save(fig, "noise")


def fig_bk_effective():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=True)
    for ax, (a, ls_) in zip(axes, ((3, (6, 8, 12)), (1, (6, 10, 14)))):
        for l, c in zip(ls_, ps.ordinal(3)):
            name = f"bk_a{a}_l{l}"
            if not (DATA / f"{name}.npz").exists():
                continue
            d, _ = load(name)
            M = d["M"].astype(float)
            lo = np.logspace(np.log10(M.min()) + 0.3, np.log10(M.max()) - 1.3, 40)
            taus, errs = [], []
            for x in lo:
                f = stats.fit_power_law(M, x, 10 * x, discrete=False)
                taus.append(f["tau"])
                errs.append(f["err"])
            taus, errs = np.array(taus), np.array(errs)
            ok = errs < 0.05
            ax.semilogx(np.sqrt(10) * lo[ok] / l, taus[ok], "-", color=c, lw=1.2, label=fr"$\ell={l}$")
        ax.axhline(2.0, color=ps.INK_MUTED, lw=0.7, ls=":")
        ax.set_title(fr"BK model, $\alpha={a}$", fontsize=9)
        ax.set_xlabel(r"window centre $M/\xi$")
        ax.legend(fontsize=7)
    axes[0].set_ylabel(r"$\tau_{\rm eff}$ (window $[M, 10M]$)")
    axes[0].set_ylim(-0.5, 3.0)
    fig.tight_layout()
    save(fig, "bk_effective")


def periodicity(N=4096, l=6.0):
    """Detect exact periodicity of the deterministic sandpile rules at the production size."""
    from bksoc.mapped import MappedChain, RULE_FIXED, RULE_RESET, bk_one_block_drop
    import json as js
    cache = DATA / "periodicity.json"
    if cache.exists():
        numbers.update(js.loads(cache.read_text()))
        return
    d = bk_one_block_drop(l, 3.0, 0.01)
    for rule, tag in ((RULE_FIXED, "Fixed"), (RULE_RESET, "Reset")):
        c = MappedChain(N, l, rule, d=d, f_res=1 - d, seed=21)
        while c.P < 300:
            c.run(500_000)
        r = c.run(400_000)
        ep, sz = r["epicenter"], r["size"]
        per = None
        for p in range(1, 150_000):
            if np.array_equal(ep[-100_000:], ep[-100_000 - p:-p]) and np.array_equal(sz[-100_000:], sz[-100_000 - p:-p]):
                per = p
                break
        numbers[f"per{tag}"] = f"{per:,}".replace(",", "{,}") if per else "none"
        if per:
            numbers[f"perPlate{tag}"] = f"{r['plate'][-1] - r['plate'][-1 - per]:.5f}"
            numbers[f"perTopplings{tag}"] = f"{sz[-per:].sum() / N:.1f}"
    numbers["dropAthreeLsix"] = f"{d:.5f}"
    keys = [k for k in numbers if k.startswith("per") or k == "dropAthreeLsix"]
    cache.write_text(js.dumps({k: numbers[k] for k in keys}, indent=1))


def table_numbers():
    import json as js
    for name in sorted(p.stem for p in DATA.glob("*.json") if (DATA / f"{p.stem}.npz").exists()):
        m = js.loads((DATA / f"{name}.json").read_text())
        d = np.load(DATA / f"{name}.npz")
        hb = d["h_blocks"]
        key = name.replace("_", "").replace(".", "p")
        key = "".join(ch for ch in key if ch.isalpha() or ch.isdigit())
        key = key.translate(str.maketrans("0123456789", "ABCDEFGHIJ"))
        numbers[f"{key}events"] = f"{m['events']:,}".replace(",", "{,}")
        numbers[f"{key}maxblocks"] = f"{np.nonzero(hb)[0].max()}"
        numbers[f"{key}plate"] = f"{m['plate_displacement']:.0f}"


def write_numbers():
    lines = [f"\\newcommand{{\\map{k}}}{{{v}}}" for k, v in numbers.items()]
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / "numbers.tex").write_text("\n".join(lines) + "\n")


def main():
    ps.use()
    fig_compare_alpha3()
    fig_chaos()
    fig_alpha_test()
    fig_blocks()
    fig_noise()
    fig_bk_effective()
    periodicity()
    for l, tag in ((6, "six"), (12, "twelve")):
        for model in ("fixed", "reset"):
            name = f"noise_{model}_w0.2_l{l}"
            if (DATA / f"{name}.npz").exists():
                d, _ = load(name)
                hb = d["h_blocks"]
                nb = np.arange(hb.size)
                numbers[f"noiseMax{model}{tag}"] = f"{nb[hb > 0].max()}"
                numbers[f"noiseMean{model}{tag}"] = f"{(nb * hb).sum() / hb.sum():.1f}"
                mu, D, _ = D_mu(d)
                numbers[f"noiseMaxM{model}{tag}"] = f"{np.exp(mu.max()) / l:.2f}"
    table_numbers()
    # exponents of the BK model, fitted where the effective exponent is flat (see bk_effective.pdf):
    # localized region for alpha = 3, the whole scaling range for alpha = 1
    L = "ABCDEFGHIJKLMNO"
    for a, ls_, lo, hi in ((3, (6, 8, 12), 1e-4, 1e-2), (1, (6, 10, 14), 1e-3, 1.0)):
        for l in ls_:
            name = f"bk_a{a}_l{l}"
            if not (DATA / f"{name}.npz").exists():
                continue
            # start above the peaks of one- and few-block events (moments up to ~9 M1)
            w = np.sqrt(2 * l * l + 1 - a * a)
            M1 = 0.01 / (2 * l * l + 1) * (1 + np.exp(np.pi * a / w))
            tau, err = bk_exponent(name, max(lo * l, 10 * M1), hi * l)
            tag = f"A{'three' if a == 3 else 'one'}L{L[l]}"
            numbers[f"tau{tag}"] = f"{tau:.3f}"
            numbers[f"err{tag}"] = f"{err:.3f}"
            numbers[f"b{tag}"] = f"{tau - 1:.2f}"
    write_numbers()


if __name__ == "__main__":
    main()
