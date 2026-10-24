"""Figures and numbers for the chapter on numerical methods and validation.

    python scripts/thesis/validation_figures.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bksoc import plotstyle as ps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_bk import chain_at_threshold, reference_event, run_one  # noqa: E402

FIG = ROOT / "thesis" / "figures" / "validation"
DATA = ROOT / "data" / "validation"
numbers = {}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def fig_convergence():
    hs = np.array([8e-3, 4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4])
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    cases = [(0, 0.003, "cascade, 10 blocks"), (3, 0.01, "cascade, 10 blocks"), (0, 0.3, "one-block event")]
    for (seed, spread, lab), c in zip(cases, (ps.C[0], ps.C[2], ps.C[1])):
        rng = np.random.default_rng(seed)
        U, P = chain_at_threshold(10, 3.0, rng, spread)
        U_ref, _ = reference_event(U, P, 3.0, 2.0, 0.05)
        err = [np.max(np.abs(run_one(U, P, 3.0, 2.0, 0.05, h)[0] - U_ref)) for h in hs]
        ax.loglog(hs, err, "o-", ms=3, color=c, lw=1.2, label=lab if seed != 3 else None)
        if seed == 0 and spread < 0.1:
            numbers["errCascadeHA"] = f"{err[3]:.1e}"
    ax.loglog(hs, 3e-2 * hs**2, "--", color=ps.INK_MUTED, lw=0.9, label=r"$\propto h^2$")
    ax.set_xlabel("time step $h$")
    ax.set_ylabel("max. position error")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "convergence")


def fig_statistical():
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for c, col, ls in ((0.1, ps.C[0], "-"), (0.05, ps.C[1], "--")):
        f = DATA / f"h_{c}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        mu = np.log(d["M"])
        edges = np.arange(-9, 7.01, 0.25)
        h, _ = np.histogram(mu, edges)
        D = h / (0.25 * float(d["dP"]) * float(d["N"]))
        k = h > 0
        cen = 0.5 * (edges[1:] + edges[:-1])
        ax.plot(cen[k], np.log(D[k]), ls, color=col, lw=1.3,
                label=fr"$h = {c}/\omega_{{\max}}$")
        numbers[f"statEvents{'A' if c == 0.1 else 'B'}"] = f"{d['M'].size:,}".replace(",", "{,}")
    ax.set_xlabel(r"$\mu = \ln M$")
    ax.set_ylabel(r"$\ln\mathcal{D}(\mu)$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "statistical")


def seed_numbers():
    import json
    f = DATA / "seeds.json"
    if not f.exists():
        return
    res = json.loads(f.read_text())
    for c, tag in ((0.1, "A"), (0.05, "B")):
        v = np.array([r["one_block"] for r in res if r["c"] == c])
        big = np.array([r["large"] for r in res if r["c"] == c])
        numbers[f"oneBlock{tag}"] = f"{v.mean():.1f}"
        numbers[f"oneBlockErr{tag}"] = f"{v.std(ddof=1) / np.sqrt(v.size):.1f}"
        numbers[f"oneBlockMin{tag}"] = f"{v.min():.0f}"
        numbers[f"oneBlockMax{tag}"] = f"{v.max():.0f}"
        numbers[f"large{tag}"] = f"{big.mean():.2f}"
        numbers[f"largeErr{tag}"] = f"{big.std(ddof=1) / np.sqrt(big.size):.2f}"


def main():
    ps.use()
    seed_numbers()
    fig_convergence()
    fig_statistical()
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\val{k}}}{{{v}}}" for k, v in numbers.items()) + "\n")
    print(numbers)


if __name__ == "__main__":
    main()
