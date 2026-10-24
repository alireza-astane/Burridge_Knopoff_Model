"""Figures for the thesis chapter on the Burridge-Knopoff model (analytical and small numerical
calculations only; the full simulations are in later chapters).

    python scripts/thesis/bk_theory_figures.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from bksoc import plotstyle as ps

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "thesis" / "figures" / "bk"


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def phi_1989(y):
    return np.sign(y) / (1 + np.abs(y))


def phi_1991(y, sigma):
    return (1 - sigma) / (1 + y / (1 - sigma))


# ------------------------------------------------------------------ friction laws

def fig_friction():
    sigma = 0.15
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6), sharey=True)
    y = np.linspace(0.001, 6, 400)
    ax = axes[0]
    ax.plot(y, phi_1989(y), color=ps.C[0])
    ax.plot(-y, phi_1989(-y), color=ps.C[0])
    ax.plot([0, 0], [-1, 1], color=ps.C[1], lw=2.2)
    ax.set_title("1989: symmetric")
    ax.set_xlim(-6, 6)
    ax = axes[1]
    ax.plot(y, phi_1991(y, sigma), color=ps.C[0], label="kinetic friction")
    ax.plot([0, 0], [-1.6, 1], color=ps.C[1], lw=2.2, label="static friction (any value)")
    ax.annotate("", xy=(0, -1.6), xytext=(0, -1.35),
                arrowprops=dict(arrowstyle="-|>", color=ps.C[1], lw=1.2))
    ax.annotate(r"starts at $1-\sigma$", xy=(0.03, 1 - sigma), xytext=(1.2, 0.55), fontsize=8,
                color=ps.INK_2, arrowprops=dict(arrowstyle="-", color=ps.INK_MUTED, lw=0.6))
    ax.annotate(r"static threshold 1", xy=(0.0, 1.0), xytext=(1.2, 1.02), fontsize=8,
                color=ps.INK_2, va="center", arrowprops=dict(arrowstyle="-", color=ps.INK_MUTED, lw=0.6))
    ax.set_title(r"1991: drop $\sigma$, no backward motion")
    ax.set_xlim(-2, 6)
    ax.legend(loc="lower right", fontsize=7)
    for ax in axes:
        ax.axhline(0, color=ps.INK_MUTED, lw=0.6)
        ax.set_xlabel(r"relative velocity $y = 2\alpha\,\dot U$ (units of $v_1$)")
    axes[0].set_ylabel(r"friction $\phi(y)$ (units of $F_0$)")
    axes[0].set_ylim(-1.7, 1.15)
    fig.tight_layout()
    save(fig, "friction")


# ------------------------------------------------------------------ one block

def single_block(alpha, sigma, nu, t_end, dt=1e-3):
    """One block on a pulling spring, 1991 friction law. Returns tau, U, Udot.

    Equation (fixed floor, plate at position nu*tau): U'' = (nu tau - U) - phi(2 alpha U').
    Stuck while the spring force nu tau - U < 1; slips until U' returns to 0.
    """
    taus, Us, Vs = [0.0], [0.0], [0.0]
    t, U, V = 0.0, 0.0, 0.0
    stuck = True
    while t < t_end:
        if stuck:
            t_slip = U + 1.0                     # nu t - U = 1  ->  t = (U + 1) / nu
            t_slip /= nu
            t_next = min(t_slip, t_end)
            taus += [t_next]
            Us += [U]
            Vs += [0.0]
            t = t_next
            stuck = False
            continue

        def rhs(_, y):
            return [y[1], (nu * _ - y[0]) - phi_1991(2 * alpha * max(y[1], 0.0), sigma)]

        def stop(_, y):
            return y[1] + 1e-12 if _ > t + 1e-6 else 1.0
        stop.terminal, stop.direction = True, -1
        sol = solve_ivp(rhs, (t, t_end), [U, V], events=stop, max_step=dt * 20, rtol=1e-9, atol=1e-12)
        taus += list(sol.t[1:])
        Us += list(sol.y[0, 1:])
        Vs += list(np.maximum(sol.y[1, 1:], 0))
        t, U, V = sol.t[-1], sol.y[0, -1], 0.0
        stuck = True
    return np.array(taus), np.array(Us), np.array(Vs)


def fig_single_block():
    nu, sigma = 0.05, 0.05
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 3.4), sharex=True)
    for alpha, c in ((2.5, ps.C[0]), (0.5, ps.C[2])):
        tau, U, V = single_block(alpha, sigma, nu, 130)
        axes[0].plot(tau, nu * tau - U, color=c, lw=1.2, label=fr"$\alpha={alpha}$")
        axes[1].plot(tau, V, color=c, lw=1.2, label=fr"$\alpha={alpha}$")
    axes[0].axhline(1, color=ps.INK_MUTED, lw=0.6, ls=":")
    axes[0].set_ylabel(r"spring force $\nu\tau-U$")
    axes[0].legend(loc="lower right", fontsize=7)
    axes[1].set_ylabel(r"velocity $\dot U$")
    axes[1].set_xlabel(r"time $\tau$")
    fig.tight_layout(h_pad=0.4)
    save(fig, "single_block")


# ------------------------------------------------------------------ linear stability

def fig_dispersion():
    q = np.linspace(0, 3, 400)
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5))
    for alpha, c in ((2.5, ps.C[0]), (0.5, ps.C[2])):
        disc = (alpha**2 - 1 - q**2).astype(complex)
        Om = alpha + np.sqrt(disc)
        axes[0].plot(q, Om.real, color=c, lw=1.4, label=fr"$\alpha={alpha}$")
        axes[1].plot(q, np.abs(Om.imag), color=c, lw=1.4, label=fr"$\alpha={alpha}$")
    axes[1].plot(q, q, color=ps.INK_MUTED, lw=0.8, ls="--", label=r"$\xi q$")
    axes[0].set_ylabel(r"growth rate $\mathrm{Re}\,\Omega$")
    axes[1].set_ylabel(r"frequency $|\mathrm{Im}\,\Omega|$")
    for ax in axes:
        ax.set_xlabel(r"wave number $\xi q$")
        ax.legend(fontsize=7)
    axes[0].set_ylim(0, 5.5)
    fig.tight_layout()
    save(fig, "dispersion")


def fig_pulses():
    """Linearised chain around the threshold state: u'' = l^2 Lap u - u + 2 alpha u', kicked at the centre."""
    N, l, alpha = 301, 10.0, 0.25
    def rhs(_, y):
        u, v = y[:N], y[N:]
        lap = np.empty(N)
        lap[1:-1] = u[2:] - 2 * u[1:-1] + u[:-2]
        lap[0] = u[1] - u[0]
        lap[-1] = u[-2] - u[-1]
        return np.concatenate([v, l**2 * lap - u + 2 * alpha * v])
    y0 = np.zeros(2 * N)
    jj = np.arange(N) - N // 2
    y0[N:] = np.exp(-0.5 * (jj / 3.0) ** 2)      # smooth kick, 3 blocks wide (much less than xi = l a)
    ts = np.linspace(0, 12, 241)
    sol = solve_ivp(rhs, (0, ts[-1]), y0, t_eval=ts, rtol=1e-8, atol=1e-11, method="DOP853")
    v = sol.y[N:].T
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6), gridspec_kw=dict(width_ratios=[1.4, 1]))
    j = np.arange(N) - N // 2
    snaps = [3, 7, 11]
    for tsnap, c in zip(snaps, ps.ordinal(len(snaps))):
        k = np.argmin(np.abs(ts - tsnap))
        axes[0].plot(j, v[k], color=c, lw=1.1, label=fr"$\tau={tsnap}$")
    axes[0].set_xlabel("block $j - j_0$")
    axes[0].set_ylabel(r"velocity $\dot u_j$")
    axes[0].set_title(r"velocity profile; pulses move $\ell$ blocks per unit $\tau$", fontsize=8.5)
    axes[0].legend(fontsize=7, loc="upper center")
    peak = np.abs(v).max(axis=1)
    axes[1].semilogy(ts, peak, color=ps.C[0], lw=1.4, label="largest velocity")
    axes[1].semilogy(ts, peak[120] * np.exp(alpha * (ts - ts[120])), "--", color=ps.C[1], lw=1.0,
                     label=r"$\propto e^{\alpha\tau}$")
    axes[1].set_xlabel(r"time $\tau$")
    axes[1].set_title("growth of the pulses", fontsize=8.5)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "pulses")


# ------------------------------------------------------------------ localization length

def fig_localization():
    sig = np.logspace(-4, -0.5, 100)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for (l, alpha), c in zip(((6, 3), (12, 3), (10, 1.5)), (ps.C[0], ps.C[2], ps.C[1])):
        xi_t = 2 * l / alpha * np.log(4 * l**2 / sig)
        ax.semilogx(sig, xi_t, color=c, lw=1.4, label=fr"$\ell={l}$, $\alpha={alpha}$")
    ax.set_xlabel(r"friction drop $\sigma$")
    ax.set_ylabel(r"$\tilde\xi/a$ (blocks)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "localization")


def main():
    ps.use()
    fig_friction()
    fig_single_block()
    fig_dispersion()
    fig_pulses()
    fig_localization()


if __name__ == "__main__":
    main()
