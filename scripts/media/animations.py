"""Animations of the Burridge-Knopoff model and the BTW sandpile (MP4 + GIF) for the README.

    python scripts/media/animations.py            # all
    python scripts/media/animations.py bk_event   # one of: bk_event, bk_cycle, btw

Output: media/<name>.mp4 and media/<name>.gif
"""

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

from bksoc.bk import BKChain
from bksoc.sandpile import BTW, open_lattice
from bksoc import plotstyle as ps

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "media"

BG = "#0b1020"
PANEL = "#111833"
INK = "#e6e9f2"
INK2 = "#9aa3bf"
STUCK = "#3a4466"
GLOW = LinearSegmentedColormap.from_list("glow", ["#2b1b4a", "#7a2a6e", "#d1495b", "#f5873a", "#ffd166", "#fffbe6"])


def style():
    ps._register_pagella()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Pagella", "DejaVu Serif"],
                         "mathtext.fontset": "cm", "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#2a3354",
                         "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG})


def save(anim, fig, name, fps, width=720, colors=128):
    OUT.mkdir(parents=True, exist_ok=True)
    mp4 = OUT / f"{name}.mp4"
    anim.save(mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=2400, codec="libopenh264",
                                                  extra_args=["-pix_fmt", "yuv420p"]), dpi=110)
    plt.close(fig)
    gif = OUT / f"{name}.gif"
    pal = OUT / f"{name}_palette.png"
    w = width
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                    f"fps={min(fps, 20)},scale={w}:-1:flags=lanczos,palettegen=max_colors={colors}", str(pal)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-i", str(pal), "-lavfi",
                    f"fps={min(fps, 20)},scale={w}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
                    str(gif)], check=True)
    pal.unlink()
    print(name, f"{mp4.stat().st_size / 1e6:.1f} MB mp4, {gif.stat().st_size / 1e6:.1f} MB gif")


def zigzag(x0, y0, x1, y1, n=8, amp=0.12):
    t = np.linspace(0, 1, 2 * n + 1)
    x = x0 + (x1 - x0) * t
    y = y0 + (y1 - y0) * t
    off = np.zeros_like(t)
    off[1:-1:2] = amp
    off[2:-1:2] = -amp
    dx, dy = x1 - x0, y1 - y0
    L = np.hypot(dx, dy) + 1e-12
    nx, ny = -dy / L, dx / L
    return np.column_stack([x + off * nx, y + off * ny])


# ------------------------------------------------------------------ 1. a great event of the BK model

def bk_event():
    N, l, alpha, sigma = 48, 4.0, 3.0, 0.01
    c = BKChain(N, l, alpha, sigma, seed=11)
    c.run(30000)
    while True:
        snap = copy.deepcopy(c)
        r = c.run(1)
        if 0.55 * N <= r["blocks"][0] < N:
            break
    # replay the event, recording the state every few steps
    c = copy.deepcopy(snap)
    U_start = c.U.copy()
    frames_U, frames_V, frames_t = [c.U.copy()] * 12, [np.zeros(N)] * 12, [0.0] * 12
    tau0 = c.tau
    while True:
        r = c.run(1, max_steps=6)
        frames_U.append(c.U.copy())
        frames_V.append(c.V.copy())
        frames_t.append(c.tau - tau0)
        if len(r["moment"]):
            break
    frames_U += [c.U.copy()] * 25
    frames_V += [np.zeros(N)] * 25
    frames_t += [frames_t[-1]] * 25
    V = np.array(frames_V)
    vmax = V.max()

    fig = plt.figure(figsize=(8.0, 5.4))
    ax = fig.add_axes([0.04, 0.47, 0.92, 0.45])
    ax2 = fig.add_axes([0.08, 0.08, 0.84, 0.33])
    for a in (ax,):
        a.set_facecolor(BG)
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    ax.set_xlim(-1.5, N + 0.5)
    ax.set_ylim(-0.6, 3.2)
    ax.add_patch(Rectangle((-1.5, -0.35), N + 2, 0.3, color="#1c2440", lw=0))
    ax.plot([-1.5, N + 0.5], [-0.05, -0.05], color="#58628a", lw=1)
    ax.add_patch(Rectangle((-1.5, 2.55), N + 2, 0.25, color="#24305a", lw=0))
    ax.text(N + 0.4, 2.95, "moving plate", ha="right", va="center", fontsize=9, color=INK2)
    ax.text(N + 0.4, -0.5, "fixed rough surface", ha="right", va="center", fontsize=9, color=INK2)
    fig.text(0.04, 0.955, "Burridge–Knopoff model: a great earthquake", fontsize=14, color=INK)
    fig.text(0.96, 0.955, fr"$\ell={l:.0f}$, $\alpha={alpha:.0f}$, $\sigma={sigma}$", fontsize=10,
             color=INK2, ha="right")
    blocks = [Rectangle((0, 0), 0.5, 0.55) for _ in range(N)]
    hold0 = 12
    scale = 1.4                                   # displacements drawn 1.4x larger than true
    pc = PatchCollection(blocks, match_original=False, lw=0)
    ax.add_collection(pc)
    springs_c = LineCollection([], colors="#6c77a3", linewidths=0.9)
    springs_p = LineCollection([], colors="#4b5580", linewidths=0.8)
    ax.add_collection(springs_c)
    ax.add_collection(springs_p)
    norm = Normalize(0, vmax)

    n_ev = len(frames_t) - hold0 - 25           # frames recorded during the event
    img = np.full((n_ev, N), np.nan)
    im = ax2.imshow(img, aspect="auto", cmap=GLOW, norm=Normalize(0, vmax), interpolation="nearest",
                    extent=[0.5, N + 0.5, frames_t[hold0 + n_ev - 1], 0])
    ax2.set_xlabel("block", fontsize=9)
    ax2.set_ylabel(r"time $\tau$", fontsize=9)
    ax2.text(0.99, 0.04, "slip velocity in space and time", transform=ax2.transAxes, ha="right",
             fontsize=8.5, color=INK2)
    cb = fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.01)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_edgecolor("#2a3354")
    cb.set_label("slip velocity", fontsize=8, color=INK2)

    def draw(i):
        U = scale * (frames_U[i] - U_start)
        Vi = frames_V[i]
        xs = np.arange(N) + U
        for j, b in enumerate(blocks):
            b.set_xy((xs[j] - 0.25, 0.0))
        cols = [STUCK if v <= 0 else GLOW(0.15 + 0.85 * norm(v)) for v in Vi]
        pc.set_paths(blocks)
        pc.set_facecolor(cols)
        springs_c.set_segments([zigzag(xs[j] + 0.25, 0.27, xs[j + 1] - 0.25, 0.27, n=4, amp=0.08)
                                for j in range(N - 1)])
        springs_p.set_segments([zigzag(xs[j], 0.55, j, 2.55, n=6, amp=0.1) for j in range(N)])
        k = i - hold0
        if 0 <= k < n_ev:
            img[k] = np.where(Vi > 0, Vi, np.nan)
            im.set_data(img)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(frames_t), interval=50, blit=False)
    save(anim, fig, "bk_event", fps=24)


# ------------------------------------------------------------------ 2. the seismic cycle of a long fault

def bk_cycle():
    N, l, alpha, sigma = 400, 6.0, 3.0, 0.01
    c = BKChain(N, l, alpha, sigma, seed=3)
    c.run(200_000)
    edges = np.arange(-9, 6.01, 0.3)
    counts = np.zeros(edges.size - 1)
    profiles, events = [], []
    prev = c.U - c.P
    n_frames = 240
    while len(profiles) < n_frames:
        r = c.run(1)
        M, nb = r["moment"][0], r["blocks"][0]
        counts += np.histogram(np.log(M), edges)[0]
        if nb >= 4:
            cur = c.U - c.P
            profiles.append(cur.copy())
            events.append((prev.copy(), M, nb, counts.copy()))
            prev = cur
        else:
            prev = c.U - c.P
    fig = plt.figure(figsize=(8.4, 4.4))
    ax = fig.add_axes([0.07, 0.15, 0.58, 0.68])
    axh = fig.add_axes([0.73, 0.15, 0.24, 0.68])
    fig.text(0.04, 0.92, "A model fault: many small earthquakes, a few great ones", fontsize=13.5, color=INK)
    allp = np.concatenate([np.array(profiles), np.array([e[0] for e in events])])
    lo, hi = allp.min(), allp.max()
    pad = 0.1 * (hi - lo)
    ax.set_xlim(0, N)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("position along the fault (block)", fontsize=9)
    ax.set_ylabel("displacement relative to the plate", fontsize=9)
    ghosts = [ax.plot([], [], color="#5b6aa0", lw=0.6, alpha=a)[0] for a in np.linspace(0.08, 0.35, 8)]
    line_prev, = ax.plot([], [], color="#7f8bb8", lw=0.9)
    fill = [None]
    line, = ax.plot([], [], color="#ffd166", lw=1.3)
    label = fig.text(0.07, 0.855, "", fontsize=10, color=INK)
    x = np.arange(N)
    centres = 0.5 * (edges[1:] + edges[:-1])
    bars = axh.barh(centres, np.ones_like(centres), height=0.27, color="#d1495b")
    axh.set_xscale("log")
    axh.set_xlim(0.8, None)
    axh.set_ylim(edges[0], edges[-1])
    axh.set_xlabel("number of events", fontsize=9)
    axh.set_ylabel(r"magnitude $\mu=\ln M$", fontsize=9)

    def draw(i):
        before, M, nb, cnt = events[i]
        after = profiles[i]
        for g, k in zip(ghosts, range(8, 0, -1)):
            if i - k >= 0:
                g.set_data(x, profiles[i - k])
        line_prev.set_data(x, before)
        line.set_data(x, after)
        if fill[0] is not None:
            fill[0].remove()
        fill[0] = ax.fill_between(x, before, after, where=after > before + 1e-9, color="#f5873a", alpha=0.55, lw=0)
        kind = "great event" if nb > 0.25 * N else ("large event" if nb > 30 else "small event")
        label.set_text(f"{kind}: {nb} blocks slipped, $\\mu = \\ln M = {np.log(M):.1f}$")
        label.set_color("#ffd166" if nb > 0.25 * N else INK)
        top = max(cnt.max(), 1)
        for b, v in zip(bars, cnt):
            b.set_width(max(v, 1e-3))
        axh.set_xlim(0.8, top * 1.5)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(events), interval=80, blit=False)
    save(anim, fig, "bk_cycle", fps=10, width=640, colors=96)


# ------------------------------------------------------------------ 3. an avalanche of the 2D BTW sandpile

def btw():
    L = 121
    pile = BTW(open_lattice((L, L)), seed=4)
    pile.fill_max()
    pile.drive(8 * L * L)
    # find a large avalanche starting near the centre
    rng = np.random.default_rng(1)
    while True:
        z0 = pile.z.copy()
        site = int(rng.integers(L * L))
        rr, cc = divmod(site, L)
        out = pile.add([site])
        if out["size"][0] > 25_000 and 0.25 * L < rr < 0.75 * L and 0.25 * L < cc < 0.75 * L:
            break
    z = z0.reshape(L, L).copy()
    z[rr, cc] += 1
    tops = np.zeros((L, L))
    frames = [(z.copy(), tops.copy())] * 10
    while (z >= 4).any():
        un = z >= 4
        z[un] -= 4
        tops[un] += 1
        z[1:, :] += un[:-1, :]
        z[:-1, :] += un[1:, :]
        z[:, 1:] += un[:, :-1]
        z[:, :-1] += un[:, 1:]
        frames.append((z.copy(), tops.copy()))
    step = max(1, len(frames) // 260)
    frames = frames[::step] + [frames[-1]] * 30
    fig = plt.figure(figsize=(8.0, 4.6))
    ax1 = fig.add_axes([0.03, 0.06, 0.45, 0.78])
    ax2 = fig.add_axes([0.52, 0.06, 0.45, 0.78])
    fig.text(0.03, 0.92, "Sandpile (Bak–Tang–Wiesenfeld): one grain, one avalanche", fontsize=13.5, color=INK)
    hcm = LinearSegmentedColormap.from_list("h", ["#0b1020", "#1f2a52", "#33508f", "#6f9ce6"])
    im1 = ax1.imshow(frames[0][0], cmap=hcm, vmin=0, vmax=4, interpolation="nearest")
    act = ax1.imshow(np.zeros((L, L, 4)), interpolation="nearest")
    im2 = ax2.imshow(frames[-1][1], cmap=GLOW, vmin=0, vmax=frames[-1][1].max(), interpolation="nearest")
    im2.set_data(np.zeros((L, L)))
    for a, t in ((ax1, "heights (grains per site); toppling sites glow"), (ax2, "number of topplings so far")):
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(t, fontsize=9, color=INK2)
    counter = ax2.text(0.02, 0.03, "", transform=ax2.transAxes, fontsize=9, color=INK)

    def draw(i):
        zi, ti = frames[i]
        im1.set_data(np.minimum(zi, 4))
        overlay = np.zeros((L, L, 4))
        hot = zi >= 4
        overlay[hot] = [1.0, 0.82, 0.4, 1.0]
        act.set_data(overlay)
        im2.set_data(ti)
        counter.set_text(f"{int(ti.sum()):,} topplings")
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=50, blit=False)
    save(anim, fig, "btw", fps=24)


if __name__ == "__main__":
    style()
    which = sys.argv[1:] or ["bk_event", "bk_cycle", "btw"]
    for w in which:
        globals()[w]()
