"""Shared matplotlib style for the book and report figures (static, print = light mode)."""

import matplotlib as mpl
import numpy as np

INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e4e3df"
SURFACE = "#ffffff"

# categorical slots, fixed order (validated reference palette, light mode)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# sequential blue, for ordered families such as system sizes (ordinal: start at step 250)
BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]


def ordinal(n):
    """n ordered colours from the blue ramp, light (small) to dark (large)."""
    idx = np.linspace(0, len(BLUES) - 1, n).round().astype(int)
    return [BLUES[i] for i in idx]


def _register_pagella():
    """Make TeX Gyre Pagella (matches the book's newpx font) visible to matplotlib if TeX Live has it."""
    import glob
    from matplotlib import font_manager
    for path in glob.glob("/usr/local/texlive/*/texmf-dist/fonts/opentype/public/tex-gyre/texgyrepagella-*.otf"):
        font_manager.fontManager.addfont(path)


def use():
    _register_pagella()
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": SURFACE,
        "font.family": "serif",
        "font.serif": ["TeX Gyre Pagella", "Palatino", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=C),
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "text.color": INK,
    })
