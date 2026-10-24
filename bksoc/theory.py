"""Exact results for Abelian sandpiles, used both in the book and as test oracles."""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def toppling_matrix(nbr):
    """Dhar's toppling matrix Delta (N x N, sparse): Delta_ii = k, Delta_ij = -1 per bond."""
    N, k = nbr.shape
    D = lil_matrix((N, N))
    for i in range(N):
        D[i, i] = k
        for m in nbr[i]:
            if m >= 0:
                D[i, m] -= 1
    return D.tocsc()


def mean_topplings_per_site(nbr):
    """g_j = sum_i G_ij with G = Delta^{-1}: the mean avalanche size for a grain added at j.

    In the stationary state <n_ij> = G_ij exactly (Dhar 1990), so the mean size
    for uniform seeding is g.mean().
    """
    D = toppling_matrix(nbr)
    return spsolve(D.T, np.ones(nbr.shape[0]))


# ---------------------------------------------------------------- 1D BTW (k = 2)

def btw1d_size(L, l, j):
    """Exact number of topplings in the 1D BTW chain of length L.

    The recurrent configuration with its single zero at site l (l = 0: no zero) gets
    a grain at site j (1-based). The zero moves to l' = (l - j) mod (L+1) and
    s = g(j) + g(l') - g(l), with g(m) = m (L+1-m) / 2.
    """
    l = np.asarray(l)
    j = np.asarray(j)
    g2 = lambda m: m * (L + 1 - m)          # 2 g(m); g itself can be a half-integer
    lp = np.mod(l - j, L + 1)
    return (g2(j) + g2(lp) - g2(l)) // 2, lp


def btw1d_all_sizes(L):
    """Sizes for all (L+1) x L equally likely (configuration, seed) pairs."""
    l, j = np.meshgrid(np.arange(L + 1), np.arange(1, L + 1), indexing="ij")
    s, _ = btw1d_size(L, l, j)
    return s.ravel()


def btw1d_moments(L):
    """<s> and <s^2> for uniform seeding (Ruelle & Sen 1992)."""
    m1 = (L + 1) * (L + 2) / 12
    m2 = (L + 1) * (L + 2) * (2 * L**2 + 4 * L + 9) / 180
    return m1, m2


def btw1d_scaled_density(y):
    """Large-L density of Y = s/L^2 (excluding the delta peak at 0); support (0, 1/4)."""
    y = np.asarray(y, dtype=float)
    r = np.sqrt(np.clip(1 - 4 * y, 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 2 * np.log((1 + r) / (1 - r))
    return np.where((y > 0) & (y < 0.25), f, 0.0)


# ---------------------------------------------------------------- 2D BTW (k = 4)

# Single-site height probabilities in the infinite-lattice stationary state, heights
# 0..3 (= 1..4 in the convention of the literature). P0 is exact (Majumdar & Dhar
# 1991); the others follow Priezzhev (1994) and Jeng, Piroux & Ruelle (2006); the
# mean height 2.125 (= 25/8 - 1) is exact.
BTW2D_P0 = 2 / np.pi**2 - 4 / np.pi**3
BTW2D_HEIGHT_PROBS = np.array([BTW2D_P0, 0.1739, 0.3063, 0.4462])
BTW2D_MEAN_HEIGHT = 2.125
