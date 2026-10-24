"""Bak-Tang-Wiesenfeld (BTW) sandpile on an arbitrary lattice with open boundaries.

Conventions
-----------
* A lattice is described by a neighbour table ``nbr`` of shape (N, k): row i lists
  the k neighbours of site i; an entry -1 means "the sink" (the grain leaves the
  system). Every site has exactly k slots, so boundary sites have some -1 entries.
* Heights z_i are integers. A site is unstable when z_i >= k. Toppling removes k
  grains and sends one to each neighbour slot (grains sent to -1 are dissipated).
  The stable heights are therefore 0..k-1 (in 1D: {0,1}; in 2D: {0,1,2,3}).
* Relaxation uses the parallel (synchronous) update: in each generation every
  site that is unstable at the start of the generation topples once. Thanks to the
  Abelian property the size, area and final configuration do not depend on this
  choice; the duration (number of generations) does, and is defined by it.

Measured per avalanche (one added grain):
    size s   = total number of topplings
    area a   = number of distinct sites that toppled
    duration = number of parallel generations
    lost     = number of grains that left through the boundary
"""

import numpy as np
from numba import njit


def open_lattice(shape):
    """Neighbour table of a hypercubic lattice with open (absorbing) boundaries.

    shape = (L,) gives a chain, shape = (L, L) a square lattice, and so on.
    Sites are numbered in C order (np.ravel_multi_index).
    """
    shape = tuple(int(n) for n in shape)
    d = len(shape)
    N = int(np.prod(shape))
    idx = np.arange(N).reshape(shape)
    nbr = np.full((N, 2 * d), -1, dtype=np.int64)
    for axis in range(d):
        for slot, step in ((2 * axis, -1), (2 * axis + 1, +1)):
            shifted = np.full(shape, -1, dtype=np.int64)
            src = [slice(None)] * d
            dst = [slice(None)] * d
            if step == +1:   # neighbour at index+1
                dst[axis], src[axis] = slice(0, -1), slice(1, None)
            else:            # neighbour at index-1
                dst[axis], src[axis] = slice(1, None), slice(0, -1)
            shifted[tuple(dst)] = idx[tuple(src)]
            nbr[:, slot] = shifted.ravel()
    return nbr


@njit(cache=True)
def _relax(z, nbr, start, stamp, mark, cur, nxt, queued, count):
    """Relax the configuration after site `start` was perturbed.

    Returns (size, area, duration, lost). `mark` holds, for each site, the stamp
    of the last avalanche in which it toppled (used to count the area without
    clearing an array every avalanche). `cur`, `nxt` are work buffers of length N
    and `queued` a boolean buffer of length N that is all False on entry and exit.
    If `count` has length N, the number of topplings of each site is added to it
    (pass an empty array to skip this bookkeeping).
    """
    k = nbr.shape[1]
    record = count.size > 0
    if z[start] < k:
        return 0, 0, 0, 0
    size = 0
    area = 0
    duration = 0
    lost = 0
    ncur = 1
    cur[0] = start
    while ncur > 0:
        duration += 1
        # phase 1: every unstable site topples once
        for c in range(ncur):
            j = cur[c]
            z[j] -= k
            size += 1
            if record:
                count[j] += 1
            if mark[j] != stamp:
                mark[j] = stamp
                area += 1
        # phase 2: distribute grains; collect candidate sites for the next generation
        nn = 0
        for c in range(ncur):
            j = cur[c]
            if z[j] >= k and not queued[j]:
                queued[j] = True
                nxt[nn] = j
                nn += 1
            for q in range(k):
                m = nbr[j, q]
                if m < 0:
                    lost += 1
                else:
                    z[m] += 1
                    if z[m] >= k and not queued[m]:
                        queued[m] = True
                        nxt[nn] = m
                        nn += 1
        # z only grows in phase 2, so every site queued is still unstable now
        for c in range(nn):
            cur[c] = nxt[c]
            queued[nxt[c]] = False
        ncur = nn
    return size, area, duration, lost


@njit(cache=True)
def _drive(z, nbr, sites, stamp0, mark, sizes, areas, durations, losts, count):
    N = z.size
    cur = np.empty(N, dtype=np.int64)
    nxt = np.empty(N, dtype=np.int64)
    queued = np.zeros(N, dtype=np.bool_)
    for e in range(sites.size):
        i = sites[e]
        z[i] += 1
        s, a, t, l = _relax(z, nbr, i, stamp0 + e + 1, mark, cur, nxt, queued, count)
        sizes[e] = s
        areas[e] = a
        durations[e] = t
        losts[e] = l


class BTW:
    """BTW sandpile on the lattice described by `nbr` (see `open_lattice`).

    Example
    -------
    >>> pile = BTW(open_lattice((64, 64)), seed=1)
    >>> pile.fill_max()                 # start from the recurrent state z = k-1
    >>> pile.drive(10_000)              # burn-in
    >>> av = pile.drive(100_000)        # dict of arrays: size, area, duration, lost
    """

    def __init__(self, nbr, seed=None, z=None):
        self.nbr = np.ascontiguousarray(nbr, dtype=np.int64)
        self.N, self.k = self.nbr.shape
        self.z = np.zeros(self.N, dtype=np.int64) if z is None else np.array(z, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self._mark = np.zeros(self.N, dtype=np.int64)
        self._stamp = 0
        self.grains_added = 0
        self.grains_lost = 0

    def fill_max(self):
        """Set every site to k-1: the maximal stable configuration (always recurrent)."""
        self.z[:] = self.k - 1

    def add(self, sites, record=False):
        """Add one grain at each site in `sites` (in order), relaxing after each one.

        With record=True the result also holds "topplings": the number of times each
        site toppled, summed over all the added grains.
        """
        sites = np.ascontiguousarray(sites, dtype=np.int64)
        count = np.zeros(self.N if record else 0, dtype=np.int64)
        n = sites.size
        out = {key: np.empty(n, dtype=np.int64) for key in ("size", "area", "duration", "lost")}
        _drive(self.z, self.nbr, sites, self._stamp, self._mark,
               out["size"], out["area"], out["duration"], out["lost"], count)
        if record:
            out["topplings"] = count
        self._stamp += n
        self.grains_added += n
        self.grains_lost += int(out["lost"].sum())
        return out

    def drive(self, n, probs=None):
        """Add n grains at random sites (uniform, or with probabilities `probs`)."""
        if probs is None:
            sites = self.rng.integers(0, self.N, size=n)
        else:
            sites = self.rng.choice(self.N, size=n, p=probs)
        return self.add(sites)
