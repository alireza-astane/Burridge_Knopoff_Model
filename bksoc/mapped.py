"""Sandpile (cellular-automaton) limit of the Burridge-Knopoff chain.

The state is the spring force f_j on each block (BK units, threshold 1). The plate drives all
blocks uniformly: between avalanches every f_j is raised by the same amount until the largest
reaches 1. A block with f_j >= 1 "topples": it slips by Delta_j, which lowers its own force by
omega_j^2 Delta_j and raises the force of each neighbour by l^2 Delta_j, exactly as a stuck block
in the BK chain would if it moved by Delta_j while its neighbours stayed put. Here
omega_j^2 = 2 l^2 + 1 in the bulk and l^2 + 1 at the free ends.

Two rules for the slip:
    RULE_FIXED: every toppling sheds the same force d, Delta_j = d / omega_j^2. Choosing d equal to
                the force shed in a one-block BK event makes isolated topplings identical to BK
                one-block events. Topplings commute (Abelian), like the BTW model.
    RULE_RESET: the toppling block drops to a fixed residual force f_r, Delta_j = (f_j - f_r) / omega_j^2
                (Olami-Feder-Christensen). Not Abelian; relaxation uses the parallel update.

Of the force shed by a bulk toppling, a fraction 2 l^2 / (2 l^2 + 1) goes to the neighbours and
a fraction eps = 1 / (2 l^2 + 1) is lost into the pulling spring.

Threshold noise (optional, width w > 0): every block has its own static threshold theta_j, and
after each toppling the block draws a new one, theta_j = 1 + eta with eta uniform in [-w/2, w/2].
This models heterogeneous friction and breaks the exact periodicity of the deterministic model.

Recorded per avalanche: number of topplings s, number of distinct blocks a, moment M = sum of
slips, duration T (parallel generations), epicenter, plate position.
"""

import numpy as np
from numba import njit

from .bk import _tree_build, _tree_set, _tree_argmax, NEG, POS

RULE_FIXED = 0
RULE_RESET = 1


@njit(cache=True)
def _omega2(j, N, l2):
    if N == 1:
        return 1.0
    if j == 0 or j == N - 1:
        return l2 + 1.0
    return 2.0 * l2 + 1.0



@njit(cache=True)
def _simulate(g, theta, state, n_av, l2, rule, d, f_res, w, seed,
              out_s, out_a, out_M, out_T, out_j, out_P):
    """g: forces relative to the plate (f = g + P); theta: thresholds; state = [P].

    The tree stores g_j - theta_j, so block j is unstable when tree value + P >= 0.
    """
    N = g.size
    if w > 0.0:
        np.random.seed(seed)
    P = state[0]
    size = 1
    while size < N:
        size *= 2
    tmax = np.empty(2 * size)
    tmin = np.empty(2 * size)
    key = np.empty(N)
    for j in range(N):
        key[j] = g[j] - theta[j]
    _tree_build(size, tmax, tmin, key)
    cur = np.empty(N, dtype=np.int64)
    nxt = np.empty(N, dtype=np.int64)
    queued = np.zeros(N, dtype=np.bool_)
    seen = np.zeros(N, dtype=np.bool_)
    seen_list = np.empty(N, dtype=np.int64)
    drops = np.empty(N)
    tol = 1e-12
    off = 0.0                      # plate displacement folded into g during this call
    for e in range(n_av):
        if e % 1000 == 999 and P != 0.0:
            # keep g of order one: fold the plate position into the stored forces (rounding)
            for j in range(N):
                g[j] += P
                key[j] = g[j] - theta[j]
            _tree_build(size, tmax, tmin, key)
            off += P
            P = 0.0
        j0 = _tree_argmax(size, tmax)
        need = -(tmax[1] + P)
        if need > 0.0:
            P += need
        out_P[e] = P + off
        out_j[e] = j0
        s = 0
        a = 0
        M = 0.0
        T = 0
        ncur = 1
        cur[0] = j0
        queued[j0] = True
        while ncur > 0:
            T += 1
            # phase 1: slips of all sites unstable at the start of the generation
            for c in range(ncur):
                j = cur[c]
                w2 = _omega2(j, N, l2)
                if rule == RULE_FIXED:
                    drops[c] = d / w2
                else:
                    drops[c] = (g[j] + P - (theta[j] - 1.0 + f_res)) / w2
            nn = 0
            for c in range(ncur):
                j = cur[c]
                queued[j] = False
                delta = drops[c]
                w2 = _omega2(j, N, l2)
                g[j] -= w2 * delta
                if w > 0.0:
                    theta[j] = 1.0 + w * (np.random.random() - 0.5)
                M += delta
                s += 1
                if not seen[j]:
                    seen[j] = True
                    seen_list[a] = j
                    a += 1
                for m in (j - 1, j + 1):
                    if 0 <= m < N:
                        g[m] += l2 * delta
            # phase 2: collect the sites that are now unstable, update the tree
            for c in range(ncur):
                j = cur[c]
                for m in (j - 1, j, j + 1):
                    if 0 <= m < N:
                        _tree_set(size, tmax, tmin, m, g[m] - theta[m], POS)
                        if g[m] - theta[m] + P >= -tol and not queued[m]:
                            queued[m] = True
                            nxt[nn] = m
                            nn += 1
            for c in range(nn):
                cur[c] = nxt[c]
            ncur = nn
        for q in range(a):
            seen[seen_list[q]] = False
        out_s[e] = s
        out_a[e] = a
        out_M[e] = M
        out_T[e] = T
    state[0] = P
    state[1] += off


class MappedChain:
    """Sandpile limit of the BK chain (see module docstring).

    d: force shed per toppling (RULE_FIXED). Use `bk_one_block_drop(l, alpha, sigma)` to match the
       one-block event of the BK model.
    f_res: residual force after a toppling (RULE_RESET).
    """

    def __init__(self, N, l, rule=RULE_FIXED, d=0.02, f_res=0.98, seed=None, f=None, noise=0.0):
        self.N, self.l, self.rule, self.d, self.f_res = N, float(l), rule, float(d), float(f_res)
        self.noise = float(noise)
        rng = np.random.default_rng(seed)
        self._seed = int(rng.integers(2**31))
        self.theta = np.ones(N)
        if self.noise > 0:
            self.theta += self.noise * (rng.random(N) - 0.5)
        f = (rng.random(N) * 0.5) if f is None else np.array(f, dtype=float)
        self.state = np.array([0.0, 0.0])     # [plate position relative to g, offset folded into g]
        self._P_offset = 0.0       # plate displacement already folded into g (keeps g small)
        self.g = f.copy()          # f = g + state[0]

    @property
    def P(self):
        """Total plate displacement."""
        return self.state[0] + self.state[1] + self._P_offset

    def forces(self):
        return self.g + self.state[0]

    def run(self, n):
        out = dict(size=np.empty(n, dtype=np.int64), area=np.empty(n, dtype=np.int64),
                   moment=np.empty(n), duration=np.empty(n, dtype=np.int64),
                   epicenter=np.empty(n, dtype=np.int64), plate=np.empty(n))
        # fold the plate position into the stored forces so that g stays of order one (rounding)
        self.g += self.state[0]
        self._P_offset += self.state[0]
        self.state[0] = 0.0
        off_before = self.state[1]
        self._seed = (self._seed * 1103515245 + 12345) % 2**31
        _simulate(self.g, self.theta, self.state, n, self.l**2, self.rule, self.d, self.f_res,
                  self.noise, self._seed,
                  out["size"], out["area"], out["moment"], out["duration"], out["epicenter"], out["plate"])
        out["plate"] += self._P_offset + off_before
        return out


def bk_one_block_drop(l, alpha, sigma):
    """Force shed by an isolated slipping block in the BK chain, to linear order in the velocity.

    Delta f = omega_1^2 * delta U_1 = sigma (1 + exp(pi alpha / omega)), omega = sqrt(2 l^2 + 1 - alpha^2).
    """
    w = np.sqrt(2 * l * l + 1 - alpha**2)
    return sigma * (1 + np.exp(np.pi * alpha / w))

