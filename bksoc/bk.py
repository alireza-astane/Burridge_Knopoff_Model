"""One-dimensional Burridge-Knopoff chain in the dimensionless form of Carlson & Langer.

Equation of motion (fixed rough surface, plate at position P(tau)):

    U_j'' = l^2 (U_{j+1} - 2 U_j + U_{j-1}) + (P - U_j) - phi(2 alpha U_j'),

with free ends (U_0 = U_1, U_{N+1} = U_N). The spring force on block j is
f_j = l^2 (...) + (P - U_j).

Friction laws (law argument):
    LAW_1991: stuck while f_j < 1 (any backward force is held, no backward slip);
              slipping friction (1 - sigma) / (1 + y / (1 - sigma)), y = 2 alpha U' > 0.
    LAW_1989: stuck while -1 < f_j < 1; slipping friction sgn(y) / (1 + |y|).

Driving: the plate moves at speed nu. For nu = 0 (the limit of Carlson et al. 1991) the
plate is frozen during events and, between events, advanced exactly until the next block
reaches the threshold. For nu > 0 the plate also moves during events, and between events
time jumps to the next threshold crossing.

Integration: classical RK4 over the set of slipping blocks only (stuck blocks are exactly
at rest). A slipping block sticks when its velocity crosses zero; its position is then
corrected to the zero crossing assuming constant acceleration within the step. A stuck
block starts to slip as soon as its force reaches the threshold; candidates are found with
a segment tree over the forces of stuck blocks, which is exact and O(log N).
"""

import numpy as np
from numba import njit

LAW_1991 = 0
LAW_1989 = 1

NEG = -1e300
POS = 1e300


# ---------------------------------------------------------------- segment tree (max and min)

@njit(cache=True)
def _tree_build(size, vals_max, vals_min, h):
    for i in range(2 * size):
        vals_max[i] = NEG
        vals_min[i] = POS
    for i in range(h.size):
        vals_max[size + i] = h[i]
        vals_min[size + i] = h[i]
    for p in range(size - 1, 0, -1):
        vals_max[p] = max(vals_max[2 * p], vals_max[2 * p + 1])
        vals_min[p] = min(vals_min[2 * p], vals_min[2 * p + 1])


@njit(cache=True)
def _tree_set(size, vals_max, vals_min, i, xmax, xmin):
    p = size + i
    vals_max[p] = xmax
    vals_min[p] = xmin
    p //= 2
    while p >= 1:
        a = vals_max[2 * p]
        b = vals_max[2 * p + 1]
        vals_max[p] = a if a > b else b
        a = vals_min[2 * p]
        b = vals_min[2 * p + 1]
        vals_min[p] = a if a < b else b
        p //= 2


@njit(cache=True)
def _tree_argmax(size, vals_max):
    p = 1
    while p < size:
        p = 2 * p if vals_max[2 * p] >= vals_max[2 * p + 1] else 2 * p + 1
    return p - size


@njit(cache=True)
def _tree_argmin(size, vals_min):
    p = 1
    while p < size:
        p = 2 * p if vals_min[2 * p] <= vals_min[2 * p + 1] else 2 * p + 1
    return p - size


# ---------------------------------------------------------------- physics helpers

@njit(cache=True)
def _elastic(U, j, N, l2):
    """Spring force without the plate term: l^2 * (discrete Laplacian, free ends) - U_j."""
    left = U[j - 1] if j > 0 else U[j]
    right = U[j + 1] if j < N - 1 else U[j]
    return l2 * (left - 2.0 * U[j] + right) - U[j]


@njit(cache=True)
def _friction(V, alpha, sigma, law):
    """Kinetic friction on a slipping block with velocity V (in units of F0)."""
    y = 2.0 * alpha * V
    if law == LAW_1991:
        if y < 0.0:
            y = 0.0
        return (1.0 - sigma) / (1.0 + y / (1.0 - sigma))
    if y > 0.0:
        return 1.0 / (1.0 + y)
    if y < 0.0:
        return -1.0 / (1.0 - y)
    return 0.0


@njit(cache=True)
def _energy(U, V, P, l2):
    N = U.size
    E = 0.0
    for j in range(N):
        E += 0.5 * V[j] ** 2 + 0.5 * (P - U[j]) ** 2
    for j in range(N - 1):
        E += 0.5 * l2 * (U[j + 1] - U[j]) ** 2
    return E


# ---------------------------------------------------------------- main loop

@njit(cache=True)
def steps_touched(j, dlist, nd):
    for q in range(nd):
        if dlist[q] == j:
            return True
    return False


@njit(cache=True)
def _simulate(U, state, n_events, l2, alpha, sigma, nu, law, h, max_steps, check_energy,
              out_M, out_n, out_T, out_j, out_P, out_t, out_W, out_dE, out_vmax, out_steps):
    """Run n_events events. state = [P, tau]; U is modified in place (all blocks stuck on entry)."""
    N = U.size
    P = state[0]
    tau = state[1]
    tol = 1e-12

    size = 1
    while size < N:
        size *= 2
    tmax = np.empty(2 * size)
    tmin = np.empty(2 * size)
    hval = np.empty(N)
    for j in range(N):
        hval[j] = _elastic(U, j, N, l2)
    _tree_build(size, tmax, tmin, hval)

    V = np.zeros(N)
    slipping = np.zeros(N, dtype=np.bool_)
    act = np.empty(N, dtype=np.int64)
    pos_in_act = -np.ones(N, dtype=np.int64)
    Us = U.copy()                       # stage positions (equal to U for stuck blocks)
    k = np.zeros((4, 2, N))             # RK4 slopes, indexed by position in `act`
    Vold = np.empty(N)
    Uold = np.empty(N)
    U0 = np.empty(N)
    touched = np.zeros(N, dtype=np.bool_)
    tlist = np.empty(N, dtype=np.int64)
    dirty = np.zeros(N, dtype=np.bool_)
    dlist = np.empty(N, dtype=np.int64)
    dirn = np.ones(N)                   # direction of motion at onset (+1 forward, -1 backward)
    hprev = np.empty(N)                 # elastic force at the start of the current step

    for e in range(n_events):
        # ---------------- load the plate until the next block reaches the threshold
        jstar = _tree_argmax(size, tmax)
        need = 1.0 - (tmax[1] + P)
        if need > 0.0:
            if nu > 0.0:
                tau += need / nu
            P += need
        E0 = _energy(U, V, P, l2) if check_energy else 0.0
        nact = 0
        nt = 0
        W = 0.0
        vmax = 0.0
        t_start = tau
        P_start = P
        steps = 0

        # activate the first block
        dirn[jstar] = 1.0
        act[nact] = jstar
        pos_in_act[jstar] = nact
        nact += 1
        slipping[jstar] = True
        V[jstar] = 0.0
        touched[jstar] = True
        tlist[nt] = jstar
        nt += 1
        U0[jstar] = U[jstar]
        _tree_set(size, tmax, tmin, jstar, NEG, POS)

        while nact > 0:
            steps += 1
            if steps > max_steps:
                break
            # ---- RK4 step for the slipping blocks (plate moves linearly in time)
            for c in range(nact):
                j = act[c]
                Uold[j] = U[j]
                Vold[j] = V[j]
            pw = 0.0
            for stage in range(4):
                if stage == 0:
                    dt_s = 0.0
                elif stage == 3:
                    dt_s = h
                else:
                    dt_s = 0.5 * h
                Pst = P + nu * dt_s
                if stage > 0:
                    for c in range(nact):
                        j = act[c]
                        Us[j] = Uold[j] + dt_s * k[stage - 1, 0, c]
                wgt = 1.0 if (stage == 0 or stage == 3) else 2.0
                for c in range(nact):
                    j = act[c]
                    vj = Vold[j] + dt_s * k[stage - 1, 1, c] if stage > 0 else Vold[j]
                    if law == LAW_1989 and vj == 0.0:
                        fr = dirn[j]            # friction starts at its static maximum
                    else:
                        fr = _friction(vj, alpha, sigma, law)
                    k[stage, 0, c] = vj
                    k[stage, 1, c] = _elastic(Us, j, N, l2) + Pst - fr
                    pw += wgt * fr * vj
            W += h * pw / 6.0
            for c in range(nact):
                j = act[c]
                U[j] = Uold[j] + h * (k[0, 0, c] + 2 * k[1, 0, c] + 2 * k[2, 0, c] + k[3, 0, c]) / 6.0
                V[j] = Vold[j] + h * (k[0, 1, c] + 2 * k[1, 1, c] + 2 * k[2, 1, c] + k[3, 1, c]) / 6.0
                Us[j] = U[j]
            P += nu * h
            tau += h

            # ---- blocks whose force changed: slipping blocks and their neighbours
            nd = 0
            for c in range(nact):
                j = act[c]
                if V[j] > vmax:
                    vmax = V[j]
                for m in (j - 1, j, j + 1):
                    if 0 <= m < N and not dirty[m]:
                        dirty[m] = True
                        dlist[nd] = m
                        nd += 1

            # ---- stopping: velocity crosses zero
            c = 0
            while c < nact:
                j = act[c]
                stop = False
                if law == LAW_1991:
                    stop = V[j] <= 0.0
                else:
                    if Vold[j] != 0.0 and Vold[j] * V[j] <= 0.0:
                        fj = _elastic(U, j, N, l2) + P
                        stop = abs(fj) < 1.0
                    elif Vold[j] == 0.0 and V[j] * dirn[j] <= 0.0:
                        stop = True
                if stop:
                    if Vold[j] != 0.0:
                        # stop at the zero crossing of the velocity (constant acceleration in the step)
                        theta = Vold[j] / (Vold[j] - V[j])
                        U[j] = Uold[j] + 0.5 * Vold[j] * theta * h
                    else:
                        U[j] = Uold[j]          # a block that never got going does not move
                    Us[j] = U[j]
                    V[j] = 0.0
                    slipping[j] = False
                    # remove from active list (swap with last)
                    last = act[nact - 1]
                    act[c] = last
                    pos_in_act[last] = c
                    for s_ in range(4):
                        k[s_, 0, c] = k[s_, 0, nact - 1]
                        k[s_, 1, c] = k[s_, 1, nact - 1]
                    pos_in_act[j] = -1
                    nact -= 1
                else:
                    c += 1

            # ---- update the tree for every block whose force changed
            # (slipping blocks are already stored as NEG/POS and need no update)
            for q in range(nd):
                m = dlist[q]
                dirty[m] = False
                hprev[m] = hval[m]
                if not slipping[m]:
                    hval[m] = _elastic(U, m, N, l2)
                    _tree_set(size, tmax, tmin, m, hval[m], hval[m] if law == LAW_1989 else POS)

            # ---- onsets: stuck blocks whose force reached the threshold
            while tmax[1] + P >= 1.0 - tol or (law == LAW_1989 and tmin[1] + P <= -1.0 + tol):
                if tmax[1] + P >= 1.0 - tol:
                    j = _tree_argmax(size, tmax)
                    dirn[j] = 1.0
                else:
                    j = _tree_argmin(size, tmin)
                    dirn[j] = -1.0
                if not touched[j]:
                    touched[j] = True
                    tlist[nt] = j
                    nt += 1
                    U0[j] = U[j]
                # The threshold was crossed during the step. Interpolate the crossing time from
                # the force at the start and the end of the step, and start the block with the
                # velocity and displacement it gained since then (keeps onsets 2nd-order accurate).
                fnow = hval[j] + P
                fold = hprev[j] + P - nu * h if steps_touched(j, dlist, nd) else fnow - nu * h
                thr = dirn[j]
                if (fnow - fold) * thr > 0.0:
                    theta = (thr - fold) / (fnow - fold)
                    if theta < 0.0:
                        theta = 0.0
                    rem = (1.0 - theta) * h
                    phi0 = (1.0 - sigma) if law == LAW_1991 else 1.0
                    acc = 0.5 * (thr + fnow) - thr * phi0
                    V[j] = acc * rem
                    U[j] += 0.5 * acc * rem * rem
                    Us[j] = U[j]
                else:
                    V[j] = 0.0
                slipping[j] = True
                act[nact] = j
                pos_in_act[j] = nact
                nact += 1
                _tree_set(size, tmax, tmin, j, NEG, POS)

        # ---------------- record the event
        M = 0.0
        for q in range(nt):
            j = tlist[q]
            M += U[j] - U0[j]
            touched[j] = False
        out_M[e] = M
        out_n[e] = nt
        out_T[e] = tau - t_start
        out_j[e] = jstar
        out_P[e] = P_start
        out_t[e] = t_start
        out_W[e] = W
        out_vmax[e] = vmax
        out_steps[e] = steps
        if check_energy:
            out_dE[e] = E0 - _energy(U, V, P, l2)
        else:
            out_dE[e] = 0.0
    state[0] = P
    state[1] = tau


class BKChain:
    """Burridge-Knopoff chain (dimensionless units of Carlson & Langer).

    Parameters
    ----------
    N : number of blocks
    l : stiffness ratio, l^2 = k_c / k_p
    alpha : velocity-weakening parameter
    sigma : friction drop at the onset of slip (LAW_1991)
    nu : loading speed (0 = event-driven limit of Carlson et al. 1991)
    law : LAW_1991 or LAW_1989
    h : time step; default 0.1 / sqrt(4 l^2 + 1) (0.1 / the highest frequency of the chain)
    """

    def __init__(self, N, l, alpha, sigma=0.01, nu=0.0, law=LAW_1991, h=None, seed=None, U=None):
        self.N, self.l, self.alpha, self.sigma, self.nu, self.law = N, float(l), float(alpha), float(sigma), float(nu), law
        self.h = h if h is not None else 0.1 / np.sqrt(4 * l * l + 1)
        self.rng = np.random.default_rng(seed)
        self.U = (1e-2 * self.rng.random(N)) if U is None else np.array(U, dtype=float)
        self.state = np.array([0.0, 0.0])          # plate position P, time tau
        # start with the plate just loading the most loaded block to threshold
        f = np.array([_elastic(self.U, j, N, self.l**2) for j in range(N)])
        self.state[0] = -f.max()                # plate placed so that the largest force is 0
        if law == LAW_1989 and nu <= 0:
            raise ValueError("the 1989 friction law needs nu > 0 (events cannot start with sigma = 0)")

    @property
    def P(self):
        return self.state[0]

    @property
    def tau(self):
        return self.state[1]

    def forces(self):
        l2 = self.l**2
        return np.array([_elastic(self.U, j, self.N, l2) for j in range(self.N)]) + self.P

    def energy(self):
        return _energy(self.U, np.zeros(self.N), self.P, self.l**2)

    def run(self, n_events, max_steps=10**9, check_energy=False):
        out = dict(moment=np.empty(n_events), blocks=np.empty(n_events, dtype=np.int64),
                   duration=np.empty(n_events), epicenter=np.empty(n_events, dtype=np.int64),
                   plate=np.empty(n_events), time=np.empty(n_events), dissipated=np.empty(n_events),
                   energy_drop=np.empty(n_events), vmax=np.empty(n_events),
                   steps=np.empty(n_events, dtype=np.int64))
        _simulate(self.U, self.state, n_events, self.l**2, self.alpha, self.sigma, self.nu, self.law,
                  self.h, max_steps, check_energy,
                  out["moment"], out["blocks"], out["duration"], out["epicenter"], out["plate"],
                  out["time"], out["dissipated"], out["energy_drop"], out["vmax"], out["steps"])
        return out
