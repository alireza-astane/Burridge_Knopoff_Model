"""One-dimensional Burridge-Knopoff chain in the dimensionless form of Carlson & Langer.

Equation of motion (fixed rough surface, plate at position P(tau)):

    U_j'' = l^2 (U_{j+1} - 2 U_j + U_{j-1}) + (P - U_j) - phi(2 alpha U_j'),

with free ends (U_0 = U_1, U_{N+1} = U_N). The spring force on block j is
f_j = l^2 (...) + (P - U_j).

Friction laws (law argument):
    LAW_1991: stuck while f_j < 1 (any backward force is held, no backward slip);
              slipping friction (1 - sigma) / (1 + y / (1 - sigma)), y = 2 alpha U' > 0.
    LAW_1989: stuck while -1 < f_j < 1; slipping friction sgn(y) / (1 + |y|).

Events: an event is a space-time connected cluster of slip. A block that starts to slip next to a
slipping neighbour joins that neighbour's event, otherwise it starts a new one; touching events merge.
For nu = 0 only one event exists at a time; for nu > 0 several events can coexist in different parts
of the chain, as in Carlson & Langer (1989).

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
def _record(slot, rec, tau, U, V, P, l2, check_energy, ev_M, ev_blocks, ev_t0, ev_P0, ev_j0, ev_W,
            ev_vmax, ev_steps, ev_E0, out_M, out_n, out_T, out_j, out_P, out_t, out_W, out_dE, out_vmax,
            out_steps):
    out_M[rec] = ev_M[slot]
    out_n[rec] = ev_blocks[slot]
    out_T[rec] = tau - ev_t0[slot]
    out_j[rec] = ev_j0[slot]
    out_P[rec] = ev_P0[slot]
    out_t[rec] = ev_t0[slot]
    out_W[rec] = ev_W[slot]
    out_vmax[rec] = ev_vmax[slot]
    out_steps[rec] = ev_steps[slot]
    out_dE[rec] = ev_E0[slot] - _energy(U, V, P, l2) if check_energy else 0.0


@njit(cache=True)
def _simulate(U, state, istate, V, slipping, act, dirn, U0, evslot, stamp, stopped_now,
              ev_M, ev_n, ev_blocks, ev_t0, ev_P0, ev_j0, ev_W, ev_vmax, ev_steps, ev_E0, ev_serial,
              ev_stepstamp, free_list, tmat,
              n_events, l2, alpha, sigma, nu, law, h, max_steps, check_energy,
              out_M, out_n, out_T, out_j, out_P, out_t, out_W, out_dE, out_vmax, out_steps):
    """Integrate until at least n_events events have been completed; returns the number recorded.

    An event is a space-time connected cluster of slip: a block that starts to slip while a
    neighbour is slipping (or stopped in the same time step) joins that neighbour's event,
    otherwise it starts a new event; events that touch are merged. With nu = 0 only one event can
    exist at a time. All state (velocities, slipping set, open events) persists between calls in
    the arrays passed in. state = [P, tau]; istate = [nact, nfree, serial counter, global step].
    tmat: (N, N) boolean matrix of the blocks touched by each open event (exact block counts when
    events can merge, i.e. nu > 0), or a (1, 1) dummy when nu = 0 (then stamps are exact).
    """
    use_mat = tmat.shape[0] == U.size
    N = U.size
    P = state[0]
    tau = state[1]
    nact = istate[0]
    nfree = istate[1]
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
    for j in range(N):
        if slipping[j]:
            _tree_set(size, tmax, tmin, j, NEG, POS)
        elif law != LAW_1989:
            _tree_set(size, tmax, tmin, j, hval[j], POS)

    Us = U.copy()
    k = np.zeros((4, 2, N))
    Vold = np.empty(N)
    Uold = np.empty(N)
    dirty = np.zeros(N, dtype=np.bool_)
    dlist = np.empty(N, dtype=np.int64)
    hprev = np.empty(N)
    pwb = np.empty(N)
    stop_list = np.empty(N, dtype=np.int64)
    stop_slot = np.empty(N, dtype=np.int64)     # event of each stopped block at the moment it stopped
    done_list = np.empty(N, dtype=np.int64)

    rec = 0
    steps_call = 0
    while rec < n_events and steps_call < max_steps:
        # ---------------- nothing slipping: load the plate until the next block reaches the threshold
        if nact == 0:
            jstar = _tree_argmax(size, tmax)
            need = 1.0 - (tmax[1] + P)
            if need > 0.0:
                if nu > 0.0:
                    tau += need / nu
                P += need
            slot = free_list[nfree - 1]
            nfree -= 1
            istate[2] += 1
            ev_serial[slot] = istate[2]
            ev_M[slot] = 0.0
            ev_n[slot] = 1
            ev_blocks[slot] = 1
            ev_t0[slot] = tau
            ev_P0[slot] = P
            ev_j0[slot] = jstar
            ev_W[slot] = 0.0
            ev_vmax[slot] = 0.0
            ev_steps[slot] = 0
            ev_E0[slot] = _energy(U, V, P, l2) if check_energy else 0.0
            stamp[jstar] = istate[2]
            if use_mat:
                tmat[slot, jstar] = True
            evslot[jstar] = slot
            dirn[jstar] = 1.0
            act[0] = jstar
            nact = 1
            slipping[jstar] = True
            V[jstar] = 0.0
            U0[jstar] = U[jstar]
            _tree_set(size, tmax, tmin, jstar, NEG, POS)

        steps_call += 1
        istate[3] += 1
        gstep = istate[3]
        # ---- RK4 step for the slipping blocks (plate moves linearly in time)
        for c in range(nact):
            j = act[c]
            Uold[j] = U[j]
            Vold[j] = V[j]
            pwb[c] = 0.0
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
                if law == LAW_1989:
                    # friction opposes the current direction of motion; if a stage velocity has
                    # crossed zero, evaluate it at zero velocity (the block is about to stop) rather
                    # than flipping its sign, which would give a spurious kick within the step
                    vv = vj * dirn[j]
                    if vv < 0.0:
                        vv = 0.0
                    fr = dirn[j] / (1.0 + 2.0 * alpha * vv)
                else:
                    fr = _friction(vj, alpha, sigma, law)
                k[stage, 0, c] = vj
                k[stage, 1, c] = _elastic(Us, j, N, l2) + Pst - fr
                pwb[c] += wgt * fr * vj
        for c in range(nact):
            j = act[c]
            U[j] = Uold[j] + h * (k[0, 0, c] + 2 * k[1, 0, c] + 2 * k[2, 0, c] + k[3, 0, c]) / 6.0
            V[j] = Vold[j] + h * (k[0, 1, c] + 2 * k[1, 1, c] + 2 * k[2, 1, c] + k[3, 1, c]) / 6.0
            Us[j] = U[j]
            slot = evslot[j]
            ev_W[slot] += h * pwb[c] / 6.0
            if V[j] > ev_vmax[slot]:
                ev_vmax[slot] = V[j]
            if ev_stepstamp[slot] != gstep:
                ev_stepstamp[slot] = gstep
                ev_steps[slot] += 1
        P += nu * h
        tau += h

        # ---- blocks whose force changed: slipping blocks and their neighbours
        nd = 0
        for c in range(nact):
            j = act[c]
            for m in (j - 1, j, j + 1):
                if 0 <= m < N and not dirty[m]:
                    dirty[m] = True
                    dlist[nd] = m
                    nd += 1

        # ---- stopping: velocity crosses zero
        nstop = 0
        c = 0
        while c < nact:
            j = act[c]
            stop = False
            if law == LAW_1991:
                stop = V[j] <= 0.0
            else:
                # 1989 law: every return of the velocity to zero ends the block's slip episode. If the
                # force still exceeds the static range, the block starts again at once (onset check
                # below) as a new episode, as in Carlson & Langer, who count every such oscillation
                # of a creeping block as a separate one-block event. The motion is the same as if
                # the block had kept slipping.
                if Vold[j] != 0.0 and Vold[j] * V[j] <= 0.0:
                    stop = True
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
                slot = evslot[j]
                ev_M[slot] += U[j] - U0[j]
                ev_n[slot] -= 1
                stopped_now[j] = True        # still counts as connected during this step's onsets
                stop_list[nstop] = j
                stop_slot[nstop] = slot
                nstop += 1
                last = act[nact - 1]
                act[c] = last
                for s_ in range(4):
                    k[s_, 0, c] = k[s_, 0, nact - 1]
                    k[s_, 1, c] = k[s_, 1, nact - 1]
                nact -= 1
            else:
                c += 1

        # ---- update the tree for every stuck block whose force changed
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
            # which event does it belong to? (neighbours slipping or stopped in this step)
            sl = -1
            sr = -1
            if j > 0 and (slipping[j - 1] or stopped_now[j - 1]):
                sl = evslot[j - 1]
            if j < N - 1 and (slipping[j + 1] or stopped_now[j + 1]):
                sr = evslot[j + 1]
            if sl < 0 and sr < 0:
                slot = free_list[nfree - 1]
                nfree -= 1
                istate[2] += 1
                ev_serial[slot] = istate[2]
                ev_M[slot] = 0.0
                ev_n[slot] = 0
                ev_blocks[slot] = 0
                ev_t0[slot] = tau
                ev_P0[slot] = P
                ev_j0[slot] = j
                ev_W[slot] = 0.0
                ev_vmax[slot] = 0.0
                ev_steps[slot] = 0
                ev_E0[slot] = _energy(U, V, P, l2) if check_energy else 0.0
            else:
                slot = sl if sl >= 0 else sr
                other = sr if (sl >= 0 and sr >= 0 and sr != sl) else -1
                if other >= 0:
                    # merge event `other` into `slot`
                    for c2 in range(nact):
                        if evslot[act[c2]] == other:
                            evslot[act[c2]] = slot
                    for q in range(nstop):
                        if evslot[stop_list[q]] == other:
                            evslot[stop_list[q]] = slot
                        if stop_slot[q] == other:
                            stop_slot[q] = slot
                    ev_M[slot] += ev_M[other]
                    ev_n[slot] += ev_n[other]
                    ev_blocks[slot] += ev_blocks[other]
                    ev_W[slot] += ev_W[other]
                    if ev_vmax[other] > ev_vmax[slot]:
                        ev_vmax[slot] = ev_vmax[other]
                    if ev_steps[other] > ev_steps[slot]:
                        ev_steps[slot] = ev_steps[other]
                    if ev_t0[other] < ev_t0[slot]:
                        ev_t0[slot] = ev_t0[other]
                        ev_P0[slot] = ev_P0[other]
                        ev_j0[slot] = ev_j0[other]
                        ev_E0[slot] = ev_E0[other]
                    # blocks already counted in `other` must not be counted again in `slot`
                    so = ev_serial[other]
                    for jj in range(N):
                        if stamp[jj] == so:
                            stamp[jj] = ev_serial[slot]
                    if use_mat:
                        cnt = 0
                        for jj in range(N):
                            if tmat[other, jj]:
                                tmat[slot, jj] = True
                                tmat[other, jj] = False
                            if tmat[slot, jj]:
                                cnt += 1
                        ev_blocks[slot] = cnt          # exact: size of the union of the two sets
                    ev_n[other] = 0
                    ev_serial[other] = 0
                    free_list[nfree] = other
                    nfree += 1
            evslot[j] = slot
            ev_n[slot] += 1
            if use_mat:
                if not tmat[slot, j]:
                    tmat[slot, j] = True
                    ev_blocks[slot] += 1
            elif stamp[j] != ev_serial[slot]:
                stamp[j] = ev_serial[slot]
                ev_blocks[slot] += 1
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
            stopped_now[j] = False
            slipping[j] = True
            act[nact] = j
            nact += 1
            _tree_set(size, tmax, tmin, j, NEG, POS)

        # ---- events whose last block stopped (and that nobody joined) are complete
        ndone = 0
        for q in range(nstop):
            j = stop_list[q]
            slot = stop_slot[q]           # the block may have restarted in a new event meanwhile
            if not slipping[j]:
                stopped_now[j] = False
                evslot[j] = -1
            else:
                stopped_now[j] = False
            if slot >= 0 and ev_n[slot] == 0 and ev_serial[slot] > 0:
                # avoid recording the same event twice when several of its blocks stopped
                dup = False
                for q2 in range(ndone):
                    if done_list[q2] == slot:
                        dup = True
                if not dup:
                    done_list[ndone] = slot
                    ndone += 1
        for q in range(ndone):
            slot = done_list[q]
            _record(slot, rec, tau, U, V, P, l2, check_energy, ev_M, ev_blocks, ev_t0, ev_P0, ev_j0, ev_W,
                    ev_vmax, ev_steps, ev_E0, out_M, out_n, out_T, out_j, out_P, out_t, out_W, out_dE,
                    out_vmax, out_steps)
            rec += 1
            if use_mat:
                for jj in range(N):
                    tmat[slot, jj] = False
            ev_serial[slot] = 0
            free_list[nfree] = slot
            nfree += 1

    state[0] = P
    state[1] = tau
    istate[0] = nact
    istate[1] = nfree
    return rec


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
        # persistent dynamical state (slipping blocks and open events survive between calls to run)
        self.V = np.zeros(N)
        self.slipping = np.zeros(N, dtype=np.bool_)
        self.act = np.empty(N, dtype=np.int64)
        self.dirn = np.ones(N)
        self.U0 = np.zeros(N)
        self.evslot = -np.ones(N, dtype=np.int64)
        self.stamp = np.zeros(N, dtype=np.int64)
        self.stopped_now = np.zeros(N, dtype=np.bool_)
        self.ev = dict(M=np.zeros(N), n=np.zeros(N, dtype=np.int64), blocks=np.zeros(N, dtype=np.int64),
                       t0=np.zeros(N), P0=np.zeros(N), j0=np.zeros(N, dtype=np.int64), W=np.zeros(N),
                       vmax=np.zeros(N), steps=np.zeros(N, dtype=np.int64), E0=np.zeros(N),
                       serial=np.zeros(N, dtype=np.int64), stepstamp=np.zeros(N, dtype=np.int64))
        self.free_list = np.arange(N, dtype=np.int64)
        self.tmat = np.zeros((N, N) if nu > 0 else (1, 1), dtype=np.bool_)
        self.istate = np.array([0, N, 0, 0], dtype=np.int64)    # nact, nfree, serial, global step

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

    @property
    def n_slipping(self):
        return int(self.istate[0])

    def run(self, n_events, max_steps=10**12, check_energy=False):
        """Run until at least n_events events are complete; returns a dict of arrays (one entry per event).

        Slightly more than n_events may be returned when several events end in the same time step.
        """
        cap = n_events + self.N
        out = dict(moment=np.empty(cap), blocks=np.empty(cap, dtype=np.int64),
                   duration=np.empty(cap), epicenter=np.empty(cap, dtype=np.int64),
                   plate=np.empty(cap), time=np.empty(cap), dissipated=np.empty(cap),
                   energy_drop=np.empty(cap), vmax=np.empty(cap), steps=np.empty(cap, dtype=np.int64))
        e = self.ev
        n = _simulate(self.U, self.state, self.istate, self.V, self.slipping, self.act, self.dirn, self.U0,
                      self.evslot, self.stamp, self.stopped_now,
                      e["M"], e["n"], e["blocks"], e["t0"], e["P0"], e["j0"], e["W"], e["vmax"], e["steps"],
                      e["E0"], e["serial"], e["stepstamp"], self.free_list, self.tmat,
                      n_events, self.l**2, self.alpha, self.sigma, self.nu, self.law,
                      self.h, max_steps, check_energy,
                      out["moment"], out["blocks"], out["duration"], out["epicenter"], out["plate"],
                      out["time"], out["dissipated"], out["energy_drop"], out["vmax"], out["steps"])
        return {key: val[:n] for key, val in out.items()}
