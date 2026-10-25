"""Tests of the Burridge-Knopoff solver (bksoc.bk).

The central test compares single events with an independent reference solution: SciPy's
adaptive integrator with exact event location for every stick and slip transition.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from bksoc.bk import BKChain, LAW_1991, LAW_1989, _elastic


# ------------------------------------------------------------------ reference solver

def phi91(y, sigma):
    return (1 - sigma) / (1 + max(y, 0.0) / (1 - sigma))


def forces(U, P, l2):
    left = np.concatenate([[U[0]], U[:-1]])
    right = np.concatenate([U[1:], [U[-1]]])
    return l2 * (left - 2 * U + right) + P - U


def reference_event(U, P, l, alpha, sigma, nu=0.0, tol=1e-11):
    """One event of the 1991 model, integrated with exact event location. Returns final U, P."""
    U = U.astype(float).copy()
    N = U.size
    l2 = l * l
    V = np.zeros(N)
    f = forces(U, P, l2)
    S = {int(np.argmax(f))}
    t = 0.0
    while S:
        idx = sorted(S)
        stuck = [j for j in range(N) if j not in S]

        def rhs(tt, y):
            u, v = y[:N], y[N:]
            ff = forces(u, P + nu * (tt - t), l2)
            du = np.zeros(N)
            dv = np.zeros(N)
            for j in idx:
                du[j] = v[j]
                dv[j] = ff[j] - phi91(2 * alpha * v[j], sigma)
            return np.concatenate([du, dv])

        evs = []
        for j in idx:
            def ev(tt, y, j=j):
                return y[N + j] if tt > t + 1e-9 else 1.0
            ev.terminal, ev.direction = True, -1
            evs.append(ev)
        for j in stuck:
            def ev(tt, y, j=j):
                return forces(y[:N], P + nu * (tt - t), l2)[j] - 1.0
            ev.terminal, ev.direction = True, 1
            evs.append(ev)
        sol = solve_ivp(rhs, (t, t + 1e4), np.concatenate([U, V]), events=evs, rtol=tol, atol=tol,
                        method="DOP853")
        U, V = sol.y[:N, -1].copy(), sol.y[N:, -1].copy()
        t_new = sol.t[-1]
        P += nu * (t_new - t)
        t = t_new
        hit = [k for k, te in enumerate(sol.t_events) if len(te)]
        k = hit[0]
        if k < len(idx):
            j = idx[k]
            V[j] = 0.0
            S.remove(j)
        else:
            j = stuck[k - len(idx)]
            S.add(j)
    return U, P


def chain_at_threshold(N, l, rng, spread=0.3):
    """Random stuck configuration, with the plate placed so that the most loaded block is at threshold."""
    U = spread * rng.random(N)
    P = 1.0 - (forces(U, 0.0, l * l)).max()
    return U, P


def run_one(U, P, l, alpha, sigma, h, nu=0.0):
    c = BKChain(U.size, l, alpha, sigma, nu=nu, h=h, U=U.copy())
    c.state[0] = P - 1e-13        # the solver loads the plate up to the threshold itself
    r = c.run(1)
    return c.U.copy(), c.P, r


# ------------------------------------------------------------------ single block in a stuck chain

def test_one_block_event_without_velocity_weakening():
    """alpha -> 0: delta U = 2 sigma / (2 l^2 + 1) and duration pi / omega_1 (exact)."""
    l, sigma = 6.0, 0.01
    U = np.zeros(21)
    U[10] = -0.5
    c = BKChain(21, l, 1e-9, sigma, U=U, h=1e-3)
    r = c.run(1)
    w1 = np.sqrt(2 * l * l + 1)
    assert r["blocks"][0] == 1
    assert r["moment"][0] == pytest.approx(2 * sigma / w1**2, rel=1e-6)
    assert r["duration"][0] == pytest.approx(np.pi / w1, abs=2e-3)


def test_one_block_event_with_velocity_weakening():
    """With alpha > 0 the slip is (sigma / w1^2)(1 + exp(pi alpha / w)) to linear order in the velocity."""
    l, sigma, alpha = 6.0, 0.01, 3.0
    U = np.zeros(21)
    U[10] = -0.5
    r = BKChain(21, l, alpha, sigma, U=U, h=1e-4).run(1)
    w1 = np.sqrt(2 * l * l + 1)
    w = np.sqrt(w1**2 - alpha**2)
    linear = sigma / w1**2 * (1 + np.exp(np.pi * alpha / w))
    assert r["moment"][0] == pytest.approx(linear, rel=0.02)
    # exact single-degree-of-freedom solution with the full nonlinear friction law
    sol = solve_ivp(lambda t, y: [y[1], 1 - w1**2 * y[0] - phi91(2 * alpha * y[1], sigma)],
                    (0, 1), [0, 0], rtol=1e-12, atol=1e-14,
                    events=lambda t, y: y[1] if t > 1e-6 else 1.0, dense_output=True)
    assert r["moment"][0] == pytest.approx(sol.y_events[0][0][0], rel=1e-4)


# ------------------------------------------------------------------ comparison with the reference solver

@pytest.mark.parametrize("seed,spread", [(0, 0.003), (1, 0.003), (0, 0.01), (3, 0.01), (0, 0.3), (1, 0.3)])
def test_event_matches_reference_solver(seed, spread):
    """Small spreads give events that cascade through the whole chain; large spreads one-block events."""
    rng = np.random.default_rng(seed)
    l, alpha, sigma = 3.0, 2.0, 0.05
    U, P = chain_at_threshold(10, l, rng, spread)
    U_ref, _ = reference_event(U, P, l, alpha, sigma)
    U_sim, _, r = run_one(U, P, l, alpha, sigma, h=1e-3)
    if spread < 0.1:
        assert r["blocks"][0] == 10                       # make sure the cascade is really tested
    assert np.max(np.abs(U_sim - U_ref)) < 2e-6
    assert r["blocks"][0] == np.count_nonzero(U_ref - U > 1e-12)


def test_error_decreases_quadratically_with_step_size():
    rng = np.random.default_rng(0)
    l, alpha, sigma = 3.0, 2.0, 0.05
    U, P = chain_at_threshold(10, l, rng, 0.003)
    U_ref, _ = reference_event(U, P, l, alpha, sigma)
    errs = np.array([np.max(np.abs(run_one(U, P, l, alpha, sigma, h)[0] - U_ref))
                     for h in (4e-3, 1e-3, 2.5e-4)])
    assert np.all(np.diff(errs) < 0)
    assert errs[0] / errs[2] > 16 * 2        # better than first order over a factor 16 in h


def test_default_step_is_accurate_for_large_events():
    rng = np.random.default_rng(0)
    l, alpha, sigma = 6.0, 3.0, 0.01
    U, P = chain_at_threshold(12, l, rng, 0.00075)
    U_ref, _ = reference_event(U, P, l, alpha, sigma)
    c = BKChain(12, l, alpha, sigma, U=U.copy())
    c.state[0] = P - 1e-13
    r = c.run(1)
    assert r["blocks"][0] == 12
    assert np.max(np.abs(c.U - U_ref)) < 1e-4 * np.max(U_ref - U)


def test_event_with_moving_plate_matches_reference():
    rng = np.random.default_rng(0)
    l, alpha, sigma, nu = 3.0, 2.0, 0.05, 0.05
    U, P = chain_at_threshold(8, l, rng, 0.003)
    U_ref, P_ref = reference_event(U, P, l, alpha, sigma, nu=nu)
    U_sim, P_sim, r = run_one(U, P, l, alpha, sigma, h=1e-3, nu=nu)
    assert r["blocks"][0] > 1
    assert np.max(np.abs(U_sim - U_ref)) < 1e-5
    assert P_sim == pytest.approx(P_ref, abs=1e-3)


# ------------------------------------------------------------------ conservation laws and invariants

def test_energy_balance():
    """With the plate at rest, the elastic + kinetic energy lost in an event equals the friction work."""
    c = BKChain(128, 6.0, 3.0, 0.01, seed=1, h=2e-3)
    c.run(2000)
    r = c.run(3000, check_energy=True)
    big = r["energy_drop"] > 1e-6
    rel = np.abs(r["energy_drop"][big] - r["dissipated"][big]) / r["dissipated"][big]
    assert np.median(rel) < 1e-4 and rel.max() < 1e-2


def test_sum_rule_average_slip_equals_plate_motion():
    c = BKChain(256, 6.0, 3.0, 0.01, seed=2)
    c.run(5000)
    P0, Ubar0 = c.P, c.U.mean()
    r = c.run(50_000)
    assert r["moment"].sum() / c.N == pytest.approx(c.U.mean() - Ubar0, rel=1e-9)
    assert abs(r["moment"].sum() / c.N - (c.P - P0)) < 1.0      # bounded lag behind the plate
    assert c.P - P0 > 20


def test_no_backward_slip_and_all_blocks_stable_after_events():
    c = BKChain(64, 4.0, 2.5, 0.02, seed=3)
    c.run(500)
    for _ in range(300):
        U_before = c.U.copy()
        c.run(1)
        assert np.all(c.U - U_before >= -1e-12)
        assert c.forces().max() <= 1.0 + 1e-9


def test_mirror_symmetry():
    rng = np.random.default_rng(5)
    U, P = chain_at_threshold(12, 3.0, rng, 0.01)
    a, _, ra = run_one(U, P, 3.0, 2.0, 0.05, h=1e-3)
    b, _, rb = run_one(U[::-1].copy(), P, 3.0, 2.0, 0.05, h=1e-3)
    assert np.allclose(a, b[::-1], atol=1e-12)
    assert ra["moment"][0] == pytest.approx(rb["moment"][0], rel=1e-12)


def test_translation_invariance():
    """Shifting all blocks and the plate by the same amount changes nothing.

    Only the first events are compared: the dynamics is chaotic, so round-off differences
    of order 1e-15 eventually grow to visible size.
    """
    c1 = BKChain(64, 5.0, 3.0, 0.01, seed=6)
    c2 = BKChain(64, 5.0, 3.0, 0.01, U=c1.U + 7.0)
    c2.state[0] = c1.state[0] + 7.0
    r1, r2 = c1.run(20), c2.run(20)
    assert np.allclose(r1["moment"], r2["moment"], rtol=1e-9, atol=1e-12)


def test_free_boundary_end_block_feels_one_spring():
    U = np.zeros(5)
    U[0] = -1.0
    l = 2.0
    f = [_elastic(U, j, 5, l * l) for j in range(5)]
    assert f[0] == pytest.approx(l * l * (0 - (-1.0)) + 1.0)     # one coupling spring + pulling spring
    assert f[1] == pytest.approx(l * l * (-1.0))


# ------------------------------------------------------------------ 1989 friction law

def test_single_block_1989_law_matches_direct_integration():
    """One block, symmetric friction, moving plate: compare the slip of each cycle."""
    alpha, nu = 2.5, 0.05
    c = BKChain(1, 1.0, alpha, 0.0, nu=nu, law=LAW_1989, U=np.zeros(1), h=1e-4)
    c.state[0] = 0.0
    r = c.run(3)
    # reference: slip starts at spring force 1 and stops when the velocity returns to 0
    def rhs(t, y):          # y = [U - U_start, V, P - P_start]
        force = 1.0 + y[2] - y[0]
        fr = np.sign(y[1]) / (1 + abs(2 * alpha * y[1])) if y[1] != 0 else 1.0
        return [y[1], force - fr, nu]
    ev = lambda t, y: y[1] if t > 1e-6 else 1.0
    ev.terminal, ev.direction = True, -1
    sol = solve_ivp(rhs, (0, 100), [0, 0, 0], events=ev, rtol=1e-11, atol=1e-13)
    slip = sol.y_events[0][0][0]
    assert r["moment"][1] == pytest.approx(slip, rel=1e-3)
    assert r["moment"][2] == pytest.approx(slip, rel=1e-3)


# ------------------------------------------------------------------ 1989 law: multi-block events

def reference_event_1989(U, P, l, alpha, nu, tol=1e-11):
    """One event with the symmetric 1989 friction law and a moving plate, exact event location.

    Stuck blocks start slipping forward at f = 1 or backward at f = -1. Each slipping block has a
    direction of motion; when its velocity crosses zero against that direction it sticks if |f| <= 1,
    and otherwise reverses direction and keeps slipping.
    """
    U = U.astype(float).copy()
    N, l2 = U.size, l * l
    V = np.zeros(N)
    j0 = int(np.argmax(forces(U, P, l2)))
    S = {j0: 1.0}                                   # slipping block -> direction of motion
    t = 0.0
    while S:
        idx, stuck = sorted(S), [j for j in range(N) if j not in S]
        dirs = dict(S)
        t0 = t

        def rhs(tt, y):
            u, v = y[:N], y[N:]
            ff = forces(u, P + nu * (tt - t0), l2)
            du, dv = np.zeros(N), np.zeros(N)
            for j in idx:
                du[j] = v[j]
                vv = v[j] if v[j] * dirs[j] > 0 else 0.0      # friction acts against the motion
                fr = dirs[j] / (1 + abs(2 * alpha * vv))
                dv[j] = ff[j] - fr
            return np.concatenate([du, dv])

        evs = []
        for j in idx:
            ev = (lambda tt, y, j=j: y[N + j])
            ev.terminal, ev.direction = True, -dirs[j]       # velocity crosses zero against the motion
            evs.append(ev)
        for j in stuck:
            for thr, dirn in ((1.0, 1), (-1.0, -1)):
                ev = (lambda tt, y, j=j, thr=thr: forces(y[:N], P + nu * (tt - t0), l2)[j] - thr)
                ev.terminal, ev.direction = True, dirn
                evs.append(ev)
        sol = solve_ivp(rhs, (t0, t0 + 1e4), np.concatenate([U, V]), events=evs, rtol=tol, atol=tol,
                        method="DOP853")
        U, V = sol.y[:N, -1].copy(), sol.y[N:, -1].copy()
        t = sol.t[-1]
        P_now = P + nu * (t - t0)
        k = [i for i, te in enumerate(sol.t_events) if len(te)][0]
        if k < len(idx):
            j = idx[k]
            V[j] = 0.0
            fj = forces(U, P_now, l2)[j]
            if abs(fj) <= 1.0:
                del S[j]
            else:
                S[j] = float(np.sign(fj))
        else:
            q = k - len(idx)
            S[stuck[q // 2]] = 1.0 if q % 2 == 0 else -1.0
        P = P_now
    return U, P


@pytest.mark.parametrize("seed,spread", [(0, 0.003), (1, 0.003), (2, 0.01)])
def test_1989_event_matches_reference_solver(seed, spread):
    rng = np.random.default_rng(seed)
    l, alpha, nu = 3.0, 2.5, 0.05
    U, P = chain_at_threshold(8, l, rng, spread)
    U_ref, P_ref = reference_event_1989(U, P, l, alpha, nu)
    c = BKChain(8, l, alpha, 0.0, nu=nu, law=LAW_1989, U=U.copy(), h=5e-5)
    c.state[0] = P - 1e-13
    r = c.run(1)
    while c.n_slipping > 0:              # the burst may consist of several consecutive events
        c.run(1)
    assert r["blocks"][0] > 1
    assert np.max(np.abs(c.U - U_ref)) < 1e-4 * max(1.0, np.max(np.abs(U_ref - U)))
    # the end of the event is set by the last block creeping to a stop at |f| ~ 1, a slow process whose
    # timing (not position) converges only linearly in h
    assert c.P == pytest.approx(P_ref, abs=5e-3)


# ------------------------------------------------------------------ events as space-time clusters

def _open_slip(c):
    """Slip already accumulated by events that are still in progress."""
    total = 0.0
    for slot in range(c.N):
        if c.ev["serial"][slot] > 0:
            total += c.ev["M"][slot]
    total += np.sum((c.U - c.U0)[c.slipping])
    return total


@pytest.mark.parametrize("law,sigma", [(LAW_1989, 0.0), (LAW_1991, 0.01)])
def test_every_slip_belongs_to_an_event_with_moving_plate(law, sigma):
    c = BKChain(100, 10.0, 2.5, sigma, nu=0.01, law=law, seed=2)
    c.run(2000)
    Ubar0, open0 = c.U.mean(), _open_slip(c)
    r = c.run(5000)
    assert r["moment"].sum() + _open_slip(c) - open0 == pytest.approx(c.N * (c.U.mean() - Ubar0), rel=1e-9)


def test_separate_regions_give_separate_events():
    """Two blocks far apart reach the threshold at the same time: two events, not one."""
    N, l = 60, 3.0
    U = np.zeros(N)
    U[10] = U[50] = -0.5                      # two equally loaded, well separated blocks
    c = BKChain(N, l, 2.0, 0.05, nu=0.0, U=U, h=1e-3)
    r = c.run(2)
    assert len(r["moment"]) == 2
    assert sorted(r["epicenter"].tolist()) == [10, 50]
    assert np.allclose(r["moment"], r["moment"][0])


def test_block_counts_never_exceed_chain_length_with_moving_plate():
    c = BKChain(100, 10.0, 4.0, 0.0, nu=0.01, law=LAW_1989, seed=1)
    c.run(1000)
    r = c.run(5000)
    assert r["blocks"].max() <= c.N
    assert np.all(r["blocks"] >= 1)


def test_1989_creeping_block_stops_after_first_oscillation():
    """A block reaching threshold under slow loading (1989 law): compare with the exact single-block ODE.

    The block creeps with an oscillating velocity; its first slip episode ends when the velocity first
    returns to zero, after a slip of order 2 pi nu / omega_1^3 (Carlson & Langer 1989, Eq. 4.8).
    """
    l, alpha, nu = 10.0, 2.5, 0.01
    w2 = 2 * l * l + 1
    U = np.zeros(21)
    U[10] = -0.002
    c = BKChain(21, l, alpha, 0.0, nu=nu, law=LAW_1989, U=U, h=1e-4)
    r = c.run(1)

    def rhs(t, y):
        v = y[1]
        fr = 1 / (1 + 2 * alpha * v) if v > 0 else 1.0
        return [v, 1 + nu * t - w2 * y[0] - fr]
    ev = lambda t, y: y[1] if t > 1e-3 else 1.0
    ev.terminal, ev.direction = True, -1
    s = solve_ivp(rhs, (0, 5), [0, 0], events=ev, rtol=1e-11, atol=1e-16, method="LSODA", max_step=1e-3)
    assert r["blocks"][0] == 1
    assert r["moment"][0] == pytest.approx(s.y_events[0][0][0], rel=2e-3)
    assert r["duration"][0] == pytest.approx(s.t_events[0][0], abs=2e-3)


def test_no_event_is_left_open_when_nothing_slips():
    """Every event whose blocks have all stopped must have been recorded (1989 law, moving plate)."""
    c = BKChain(100, 10.0, 2.5, 0.0, nu=0.01, law=LAW_1989, seed=1)
    for _ in range(40):
        c.run(5000)
        open_slots = [s for s in range(c.N) if c.ev["serial"][s] > 0]
        slipping_slots = {int(c.evslot[j]) for j in np.nonzero(c.slipping)[0]}
        assert set(open_slots) <= slipping_slots


def test_sum_rule_with_moving_plate():
    c = BKChain(100, 10.0, 2.5, 0.0, nu=0.01, law=LAW_1989, seed=3)
    c.run(50_000)
    P0 = c.P
    r = c.run(300_000)
    assert r["moment"].sum() / c.N == pytest.approx(c.P - P0, rel=2e-3)
