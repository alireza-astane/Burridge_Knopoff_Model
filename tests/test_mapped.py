"""Tests of the sandpile limit of the BK chain (bksoc.mapped)."""

import numpy as np
import pytest

from bksoc.bk import BKChain
from bksoc.mapped import MappedChain, RULE_FIXED, RULE_RESET, bk_one_block_drop


def omega2(j, N, l2):
    return l2 + 1 if j in (0, N - 1) else 2 * l2 + 1


def reference_relax_fixed(f, l, d, rng):
    """Topple unstable sites one at a time, in random order (fixed-drop rule)."""
    f = f.copy()
    N, l2 = f.size, l * l
    n = np.zeros(N, dtype=int)
    M = 0.0
    while True:
        unstable = np.flatnonzero(f >= 1 - 1e-12)
        if unstable.size == 0:
            return f, n, M
        j = rng.choice(unstable)
        delta = d / omega2(j, N, l2)
        f[j] -= omega2(j, N, l2) * delta
        for m in (j - 1, j + 1):
            if 0 <= m < N:
                f[m] += l2 * delta
        n[j] += 1
        M += delta


@pytest.mark.parametrize("seed", range(5))
def test_fixed_rule_is_abelian(seed):
    rng = np.random.default_rng(seed)
    N, l, d = 40, 3.0, 0.3
    f0 = 1 - 0.05 * rng.random(N)          # dense configuration -> large avalanches
    c = MappedChain(N, l, RULE_FIXED, d=d, f=f0)
    r = c.run(1)
    j0 = int(np.argmax(f0))
    f_start = f0 + (1 - f0[j0])
    f_ref, n_ref, M_ref = reference_relax_fixed(f_start, l, d, rng)
    assert r["size"][0] == n_ref.sum() > 5
    assert r["area"][0] == np.count_nonzero(n_ref)
    assert r["moment"][0] == pytest.approx(M_ref, rel=1e-12)
    assert np.allclose(c.forces(), f_ref, atol=1e-12)


def test_force_balance_identity():
    """Total force drops by exactly the total slip; the plate raises it by N per unit displacement."""
    c = MappedChain(200, 5.0, RULE_FIXED, d=0.05, seed=1)
    F0, P0 = c.forces().sum(), c.P
    r = c.run(20_000)
    assert c.forces().sum() - F0 == pytest.approx(c.N * (c.P - P0) - r["moment"].sum(), abs=1e-7)


@pytest.mark.parametrize("rule", [RULE_FIXED, RULE_RESET])
def test_all_sites_stable_after_avalanches(rule):
    c = MappedChain(300, 4.0, rule, d=0.05, f_res=0.95, seed=2)
    for _ in range(50):
        c.run(100)
        assert c.forces().max() < 1 + 1e-12


def test_isolated_toppling_equals_bk_one_block_event():
    """With d from the BK one-block event, an isolated toppling has the BK slip."""
    l, alpha, sigma = 6.0, 3.0, 0.01
    U = np.zeros(21)
    U[10] = -0.5
    bk = BKChain(21, l, alpha, sigma, U=U, h=1e-4).run(1)["moment"][0]
    f = np.zeros(21)
    f[10] = 0.9
    mp = MappedChain(21, l, RULE_FIXED, d=bk_one_block_drop(l, alpha, sigma), f=f).run(1)
    assert mp["size"][0] == 1
    assert mp["moment"][0] == pytest.approx(bk, rel=0.015)          # linear-order drop vs exact BK


def test_drop_only_sets_the_units():
    """Configurations related by f' = 1 + (f - 1) d'/d give identical avalanches up to the slip unit."""
    rng = np.random.default_rng(3)
    f = 1 - 0.3 * rng.random(100)
    r1 = MappedChain(100, 4.0, RULE_FIXED, d=0.05, f=f).run(2000)
    r2 = MappedChain(100, 4.0, RULE_FIXED, d=0.10, f=1 + (f - 1) * 2).run(2000)
    assert np.array_equal(r1["size"][:200], r2["size"][:200])
    assert np.allclose(r2["moment"][:200], 2 * r1["moment"][:200], rtol=1e-9)


def test_reset_rule_one_site():
    """A single unstable site drops exactly to the residual force."""
    f = np.zeros(11)
    f[5] = 0.5
    c = MappedChain(11, 3.0, RULE_RESET, f_res=0.9, f=f)
    r = c.run(1)
    assert r["size"][0] == 1
    assert c.forces()[5] == pytest.approx(0.9)
    assert r["moment"][0] == pytest.approx(0.1 / 19)


def test_threshold_noise_breaks_periodicity_and_keeps_sum_rule():
    d = bk_one_block_drop(6.0, 3.0, 0.01)
    det = MappedChain(64, 6.0, RULE_FIXED, d=d, seed=3)
    noisy = MappedChain(64, 6.0, RULE_FIXED, d=d, seed=3, noise=0.05)
    for c in (det, noisy):
        while c.P < 100:
            c.run(50_000)
    rd, rn = det.run(200_000), noisy.run(200_000)

    def has_period(r, pmax=20_000):
        ep = r["epicenter"]
        return any(np.array_equal(ep[-20_000:], ep[-20_000 - p:-p]) for p in range(1, pmax))
    assert has_period(rd)
    assert not has_period(rn)
    F0 = noisy.forces().sum()
    P0 = noisy.P
    r = noisy.run(50_000)
    assert noisy.forces().sum() - F0 == pytest.approx(noisy.N * (noisy.P - P0) - r["moment"].sum(), abs=1e-7)



def test_reported_plate_positions_are_absolute_and_continuous():
    c = MappedChain(200, 6.0, RULE_FIXED, d=0.04, seed=8)
    c.run(2000)
    r1 = c.run(3000)
    r2 = c.run(3000)
    plate = np.concatenate([r1["plate"], r2["plate"]])
    assert np.all(np.diff(plate) >= -1e-9)
    assert plate[-1] == pytest.approx(c.P, rel=1e-12)
