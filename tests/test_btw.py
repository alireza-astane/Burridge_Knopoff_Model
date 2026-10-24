"""Tests of the BTW sandpile engine against exact results.

The checks go from purely mechanical (conservation, Abelian property versus an
independent slow implementation) to statistical (stationary-state averages against
exact theory).
"""

import itertools

import numpy as np
import pytest

from bksoc import BTW, open_lattice
from bksoc import theory


def reference_relax(z, nbr, rng):
    """Slow, obviously-correct relaxation: topple unstable sites one at a time in random order."""
    z = z.copy()
    k = nbr.shape[1]
    topplings = np.zeros(len(z), dtype=np.int64)
    lost = 0
    while True:
        unstable = np.flatnonzero(z >= k)
        if unstable.size == 0:
            return z, topplings, lost
        i = rng.choice(unstable)
        z[i] -= k
        topplings[i] += 1
        for m in nbr[i]:
            if m < 0:
                lost += 1
            else:
                z[m] += 1


# ----------------------------------------------------------------------------- lattice

def test_open_lattice_1d():
    nbr = open_lattice((4,))
    assert nbr.tolist() == [[-1, 1], [0, 2], [1, 3], [2, -1]]


def test_open_lattice_2d_symmetric_and_degree():
    nbr = open_lattice((5, 7))
    N = 35
    for i in range(N):
        for m in nbr[i]:
            if m >= 0:
                assert i in nbr[m]          # bonds are symmetric
    # corners have 2 sink slots, edges 1, bulk 0
    n_sink = (nbr < 0).sum(axis=1).reshape(5, 7)
    assert n_sink[0, 0] == 2 and n_sink[0, 3] == 1 and n_sink[2, 3] == 0
    assert n_sink.sum() == 2 * (5 + 7)


# ------------------------------------------------------------------- mechanical checks

@pytest.mark.parametrize("shape", [(30,), (12, 12), (5, 4, 6)])
def test_grain_conservation(shape):
    pile = BTW(open_lattice(shape), seed=0)
    av = pile.drive(20_000)
    assert pile.z.sum() == pile.grains_added - pile.grains_lost
    assert pile.grains_lost == av["lost"].sum()
    assert pile.z.min() >= 0 and pile.z.max() < pile.k


@pytest.mark.parametrize("shape", [(25,), (9, 9), (4, 4, 4)])
def test_abelian_matches_reference(shape):
    """Final state, size, area and loss must not depend on the toppling order."""
    nbr = open_lattice(shape)
    rng = np.random.default_rng(1)
    k = nbr.shape[1]
    for _ in range(40):
        z0 = rng.integers(0, k, size=nbr.shape[0])
        site = rng.integers(nbr.shape[0])
        pile = BTW(nbr, z=z0)
        out = pile.add([site])
        z_ref = z0.copy()
        z_ref[site] += 1
        z_ref, topplings, lost = reference_relax(z_ref, nbr, rng)
        assert np.array_equal(pile.z, z_ref)
        assert out["size"][0] == topplings.sum()
        assert out["area"][0] == np.count_nonzero(topplings)
        assert out["lost"][0] == lost


def test_recorded_topplings_match_reference():
    nbr = open_lattice((11, 11))
    rng = np.random.default_rng(2)
    z0 = rng.integers(0, 4, size=121)
    pile = BTW(nbr, z=z0)
    out = pile.add([60], record=True)
    z_ref = z0.copy()
    z_ref[60] += 1
    _, topplings, _ = reference_relax(z_ref, nbr, rng)
    assert np.array_equal(out["topplings"], topplings)


def test_strongly_unstable_start_matches_reference():
    """A site holding many grains (multiple topplings per generation) is still handled exactly."""
    nbr = open_lattice((7, 7))
    z0 = np.full(49, 3)
    z0[24] = 3 + 40              # dump 40 extra grains at the centre, then add one more
    pile = BTW(nbr, z=z0)
    out = pile.add([24])
    z_ref = z0.copy()
    z_ref[24] += 1
    z_ref, topplings, lost = reference_relax(z_ref, nbr, np.random.default_rng(0))
    assert np.array_equal(pile.z, z_ref)
    assert out["size"][0] == topplings.sum()


def test_duration_of_single_line_avalanche():
    """1D chain with all heights 1: a grain at the left end sweeps right in L generations."""
    L = 10
    pile = BTW(open_lattice((L,)), z=np.ones(L))
    out = pile.add([0])
    # zero moves to (0 - 1) mod (L+1) = L, i.e. the last site, after L topplings
    assert out["size"][0] == L and out["duration"][0] == L
    assert pile.z.tolist() == [1] * (L - 1) + [0]


# ----------------------------------------------------------------- exact 1D BTW results

@pytest.mark.parametrize("L", [1, 2, 7, 20])
def test_1d_closed_form_every_configuration(L):
    """s(l, j) = g(j) + g(l') - g(l) and l' = (l - j) mod (L+1), for every recurrent state and seed."""
    nbr = open_lattice((L,))
    for l in range(L + 1):
        for j in range(1, L + 1):
            z0 = np.ones(L, dtype=np.int64)
            if l > 0:
                z0[l - 1] = 0
            pile = BTW(nbr, z=z0)
            s = pile.add([j - 1])["size"][0]
            s_exact, lp = theory.btw1d_size(L, l, j)
            assert s == s_exact
            zeros = np.flatnonzero(pile.z == 0)
            assert zeros.tolist() == ([] if lp == 0 else [lp - 1])


def test_1d_toppling_profile_is_green_function():
    """n = G (delta_j + delta_l' - delta_l) with G_ik = min(i,k) - i k / (L+1)."""
    L, l, j = 40, 30, 12
    z0 = np.ones(L, dtype=np.int64)
    z0[l - 1] = 0
    out = BTW(open_lattice((L,)), z=z0).add([j - 1], record=True)
    _, lp = theory.btw1d_size(L, l, j)
    i = np.arange(1, L + 1)
    G = lambda k: np.minimum(i, k) - i * k / (L + 1) if k > 0 else 0 * i
    assert np.allclose(out["topplings"], G(j) + G(lp) - G(l))


@pytest.mark.parametrize("L", [5, 16, 40])
def test_1d_exact_moments(L):
    s = theory.btw1d_all_sizes(L)
    m1, m2 = theory.btw1d_moments(L)
    assert s.mean() == pytest.approx(m1, rel=1e-12)
    assert (s.astype(float) ** 2).mean() == pytest.approx(m2, rel=1e-12)


def test_1d_dynamics_reaches_recurrent_set():
    """From a random start the chain ends up with at most one empty site (Dhar's burning test)."""
    L = 50
    pile = BTW(open_lattice((L,)), seed=3, z=np.random.default_rng(3).integers(0, 2, L))
    pile.drive(5_000)
    for _ in range(200):
        pile.drive(1)
        assert np.count_nonzero(pile.z == 0) <= 1


def test_1d_stationary_size_distribution():
    """Long-run histogram of sizes equals the exact uniform-over-recurrent-states prediction."""
    L = 12
    pile = BTW(open_lattice((L,)), seed=4)
    pile.fill_max()
    pile.drive(1_000)
    n = 400_000
    s = pile.drive(n)["size"]
    exact = theory.btw1d_all_sizes(L)
    smax = exact.max()
    p_sim = np.bincount(s, minlength=smax + 1) / n
    p_exact = np.bincount(exact, minlength=smax + 1) / exact.size
    # chi-square test over all sizes with non-zero exact probability
    nz = p_exact > 0
    chi2 = n * ((p_sim[nz] - p_exact[nz]) ** 2 / p_exact[nz]).sum()
    dof = nz.sum() - 1
    assert p_sim[~nz].sum() == 0
    assert chi2 < dof + 5 * np.sqrt(2 * dof)


# ----------------------------------------------------------------- exact results in 2D

def test_number_of_recurrent_states_is_det_delta():
    """Dhar: |R| = det(Delta). On a 2x3 lattice, count the distinct states visited in a long run."""
    nbr = open_lattice((2, 3))
    D = theory.toppling_matrix(nbr).toarray()
    n_rec = round(np.linalg.det(D))
    pile = BTW(nbr, seed=5)
    pile.fill_max()
    seen = set()
    sites = pile.rng.integers(0, 6, 300_000)
    for start in range(0, sites.size, 1000):
        for i in sites[start:start + 1000]:
            pile.add([i])
            seen.add(pile.z.tobytes())
    assert len(seen) == n_rec


def test_2d_mean_size_equals_green_function():
    """Stationary <s> for uniform seeding equals (1/N) sum_ij G_ij, with G = Delta^{-1}."""
    L = 16
    nbr = open_lattice((L, L))
    exact = theory.mean_topplings_per_site(nbr).mean()
    pile = BTW(nbr, seed=6)
    pile.fill_max()
    pile.drive(20 * L * L)
    s = pile.drive(400_000)["size"].astype(float)
    # avalanches are correlated only weakly here; use batch means for the error bar
    batches = s.reshape(100, -1).mean(axis=1)
    err = batches.std(ddof=1) / np.sqrt(batches.size)
    assert abs(s.mean() - exact) < 4 * err


@pytest.mark.slow
def test_2d_height_probabilities():
    """Bulk height probabilities of the 2D BTW model (Majumdar-Dhar, Priezzhev)."""
    L = 96
    nbr = open_lattice((L, L))
    pile = BTW(nbr, seed=7)
    pile.fill_max()
    pile.drive(30 * L * L)
    centre = np.zeros((L, L), dtype=bool)
    centre[L // 4: 3 * L // 4, L // 4: 3 * L // 4] = True
    centre = centre.ravel()
    counts = np.zeros(4)
    for _ in range(300):
        pile.drive(2 * L * L // 10)
        counts += np.bincount(pile.z[centre], minlength=4)
    p = counts / counts.sum()
    assert np.allclose(p, theory.BTW2D_HEIGHT_PROBS, atol=0.004)
