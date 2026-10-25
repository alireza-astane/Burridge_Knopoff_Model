"""Tests of the approximate-theory helpers."""

import numpy as np
import pytest

from bksoc import bktheory as T
from bksoc.bk import BKChain


def test_group_of_one_is_the_exact_one_block_event():
    l, alpha, sigma = 6.0, 3.0, 0.01
    U = np.zeros(21)
    U[10] = -0.5
    bk = BKChain(21, l, alpha, sigma, U=U, h=1e-4).run(1)["moment"][0]
    assert T.group_slip(1, l, alpha, sigma) == pytest.approx(bk, rel=1e-4)


def test_group_slip_grows_with_size_and_saturates():
    s = [T.group_slip(n, 6.0, 3.0, 0.01) for n in (1, 4, 16, 64, 10**6)]
    assert np.all(np.diff(s) > 0)
    assert s[-1] == pytest.approx(T.group_slip(10**9, 6.0, 3.0, 0.01), rel=1e-2)


def test_runaway_size():
    assert T.runaway_size(6.0, 3.0) == pytest.approx(9.0)
    assert T.runaway_size(6.0, 1.0) == np.inf


def test_binary_branching_is_normalised_and_critical_tail():
    s = np.arange(1, 400001, 2)
    for p in (0.3, 0.45, 0.5):
        total = T.binary_branching_pmf(s, p).sum()
        assert total == pytest.approx(min(1.0, (1 - p) / p), rel=2e-3)
    # at criticality P(s) ~ sqrt(2/pi) s^(-3/2) for odd s (density 1/2 of integers)
    big = np.array([10001, 100001])
    ratio = T.binary_branching_pmf(big, 0.5) * big**1.5
    assert np.allclose(ratio, np.sqrt(2 / np.pi), rtol=1e-3)
