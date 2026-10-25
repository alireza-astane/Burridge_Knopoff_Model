"""Approximate theory of the Burridge-Knopoff chain and of avalanche statistics (theory chapter)."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gammaln


def phi1991(y, sigma):
    return (1 - sigma) / (1 + max(y, 0.0) / (1 - sigma))


def group_slip(n, l, alpha, sigma, ends=2):
    """Slip per block of n neighbouring blocks that slip together as a rigid group, from threshold.

    The group feels its n pulling springs and the coupling springs at its `ends` attached ends:
    restoring constant per block Omega_n^2 = 1 + ends * l^2 / n. Solves
    delta'' = 1 - Omega_n^2 delta - phi(2 alpha delta') from rest until the velocity returns to zero.
    """
    W2 = 1.0 + ends * l * l / n
    def stop(t, y):
        return y[1] if t > 1e-9 else 1.0
    stop.terminal, stop.direction = True, -1
    T = 60 * np.pi / np.sqrt(W2) + 60
    sol = solve_ivp(lambda t, y: [y[1], 1 - W2 * y[0] - phi1991(2 * alpha * y[1], sigma)], (0, T), [0.0, 0.0],
                    events=stop, rtol=1e-10, atol=1e-15, max_step=0.02 / np.sqrt(W2))
    return sol.y_events[0][0][0]


def runaway_size(l, alpha):
    """Smallest rigid group whose linearised motion is not oscillatory: Omega_n <= alpha.

    n* = 2 l^2 / (alpha^2 - 1); infinite for alpha <= 1.
    """
    return np.inf if alpha <= 1 else 2 * l * l / (alpha**2 - 1)


def carlson_localization_length(l, alpha, sigma):
    """xi_tilde / a = (2 l / alpha) ln(4 l^2 / sigma) (Carlson et al. 1991, Eq. 3.6)."""
    return 2 * l / alpha * np.log(4 * l * l / sigma)


def binary_branching_pmf(s, p):
    """Probability that a binary branching process (0 or 2 offspring, probabilities 1-p and p) has
    total progeny s (odd): P(s = 2k+1) = C_k p^k (1-p)^(k+1), with C_k the Catalan numbers."""
    s = np.asarray(s)
    k = (s - 1) // 2
    logC = gammaln(2 * k + 1) - gammaln(k + 2) - gammaln(k + 1)
    out = np.exp(logC + k * np.log(p) + (k + 1) * np.log(1 - p))
    return np.where((s % 2 == 1) & (s >= 1), out, 0.0)
