"""Deterministic finite-size LCD--PORT profile comparisons.

All transport distances have order one; p is the ground-norm exponent.
No random sampling is used in this module.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import betaln, digamma, zeta


def root_mean_scaled(n: int) -> float:
    """E[S_n(0)] / sqrt(n), evaluated without subtracting large log-Gammas."""
    if n < 1:
        raise ValueError("n must be positive")
    return math.exp(math.log(math.pi) - betaln(n + 0.5, 0.5) - 0.5 * math.log(n))


def elementary_coefficients(n: int, degree: int) -> np.ndarray:
    """Coefficients through degree of product_{j=2}^n (1+u/j).

    Newton's identities use O(degree^2) arithmetic. The special-function
    evaluations avoid a loop over n. Intended for the modest degrees in
    these experiments; this is not an interval-arithmetic certificate.
    """
    if n < 1 or degree < 0:
        raise ValueError("require n >= 1 and degree >= 0")
    e = np.zeros(degree + 1)
    e[0] = 1.0
    degree = min(degree, n - 1)
    powers = np.zeros(degree + 1)
    if degree:
        powers[1] = digamma(n + 1) - digamma(2)
    for j in range(2, degree + 1):
        powers[j] = zeta(j, 2) - zeta(j, n + 1)
    for k in range(1, degree + 1):
        e[k] = math.fsum((-1) ** (j - 1) * powers[j] * e[k - j]
                         for j in range(1, k + 1)) / k
    return e


def normalized_means(n: int, H: int) -> np.ndarray:
    """E[X_{n,h}], h=0,...,H, X=2^h h! S_h/(sqrt(n) log(n)^h)."""
    if n < 2:
        raise ValueError("normalization requires n >= 2")
    e = elementary_coefficients(n, H)
    L = math.log(n)
    return root_mean_scaled(n) * np.array([
        math.factorial(h) * e[h] / L ** h for h in range(H + 1)])


def mean_gaps(n: int, H: int) -> np.ndarray:
    """Exact finite-n mean differences in normalized coordinates 0,...,H."""
    if H < 0:
        raise ValueError("H must be nonnegative")
    gaps = np.zeros(H + 1)
    if H:
        gaps[1:] = np.arange(1, H + 1) * normalized_means(n, H - 1) / math.log(n)
    return gaps


def mean_lower_bound(n: int, H: int, p: float) -> float:
    """Norm of mean difference; equals transport distance when p=1."""
    return float(np.linalg.norm(mean_gaps(n, H), ord=p))


def leading(n: int, H: int, p: float = 1) -> float:
    return math.sqrt(math.pi) * float(np.linalg.norm(np.arange(1, H + 1), ord=p)) / math.log(n)


def corrected_l1(n: int, H: int) -> float:
    """Leading term plus first logarithmic correction."""
    return leading(n, H) * (1 + 2 * (np.euler_gamma - 1) * (H - 1) / (3 * math.log(n)))


def direct_mean_recurrence(n: int, H: int) -> np.ndarray:
    """Independent O(nH) check using the one-step mean recurrence."""
    m = np.zeros(H + 1)
    m[0] = 2.0
    for k in range(1, n):
        m += (m + np.r_[0.0, m[:-1]]) / (2 * k + 1)
    L = math.log(n)
    return m * np.array([2 ** h * math.factorial(h) / (math.sqrt(n) * L ** h)
                         for h in range(H + 1)])
