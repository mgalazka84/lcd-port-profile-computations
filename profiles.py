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


def full_log_polynomial_l1(n: int, H: int) -> float:
    """Use the complete fixed-level logarithmic polynomial in every mean.

    This omits finite-n Gamma-ratio corrections; it is not an exact
    formula or a uniform guarantee for arbitrary growing H.
    """
    if n < 2 or H < 0:
        raise ValueError("require n >= 2 and H >= 0")
    if H == 0:
        return 0.0
    g = np.zeros(H)
    g[0] = 1.0
    # j times the coefficient of u^j in log(1/Gamma(2+u)).
    log_derivative = np.zeros(H)
    if H > 1:
        log_derivative[1] = np.euler_gamma - 1
    for j in range(2, H):
        log_derivative[j] = (-1)**(j+1) * zeta(j, 2)
    for k in range(1, H):
        g[k] = math.fsum(log_derivative[j]*g[k-j] for j in range(1, k+1))/k
    L = math.log(n)
    return math.sqrt(math.pi)*math.fsum(
        h/L * math.fsum(math.factorial(h-1)/math.factorial(h-1-j)*g[j]/L**j
                        for j in range(h)) for h in range(1, H+1))


def retained_level_cutoff(n: int, tolerance: float, search_limit: int = 12) -> int:
    """Largest H with exact-formula sum cost <= tolerance, H in 0,...,n.

    Search modest degrees only, raising if the crossing is not reached.
    The zero coordinate agrees exactly, so H counts non-root levels.
    """
    if tolerance < 0 or search_limit < 0:
        raise ValueError("tolerance and search_limit must be nonnegative")
    upper = min(n, search_limit+1)
    costs = np.cumsum(mean_gaps(n, upper))
    crossings = np.flatnonzero(costs > tolerance)
    if len(crossings):
        return int(crossings[0])-1
    if upper == n:
        return n
    raise ValueError("increase search_limit: no crossing found")


def direct_mean_recurrence(n: int, H: int) -> np.ndarray:
    """Independent O(nH) check using the one-step mean recurrence."""
    m = np.zeros(H + 1)
    m[0] = 2.0
    for k in range(1, n):
        m += (m + np.r_[0.0, m[:-1]]) / (2 * k + 1)
    L = math.log(n)
    return m * np.array([2 ** h * math.factorial(h) / (math.sqrt(n) * L ** h)
                         for h in range(H + 1)])
