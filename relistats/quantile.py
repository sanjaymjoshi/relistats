from math import ceil
from typing import Any, Optional

import scipy.optimize as opt
import scipy.stats as stats

from relistats import logger


def confidence_in_quantile_at_index(k: int, n: int, q: float) -> float:
    """Confidence that quantile q (0 < q < 1) is at less than k'th index out of sorted n samples"""
    return stats.binom.cdf(k, n, q)


def quantile_interval_indices(n: int, q: float, c: float) -> Optional[tuple[int, int]]:
    """Returns tuple of two indices (1 to n-2) such that quantile q (0<q<1) lies within
    these two indices of a sorted list with confidence c (0<c<1)
    Return None if such a tuple cannot be computed.
    """
    if q <= 0 or q >= 1:
        logger.error(f"Quantile has to be > 0 and < 1, found: {q}")
        return None

    if c <= 0 or c >= 1:
        logger.error(f"Confindence c has to be > 0 and < 1, found: {c}")
        return None

    n_min = 4
    if n <= n_min:
        logger.error(f"Need at least {n_min} samples, found: {n}")
        return None

    k_hi = ceil(q * n)
    if k_hi >= n - 1:
        logger.error(f"Not enough samples, {n}, for quantile {q}. Need more.")
        return None

    c_hi = confidence_in_quantile_at_index(k_hi, n, q)
    while c_hi < c:
        k_hi += 1
        if k_hi == n - 1:
            logger.error(f"Highest confidence in {n-2} samples {c_hi} < {c}")
            return None

        c_hi = confidence_in_quantile_at_index(k_hi, n, q)

    logger.debug(
        f"Found higher index {k_hi} at confidence {c_hi} for n={n}, q={q}, c={c}"
    )

    k_lo = k_hi - 1
    c_lo = confidence_in_quantile_at_index(k_lo, n, q)
    while c_hi - c_lo < c:
        k_lo = k_lo - 1
        if k_lo == 0:
            logger.error(
                "Lower bound of interval needs to be > 0.",
                f" Upper bound {k_hi} at confidence {c_hi}",
            )
            return None
        c_lo = confidence_in_quantile_at_index(k_lo, n, q)

    return (k_lo, k_hi)


def _assurance_quantile_fn(x: float, k: int, n: int) -> float:
    """Function to find roots of x = confidence_in_quantile(n, f, x)"""
    x_hat = confidence_in_quantile_at_index(k, n, x) or 0
    return x_hat - x


def assurance_in_quantile(k: int, n: int, tol=0.001) -> Optional[float]:
    """Assurance level at k'th index out of n sorted samples. The confidence
       is at least the quantile level.

    :param k: sample index (0-based)
    :type k: int, >0
    :param n: number of samples
    :type n: int, >=0
    :param tol: accuracy tolerance
    :type tol: float, optional

    :return: Assurance or None if it could not be computed
    :rtype: float, optional
    """
    if n <= 1 or k <= 0 or k > n - 1:
        return None

    # Use numerical optimization to find real root of the confidence equation
    # c - confidence(n, f, r)
    return opt.brentq(
        _assurance_quantile_fn,
        a=0,  # Lowest possible value
        b=1,  # Highest possible value
        args=(k, n),
        xtol=tol,
    )


def median_interval(c: float, *args) -> Optional[tuple[Any, Any]]:
    """Returns median interval from args at confidence of at least c, if possible.
    Returns None if not possible.
    args is any iterable (list, tuple, set)
    """
    return quantile_interval(0.5, c, *args)


def quantile_interval(q: float, c: float, *args) -> Optional[tuple[Any, Any]]:
    """Returns q'th quantile interval from args at confidence of at least c, if possible.
    Returns None if not possible.
    args is any iterable (list, tuple, set)
    """
    n = len(*args)
    ii = quantile_interval_indices(n, q, c)
    return tuple(sorted(*args)[slice(ii[0], ii[1] + 1, ii[1] - ii[0])]) if ii else None
