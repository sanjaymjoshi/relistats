from math import ceil
from typing import Any, Optional

import scipy.optimize as opt
import scipy.stats as stats

from relistats import logger


def confidence_in_quantile_at_index(j: int, n: int, p: float) -> float:
    """Confidence (probability) that in sorted samples of size n, 
       p^th quantile (0 < p < 1) is greater than the sample
       at j'th place.
       From https://online.stat.psu.edu/stat415/lesson/19/19.2
       c = sum_{k=0}^{j-1} nCk * p^k * (1-p)^(n-k)
       This is same as cumulative density function for a binomial
       distribution, evaluated at j-1 out of n samples. 
    """
    return stats.binom.cdf(j-1, n, p)


def quantile_interval_indices(n: int, pp: float, c: float) -> Optional[tuple[int, int]]:
    """Returns tuple of two indices (1 to n-2) such that quantile q (0<q<1) lies within
    these two indices of a sorted list with confidence c (0<c<1)
    Return None if such a tuple cannot be computed.
    """
    if pp <= 0 or pp >= 1:
        logger.error("Quantile has to be > 0 and < 1, found: %f", pp)
        return None

    if c <= 0 or c >= 1:
        logger.error("Confidence c has to be > 0 and < 1, found: %f", c)
        return None

    n_min = 4
    if n <= n_min:
        logger.error("Need at least %d samples, found: %d", n_min, n)
        return None

    k_hi = ceil(pp * n)
    if k_hi >= n - 1:
        logger.error("Not enough samples, %d, for quantile %f. Need more.", n, pp)
        return None

    c_hi = confidence_in_quantile_at_index(k_hi, n, pp)
    while c_hi < c:
        k_hi += 1
        if k_hi == n - 1:
            logger.error("Highest confidence in %d samples %f < %f", n-2, c_hi, c)
            return None

        c_hi = confidence_in_quantile_at_index(k_hi, n, pp)

    logger.debug(
        "Found higher index %d at confidence %f for n=%d, q=%f, c=%f",
        k_hi, c_hi, n, pp
    )

    k_lo = k_hi - 1
    c_lo = confidence_in_quantile_at_index(k_lo, n, pp)
    while c_hi - c_lo < c:
        k_lo = k_lo - 1
        if k_lo == 0:
            logger.error(
                "Lower bound of interval needs to be > 0. Upper bound %d at confidence %f:.2", k_hi, c_hi
            )
            return None
        c_lo = confidence_in_quantile_at_index(k_lo, n, pp)

    return (k_lo, k_hi)


def tolerance_interval_indices(n: int, t: float, c: float) -> Optional[tuple[int, int]]:
    q = (1-t)/2 # Half budget
    ii1 = quantile_interval_indices(n, q, c)
    ii2 = quantile_interval_indices(n, 1-q, c)

    if ii1 is None or ii2 is None:
        return None
    return (min(ii1[0], ii2[0]), max(ii1[1], ii2[1]))


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


def confidence_interval_of_median(c: float, *args) -> Optional[tuple[Any, Any]]:
    """Returns median interval from args at confidence of at least c, if possible.
    Returns None if not possible.
    args is any iterable (list, tuple, set)
    """
    return confidence_interval_of_quantile(0.5, c, *args)


def confidence_interval_of_quantile(q: float, c: float, *args) -> Optional[tuple[Any, Any]]:
    """Returns q'th quantile interval from args at confidence of at least c, if possible.
    Returns None if not possible.
    args is any iterable (list, tuple, set)
    """
    n = len(*args)
    ii = quantile_interval_indices(n, q, c)
    return tuple(sorted(*args)[slice(ii[0], ii[1] + 1, ii[1] - ii[0])]) if ii else None
