"""Statistical methods for quantiles.
"""
from math import ceil
from typing import Any, Optional

import scipy.optimize as opt
import scipy.stats as stats

from relistats import logger


def confidence_in_quantile(j: int, n: int, p: float) -> float:
    """Confidence (probability) that in sorted samples of size n,
    p^th quantile (0 < p < 1) is greater than the sample
    at j'th place out of n, 1 <= j <= n
    From https://online.stat.psu.edu/stat415/lesson/19/19.2
    c = sum_{k=0}^{j-1} nCk * p^k * (1-p)^(n-k)
    This is same as cumulative density function for a binomial
    distribution, evaluated at j-1 out of n samples.
    Note that using j=n+1 will return 1.
    """
    return stats.binom.cdf(j - 1, n, p)


def _quantile_invalid(pp: float) -> bool:
    if pp <= 0 or pp >= 1:
        logger.error("Quantile has to be > 0 and < 1, found: %f", pp)
        return True
    return False


def _confidence_invalid(c: float) -> bool:
    if c <= 0 or c >= 1:
        logger.error("Confidence c has to be > 0 and < 1, found: %f", c)
        return True
    return False


def _num_samples_invalid(n: int) -> bool:
    n_min = 3
    if n < n_min:
        logger.error("Need at least %d samples, found: %d", n_min, n)
        return True
    return False


def quantile_interval_places(n: int, pp: float, c: float) -> Optional[tuple[int, int]]:
    """Returns tuple of two places (1..n) such that quantile pp (0<pp<1) lies within
    these two places of n sorted samples at confidence of at least c (0<c<1).

    Note that the places are not indexed at zero!

    Use this method if you plan to sort samples yourself, else you can use
    confidence_interval_of_quantile method.

    Return None if such a tuple cannot be computed. If that happens, try to increase n,
    reduce pp, or reduce c.
    """
    if _quantile_invalid(pp) or _confidence_invalid(c) or _num_samples_invalid(n):
        return None

    k_hi = ceil(pp * n)
    if k_hi > n - 1:
        logger.error("Not enough samples, %d, for quantile %f. Need more.", n, pp)
        return None

    lowest_conf = confidence_in_quantile(1, n, pp)
    c_hi = confidence_in_quantile(k_hi, n, pp)
    while c_hi < c + lowest_conf:
        k_hi += 1
        if k_hi == n + 1:
            logger.error("Highest confidence in %d samples %f < %f", k_hi - 1, c_hi, c)
            return None

        c_hi = confidence_in_quantile(k_hi, n, pp)

    logger.debug(
        "Found higher index %d at confidence %f for n=%d, q=%f, c=%f",
        k_hi,
        c_hi,
        n,
        pp,
        c,
    )

    k_lo = k_hi - 1
    c_lo = confidence_in_quantile(k_lo, n, pp)
    while c_hi - c_lo < c:
        k_lo = k_lo - 1
        if k_lo == 0:
            logger.error(
                "Lower bound of interval needs to be > 0. Upper bound %d at confidence %f:.2",
                k_hi,
                c_hi,
            )
            return None
        c_lo = confidence_in_quantile(k_lo, n, pp)

    return (k_lo, k_hi)


def tolerance_interval_places(n: int, t: float, c: float) -> Optional[tuple[int, int]]:
    """Returns tolerance interval places. Out of n sorted samples, a fraction of t samples
    are expected to be within these two places, with a probability of at least c.
    Returns None if such tuple cannot be calculated. If that happens, try to increase n,
    reduce t, or reduce c.
    """
    # Construct the interval in two halves, each with t/2 fraction.
    # At the high end, start at the median and expand until confidence exceeds c for
    # quantile of 0.5 + t/2.
    # At the low end, start at zero and keep incrementing the index until confidence
    # exceeds 1-c for quantile of 0.5 - t/2.
    if _quantile_invalid(t) or _confidence_invalid(c) or _num_samples_invalid(n):
        return None

    median_index = (n + 1) // 2
    j_hi = median_index
    p_hi = 0.5 + t / 2
    while confidence_in_quantile(j_hi, n, p_hi) < c:
        j_hi += 1
        if j_hi == n + 1:
            logger.error(
                "Not enough samples, %d, for tolerance %f at confidence %f. Need more.",
                n,
                t,
                c,
            )
            return None
    # Now start at zero index and quantile of 0.5 - t/2
    j_lo = 0
    p_lo = 0.5 - t / 2
    while confidence_in_quantile(j_lo, n, p_lo) < 1 - c:
        j_lo += 1
        if j_lo == median_index:
            logger.error(
                "Not enough samples, %d, for tolerance %f at confidence %f. Need more.",
                n,
                t,
                c,
            )
            return None

    return (j_lo, j_hi)


def _assurance_quantile_fn(x: float, j: int, n: int) -> float:
    """Function to find roots of x = confidence_in_quantile(n, f, x)"""
    x_hat = confidence_in_quantile(j, n, x) or 0
    return x_hat - x


def assurance_in_quantile(j: int, n: int, tol=0.001) -> Optional[float]:
    """Assurance level at j'th index out of n sorted samples. The confidence
       is at least the quantile level.

    :param j: sample index
    :type j: int, >0
    :param n: number of samples
    :type n: int, >=0
    :param tol: accuracy tolerance
    :type tol: float, optional

    :return: Assurance or None if it could not be computed
    :rtype: float, optional
    """
    if _num_samples_invalid(n):
        return None

    if j <= 0 or j > n - 1:
        logger.error(
            "Sample index %d out of range, need to be between 0 and %d", j, n - 1
        )
        return None

    # Use numerical optimization to find real root of the confidence equation
    # x - confidence_in_quantile(j, n, x)
    return opt.brentq(
        _assurance_quantile_fn,
        a=0,  # Lowest possible value
        b=1,  # Highest possible value
        args=(j, n),
        xtol=tol,
    )


def confidence_interval_of_median(c: float, *args) -> Optional[tuple[Any, Any]]:
    """Returns median interval from args at confidence of at least c, if possible.
    Returns None if not possible.
    args is any iterable (list, tuple, set)
    """
    return confidence_interval_of_quantile(0.5, c, *args)


def confidence_interval_of_quantile(
    q: float, c: float, *args
) -> Optional[tuple[Any, Any]]:
    """Returns q'th quantile interval from args at confidence of at least c, if possible.
    Use this method if you data is not sorted already, else you can use quantile_interval_places.
    Returns None if not possible.
    args is any iterable (list, tuple, set)
    """
    n = len(*args)
    ii = quantile_interval_places(n, q, c)
    # Need to subtract 1 from the places, to account for 0-based index
    return (
        tuple(sorted(*args)[slice(ii[0] - 1, ii[1], ii[1] - ii[0])]) if ii else None  # type: ignore
    )


def tolerance_interval(t: float, c: float, *args) -> Optional[tuple[Any, Any]]:
    """Returns tolerance interval for middle t (0<t<1) fraction of samples,
    with confidence c (0<c<1), if possible.
    Use this method if you data is not sorted already, else you can use tolerance_interval_places.
    Returns None if not possible.
    args is any iterable (list, tuple, set)
    """
    n = len(*args)
    ii = tolerance_interval_places(n, t, c)
    # Need to subtract 1 from the places, to account for 0-based index
    return (
        tuple(sorted(*args)[slice(ii[0] - 1, ii[1], ii[1] - ii[0])]) if ii else None  # type: ignore
    )


def _assurance_interval_fn(x: float, j_lo: int, j_hi: int, n: int) -> float:
    """Function to find roots of x = confidence_in_quantile(n, f, x)"""
    x_hi_hat = confidence_in_quantile(j_hi, n, x) or 0
    x_lo_hat = confidence_in_quantile(j_lo, n, 1 - x) or 0
    x_hat = min(x_hi_hat, 1 - x_lo_hat)
    return x_hat - x


def assurance_in_interval(j_lo: int, j_hi: int, n: int, tol=0.001) -> Optional[float]:
    """Assurance level for interval [j_lo, j_hi] out of n sorted samples. Assurance
    level of a means a% of samples will be within this interval with a% confidence.
    Example: Out of 16 ordered samples, we can be 80% confident that 80% samples will
    be between 1st and 15th place.

    :param j_lo: sample place at lower end
    :type j_lo: int, >0
    :param j_hi: sample place at upper end
    :type j_hi: int, n > j_hi > j_lo
    :param n: number of samples
    :type n: int, >=0
    :param tol: accuracy tolerance
    :type tol: float, optional

    :return: Assurance or None if it could not be computed
    :rtype: float, optional
    """
    if _num_samples_invalid(n):
        return None

    if j_lo <= 0 or j_lo > n - 1 or j_hi <= 0 or j_hi > n - 1:
        logger.error(
            "Sample places %d, %d out of range, need to be between 0 and %d",
            j_lo,
            j_hi,
            n - 1,
        )
        return None

    if j_lo >= j_hi:
        logger.error("Places %d >= %d not supported", j_lo, j_hi)
        return None

    # Use numerical optimization to find real root of the confidence equation
    # x - _assurance_interval_fn(x, j_lo, j_hi, n)
    return opt.brentq(
        _assurance_interval_fn,
        a=0,  # Lowest possible value
        b=1,  # Highest possible value
        args=(j_lo, j_hi, n),
        xtol=tol,
    )
