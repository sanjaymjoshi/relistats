"""Statistical methods for confidence and tolerance intervals
"""
from typing import Any, Optional

import scipy.optimize as opt
import scipy.stats as stats

from relistats import logger
from relistats.percentile import _num_samples_invalid, confidence_in_percentile


def confidence_interval_of_mean(c: float, *args) -> tuple[Any, Any]:
    """Returns confidence interval of mean from args at confidence of c, 0 < c < 1"""
    mean, sem = stats.tmean(*args), stats.sem(*args)
    return stats.norm.interval(c, loc=mean, scale=sem)


def confidence_interval_of_median(c: float, *args) -> Optional[tuple[Any, Any]]:
    """Returns median interval from args at confidence of at least c, if possible.
    Returns None if not possible.
    args is any iterable (list, tuple, set)
    """
    return confidence_interval_of_percentile(0.5, c, *args)


def confidence_interval_of_percentile(
    p: float, c: float, *args
) -> Optional[tuple[Any, Any]]:
    """Returns p'th percentile/quantile interval from args at confidence of at least c, if possible.
    Use this method if you data is not sorted already, else you can use quantile_interval_places.
    Returns None if not possible.
    args is any iterable (list, tuple, set)
    """
    n = len(*args)
    ii = percentile_interval_locs(n, p, c)
    # Need to subtract 1 from the places, to account for 0-based index
    return (
        tuple(sorted(*args)[slice(ii[0] - 1, ii[1], ii[1] - ii[0])]) if ii else None  # type: ignore
    )


def percentile_interval_locs(n: int, p: float, c: float) -> Optional[tuple[int, int]]:
    """Returns tuple of two locations (1..n) such that percentile/quantile p
    (0 < p < 1) lies within these two locations of n sorted samples with confidence
    of at least c (0 < c < 1).

    Note that the locations are indexed at 1 and not zero!

    Use this method if you plan to sort samples yourself, else you can use
    confidence_interval_of_quantile method.

    Return None if such a tuple cannot be computed. If that happens, try to increase n,
    reduce p, or reduce c.
    """
    if _percentile_invalid(p) or _confidence_invalid(c) or _num_samples_invalid(n):
        return None

    candidates = _percentile_interval_locs_candidates(n, p, c)
    if len(candidates) == 0:
        return None
    interval_sizes = [x[1] - x[0] for x in candidates]
    min_id = interval_sizes.index(min(interval_sizes))
    return candidates[min_id]


def _percentile_invalid(p: float) -> bool:
    if p <= 0 or p >= 1:
        logger.error("Percentile/quantile has to be > 0 and < 1, found: %f", p)
        return True
    return False


def _confidence_invalid(c: float) -> bool:
    if c <= 0 or c >= 1:
        logger.error("Confidence has to be > 0 and < 1, found: %f", c)
        return True
    return False


def _percentile_interval_locs_candidates(
    n: int, p: float, c: float
) -> list[tuple[int, int]]:
    """Returns list of tuples of two locations, (1..n), that meet requirement that
       percentile/quantile p, 0 < p < 1 lies within these two locations with
       confidence >= c, 0 < c < 1.

    Returns empty list if none found.
    """
    c_max = confidence_in_percentile(n, n, p)
    c_min = confidence_in_percentile(1, n, p)
    if c_max - c_min < c:
        logger.info("Confidence %f < required %f, n=%d, p=%f", c_max - c_min, c, n, p)
        return []

    # Start from mid-point and expand until first candidate for higher bound
    j_hi = (n + 1) // 2
    j_lo = 1
    c_lo = c_min
    while j_hi <= n:
        c_hi = confidence_in_percentile(j_hi, n, p)
        if c_hi - c_lo > c:
            logger.debug(
                "Candidate: j_hi=%d, c_hi=%f - c_lo=%f > c=%f", j_hi, c_hi, c_lo, c
            )
            break
        j_hi += 1

    rc: list[tuple[int, int]] = []
    while j_hi <= n:
        c_hi = confidence_in_percentile(j_hi, n, p)
        c_lo_temp = confidence_in_percentile(j_lo, n, p)
        while c_hi - c_lo_temp > c:
            j_lo += 1
            c_lo_temp = confidence_in_percentile(j_lo, n, p)
        # Now step back j_lo
        j_lo -= 1
        logger.debug(
            "Candidate: j_lo=%d, c_hi=%f - c_lo=%f > c=%f", j_lo, c_hi, c_lo_temp, c
        )
        rc.append((j_lo, j_hi))
        j_hi += 1

    return rc


def tolerance_interval(t: float, c: float, *args) -> Optional[tuple[Any, Any]]:
    """Returns tolerance interval for middle t (0<t<1) fraction of samples,
    with confidence c (0<c<1), if possible.
    Use this method if you data is not sorted already, else you can use tolerance_interval_places.
    Returns None if not possible.
    args is any iterable (list, tuple, set)
    """
    n = len(*args)
    ii = tolerance_interval_locs(n, t, c)
    # Need to subtract 1 from the places, to account for 0-based index
    return (
        tuple(sorted(*args)[slice(ii[0] - 1, ii[1], ii[1] - ii[0])]) if ii else None  # type: ignore
    )


def tolerance_interval_locs(n: int, t: float, c: float) -> Optional[tuple[int, int]]:
    """Returns tolerance interval locations. Out of n sorted samples, a fraction of t samples
    (0 < t < 1) are expected to be within these two places, with a probability of at least c,
    0 < c < 1.

    Returns None if such tuple cannot be calculated. If that happens, try to increase n,
    reduce t, or reduce c.
    """
    if _percentile_invalid(t) or _confidence_invalid(c) or _num_samples_invalid(n):
        return None

    # Construct the interval (g, h) in two halves, each with t/2 fraction around median.
    # Let c_hi = probability that h out of n samples are smaller than quantile (0.5+t/2)
    # Let c_lo = probability that g out of n samples are smaller than quantile (0.5-t/2)
    # 1 - c_lo = probability that g out of n samples are higher than quantile (0.5-t/2)
    # Then c_hi*(1 - c_lo) = probability that between (g, h) samples out of n are
    # within quantile t

    # At the high end, start at the median and expand until confidence exceeds c for
    # quantile of 0.5 + t/2.
    median_index = (n + 1) // 2
    j_hi = median_index
    p_hi = 0.5 + t / 2
    while confidence_in_percentile(j_hi, n, p_hi) < c:
        j_hi += 1
        if j_hi == n + 1:
            logger.error(
                "Not enough samples, %d, for tolerance %f at confidence %f.", n, t, c
            )
            return None
    c_hi = confidence_in_percentile(j_hi, n, p_hi)
    logger.debug(
        "p_hi = %f, j_hi = %d, c_hi = %f, c_lo < %f", p_hi, j_hi, c_hi, 1 - c / c_hi
    )

    # Now start at zero index and quantile of 0.5 - t/2
    # c_hi * (1 - c_lo) >= c
    # c_lo <=  1 - c / c_hi
    j_lo = 0
    p_lo = 0.5 - t / 2
    c_lo = confidence_in_percentile(j_lo, n, p_lo)
    logger.debug("p_lo = %f, j_lo = %d, c_lo = %f", p_lo, j_lo, c_lo)
    while c_lo := confidence_in_percentile(j_lo, n, p_lo) < 1 - c / c_hi:
        logger.debug("p_lo = %f, j_lo = %d, c_lo = %f", p_lo, j_lo, c_lo)
        j_lo += 1
        if j_lo == median_index:
            logger.error(
                "Not enough samples, %d, for tolerance %f, confidence %f", n, t, c
            )
            return None

    return (j_lo + 1, j_hi)


def assurance_interval(a: float, *args) -> Optional[tuple[Any, Any]]:
    """Returns assurance interval for middle a (0<a<1) fraction of samples, if possible.
    Same as tolerance interval for fraction a with confidence a.
    Use this method if you data is not sorted already, else you can use assurance_interval_places.
    Returns None if not possible.
    args is any iterable (list, tuple, set)
    """
    n = len(*args)
    ii = assurance_interval_locs(n, a)
    # Need to subtract 1 from the places, to account for 0-based index
    return (
        tuple(sorted(*args)[slice(ii[0] - 1, ii[1], ii[1] - ii[0])]) if ii else None  # type: ignore
    )


def assurance_interval_locs(n: int, a: float) -> Optional[tuple[int, int]]:
    """Returns assurance interval locations. Out of n sorted samples, a fraction of a samples
    are expected to be within these two locations, with a probability of at least a.

    Returns None if such tuple cannot be calculated. If that happens, try to increase n
    or reduce a.
    """
    return tolerance_interval_locs(n, a, a)


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
        logger.error("Sample places %d, %d not between 0 and %d", j_lo, j_hi, n - 1)
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


def _assurance_interval_fn(x: float, j_lo: int, j_hi: int, n: int) -> float:
    """Function to find roots of x = confidence_in_quantile(n, f, x)"""
    x_hi_hat = confidence_in_percentile(j_hi, n, x) or 0
    x_lo_hat = confidence_in_percentile(j_lo, n, 1 - x) or 0
    x_hat = min(x_hi_hat, 1 - x_lo_hat)
    return x_hat - x
