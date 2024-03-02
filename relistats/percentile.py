"""Statistical methods for percentiles or quantiles and tolerance interval.

Reference:
S.M. Joshi, "Confidence and Assurance of Percentiles," arXiv:2402.19109 [stat.ME], Feb 2024.
https://doi.org/10.48550/arXiv.2402.19109
"""

from typing import Optional

import scipy.optimize as opt
import scipy.stats as stats

from relistats import logger


def confidence_in_percentile(j: int, n: int, p: float) -> float:
    """Returns confidence (probability) that in a population of n samples,
    pp^th percentile/quantile (0 < p < 1) is greater than j samples, 1 <= j <= n.

    From https://online.stat.psu.edu/stat415/lesson/19/19.2

    .. math::
        c = \sum_{k=0}^{j-1} {n\choose k}  p^k  (1-p)^{n-k}

    This is same as cumulative density function for a binomial
    distribution, evaluated at j-1 out of n samples.

    Note that j=n+1 will return 1.
    """
    return stats.binom.cdf(j - 1, n, p)


def _num_samples_invalid(n: int) -> bool:
    n_min = 3
    if n < n_min:
        logger.error("Need at least %d samples, found: %d", n_min, n)
        return True
    return False


def _assurance_percentile_fn(x: float, j: int, n: int) -> float:
    """Function to find roots of x = confidence_in_quantile(n, f, x)"""
    x_hat = confidence_in_percentile(j, n, x) or 0
    return x_hat - x


def assurance_in_percentile(j: int, n: int, tol=0.001) -> Optional[float]:
    """Assurance level at j'th index out of n sorted samples. The confidence
       is at least the percentile/quantile level.

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
        _assurance_percentile_fn,
        a=0,  # Lowest possible value
        b=1,  # Highest possible value
        args=(j, n),
        xtol=tol,
    )
