""" Reliability Engineering Statistics for Binomial Distributions

Also known as Bernoulli Trials.

Reference:
S.M. Joshi, "Computation of Reliability Statistics for
Success-Failure Experiments," arXiv:2303.03167 [stat.ME], March 2023.
https://doi.org/10.48550/arXiv.2303.03167
"""
from math import sqrt, floor
from typing import Optional

import scipy.optimize as opt
import scipy.stats as st


def confidence(n: int, f: int, r: float, m: int=None) -> Optional[float]:
    """Confidence [0, 1] in reliability r using closed-form expression.

    :param n: number of samples
    :type n: int, >=0
    :param f: number of failures
    :type f: int, >=0
    :param r: reliability level
    :type r: float, [0, 1]
    :param m: remaining samples in population (None for infinite)
    :type m: int, >= 0, optional
    :return: Confidence or None if it could not be computed
    :rtype: float, optional
    """
    if n <= 0 or f < 0 or r < 0 or r > 1:
        return None
    r_needed = r

    if m is not None:        
        # Finite population case
        if m < 0:
            return None
        total_samples = n + m
        max_f_at_r = floor(total_samples * (1-r) )

        num_failures = max_f_at_r - f # number of failures we can afford
        num_samples = m # in these many samples
        
        if num_failures < 0:
            # got too many failures already, zero confidence
            return 0
        
        if num_failures >= m:
            # even if all remaining samples fail, we are still ok. Full confidence.
            return 1
        
        if num_failures == 0:
            # Cannot calculate probability of zero failures, hence bump up the 
            # remaining samples by 1 and calculate probability that there is exactly
            # 1 failure
            num_samples = num_samples + 1
            num_failures = 1

        r_needed = 1 - num_failures / num_samples

    # Scipy's binom object provides 'survival function', which is 1 - CDF.
    prob_failure = 1 - r_needed
    return st.binom.sf(f, n, prob_failure)



def _wilson_center(p, n, c):
    """Center of Wilson score interval. See reference below."""
    z = st.norm.ppf(c)
    return (p + z * z / (2 * n)) / (1 + z * z / n)


def _wilson_lower(p, n, c):
    """Lower bound of Wilson score interval. See reference below."""
    z = st.norm.ppf(c)
    p50 = _wilson_center(p, n, c)
    part2 = z / (1 + z * z / n) * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p50 - part2


def _wilson_lower_corrected(p, n, c):
    """Lower bound of Wilson score interval, with continuity correction."""
    return _wilson_lower(max(p - 1 / (2 * n), 0), n, c)


def reliability_closed(n: int, f: int, c: float) -> Optional[float]:
    """Approximate minimum reliability [0, 1] at confidence level c.
    The approximation is within about 5% of actual reliability and uses
    closed-form expression for computation called 'Wilson Score Interval
    with Continuity Correction' [Wallis, Sean A. (2013). "Binomial
    confidence intervals and contingency tests: mathematical fundamentals
    and the evaluation of alternative methods". Journal of Quantitative
    Linguistics. 20 (3): 178–208].

    :param n: number of samples
    :type n: int, >=0
    :param f: number of failures
    :type f: int, >=0
    :param c: confidence level
    :type c: float, [0, 1]
    :return: Reliability or None if it could not be computed
    :rtype: float, optional
    """
    if n <= 0 or f < 0 or c < 0 or c > 1:
        return None

    return _wilson_lower_corrected((n - f) / n, n, c)


def _reliability_fn(x: float, n: int, f: int, c: float) -> float:
    """Function to find roots of c = confidence(n, f, x)"""
    c_hat = confidence(n, f, x) or 0
    return c_hat - c


def reliability_optim(n: int, f: int, c: float, tol=0.001) -> Optional[float]:
    """Minimum reliability [0, 1] at confidence level c using numerical
    optimization (Brent's method). The approximation is within specified
    tolerance limit.

    :param n: number of samples
    :type n: int, >=0
    :param f: number of failures
    :type f: int, >=0
    :param c: confidence level
    :type c: float, [0, 1]
    :param tol: accuracy tolerance
    :type tol: float, optional

    :return: Reliability or None if it could not be computed
    :rtype: float, optional
    """
    if n <= 0 or f < 0 or c < 0 or c > 1:
        return None

    # Use numerical optimization to find real root of the confidence equation
    # c - confidence(n, f, r)
    return opt.brentq(
        _reliability_fn,
        a=0,  # Lowest possible value
        b=1,  # Highest possible value
        args=(n, f, c),
        xtol=tol,
    )


def reliability(n: int, f: int, c: float, m: int=None) -> Optional[float]:
    """Minimum reliability at confidence level c

    :param n: number of samples
    :type n: int, >=0
    :param f: number of failures
    :type f: int, >=0
    :param c: confidence level
    :type c: float, [0, 1]
    :param m: remaining samples in population (None for infinite)
    :type m: int, >= 0, optional
    :return: Reliability or None if it could not be computed
    :rtype: float, optional
    """
    if m is None:
        # Infinite population case
        return reliability_optim(n, f, c)

    # Finite population case
    if m < 0:
        return None
    
    # Calculate confidence for each case of remaining failures
    # Return the highest reliability
    total_samples = n+m
    for f2 in range(m):
        r = 1 - (f+f2) / total_samples
        c2 = confidence(n, f, r, m)
        print(n, f, r, m, c2)
        if c2 >= c:
            return  r
    return 0
        

        


def _assurance_fn(x: float, n: int, f: int) -> float:
    """Function to find roots of x = confidence(n, f, x)"""
    c = confidence(n, f, x) or 0
    return x - c


def assurance(n: int, f: int, tol=0.001) -> Optional[float]:
    """Assurance [0, 1], i.e., confidence = reliability. For example,
    90% assurance means 90% confidence in 90% reliability (at n=22, f=0).
    This method uses numerical approach of Brent's method to compute
    the solution within the specified tolerance.

    :param n: number of samples
    :type n: int, >=0
    :param f: number of failures
    :type f: int, >=0
    :param tol: accuracy tolerance
    :type tol: float, optional
    :return: Assurance or None if it could not be computed
    :rtype: float, optional
    """
    if n <= 0 or f < 0:
        return None
    # Use brentq method to find real root of the assurance equation
    # a = c = r. Meaning a = confidence(n, f, a)
    return opt.brentq(
        _assurance_fn,
        a=0,  # Lowest possible value
        b=1,  # Highest possible value
        args=(n, f),
        xtol=tol,
    )
