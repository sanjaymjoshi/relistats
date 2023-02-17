from math import sqrt
from typing import Optional

import scipy.optimize as opt
import scipy.stats as st


def confidence(n: int, f: int, r: float) -> Optional[float]:
    """Returns confidence [0, 1] in reliability r
    n -- number of samples, >= 0
    f -- number of failures >= 0
    r -- reliability level [0, 1]
    Returns None if confidence could not be computed
    """
    if n <= 0 or f < 0 or r < 0 or r > 1:
        return None
    # Scipy's binom object provides 'survival function', which is 1 - CDF.
    prob_failure = 1 - r
    return st.binom.sf(f, n, prob_failure)


# Wilson score interval from
# https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
# Wallis, Sean A. (2013). "Binomial confidence intervals and contingency tests: mathematical fundamentals and the evaluation of alternative methods".
# Journal of Quantitative Linguistics. 20 (3): 178–208.


def _wilson_center(p, n, c):
    """Center of Wilson score interval"""
    z = st.norm.ppf(c)
    return (p + z * z / (2 * n)) / (1 + z * z / n)


def _wilson_lower(p, n, c):
    """Lower bound of Wilson score interval"""
    z = st.norm.ppf(c)
    p50 = _wilson_center(p, n, c)
    part2 = z / (1 + z * z / n) * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p50 - part2


def _wilson_lower_corrected(p, n, c):
    """Lower bound of Wilson score interval, with continuity correction"""
    return _wilson_lower(max(p - 1 / (2 * n), 0), n, c)


def reliability_closed(n: int, f: int, c: float) -> Optional[float]:
    """Returns approximate minimum reliability [0, 1] at confidence level c
    The approximation is within about 5% of actual reliability and uses
    closed-form expression for computation
    n -- number of samples, >= 0
    f -- number of failures >= 0
    c -- confidence level [0, 1]
    Returns None if reliability could not be computed
    """
    if n <= 0 or f < 0 or c < 0 or c > 1:
        return None

    return _wilson_lower_corrected((n - f) / n, n, c)


def _reliability_fn(x: float, n: int, f: int, c: float) -> float:
    """Function to find roots of c = confidence(n, f, x)"""
    c_hat = confidence(n, f, x) or 0
    return c_hat - c


def reliability_optim(n: int, f: int, c: float, tol=0.001) -> Optional[float]:
    """Returns minimum reliability [0, 1] at confidence level c using numerical optimization
    The approximation is within about specified tolerance limit
    n -- number of samples, >= 0
    f -- number of failures >= 0
    c -- confidence level [0, 1]
    tol -- [optional] accurancy tolerance limit]
    Returns None if reliability could not be computed
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


def reliability(n: int, f: int, c: float) -> Optional[float]:
    """Minimum reliability at confidence level c
    n -- number of samples, >= 0
    f -- number of failures >= 0
    c -- confidence level [0, 1]
    Returns None if reliability could not be computed
    """
    return reliability_optim(n, f, c)


def _assurance_fn(x: float, n: int, f: int) -> float:
    """Function to find roots of x = confidence(n, f, x)"""
    c = confidence(n, f, x) or 0
    return x - c


def assurance(n: int, f: int, tol=0.001) -> Optional[float]:
    """Returns assurance [0, 1], i.e., confidence = reliability
    n -- number of samples, >= 0
    f -- number of failures >= 0
    tol -- [optional] accurancy tolerance limit
    Returns None if assurance could not be computed
    """
    if n <= 0 or f < 0:
        return None
    # Use brentq method to find real root of the assurance equation
    # a = c = r. Meaning a = confidence(n, f, a)
    return opt.brentq(
        _assurance_fn,
        a=0,  # Lowest possible value
        b=1,  # Highest possible value
        args=(
            n,
            f,
        ),
        xtol=tol,
    )
