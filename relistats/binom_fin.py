""" Reliability Engineering Statistics for Binomial Distributions

Also known as Bernoulli Trials.

Reference:
S.M. Joshi, "Computation of Reliability Statistics for
Success-Failure Experiments," arXiv:2303.03167 [stat.ME], March 2023.
https://doi.org/10.48550/arXiv.2303.03167
"""
from math import floor
from relistats.binomial import confidence


def conf_fin(n: int, f: int, r: float, m: int) -> tuple:
    """Confidence [0, 1] in reliability r for finite population size.
    Returns tuple with second value as actual reliability used for computations.

    :param n: number of samples
    :type n: int, >=0
    :param f: number of failures
    :type f: int, >=0
    :param r: reliability level
    :type r: float, [0, 1]
    :param m: remaining samples in population
    :type m: int, >= 0
    :return: Tuple of (confidence, actual reliability)
    :rtype: tuple
    """
    if n <= 0 or f < 0 or r < 0 or r > 1:
        return (None, r)

    # Finite population case
    if m < 0:
        return (None, r)
    
    total_samples = n + m
    max_f_at_r = floor(total_samples * (1-r) )
    actual_r = 1 - max_f_at_r/total_samples

    num_failures = max_f_at_r - f # number of failures we can afford
    num_samples = m # in these many samples

    if num_failures < 0:
        # got too many failures already, zero confidence
        return (0, actual_r)
    
    if num_failures >= m:
        # even if all remaining samples fail, we are still ok. Full confidence.
        return (1, actual_r)
    
    if num_failures == 0:
        # Cannot calculate probability of zero failures, hence bump up the 
        # remaining samples by 1 and calculate probability that there is exactly
        # 1 failure
        num_samples = num_samples + 1
        num_failures = 1
        total_samples += 1
        actual_r = 1 - max_f_at_r / total_samples
        
    r_needed = 1 - num_failures / num_samples

    return (confidence(n, f, r_needed), actual_r)


def reli_fin(n: int, f: int, c: float, m: int) -> tuple:
    """Minimum reliability at confidence level c for finite population size.
    Returns tuple with second value as actual confidence used for computations.

    :param n: number of samples
    :type n: int, >=0
    :param f: number of failures
    :type f: int, >=0
    :param c: confidence level
    :type c: float, [0, 1]
    :param m: remaining samples in population
    :type m: int, >= 0
    :return: (reliability, actual confidence)
    """
    if n <= 0 or f < 0 or c < 0 or c > 1:
        return (None, c)
    
    # Calculate confidence for each case of remaining failures
    # Start with 0 failures, i.e., highest reliability possible.
    # The confidence will be lowest at this level. If the 
    # desired confidence is higher than this, keep increasing
    # failures, i.e., keep reducing reliability until the 
    # desired confidence level is met or exceeded.
    # Return that reliability (or 0 if it is not possible to
    # achieve the desired level of confidence)
    total_samples = n+m
    for f2 in range(m+1):
        r = 1 - (f+f2) / total_samples
        c2, r2 = conf_fin(n, f, r, m)
        if c2 >= c:
            return (r2, c2)
    return (0, c)
        

def assur_fin(n: int, f: int, m: int, tol=0.001) -> tuple:
    """Assurance [0, 1], i.e., confidence = reliability.
    Returns tuple with other values as reliability and confidence
    used for computations.

    :param n: number of samples
    :type n: int, >=0
    :param f: number of failures
    :type f: int, >=0
    :param tol: accuracy tolerance
    :param m: remaining samples in population
    :type m: int, >= 0
    :type tol: float, optional
    :return: (Assurance, reliability, confidence)
    :rtype: tuple
    """
    if n <= 0 or f < 0:
        return (None, 0, 0)
    
    # Calculate confidence for each case of remaining failures
    # Start with 0 failures, i.e., highest reliability possible.
    # The confidence will be lowest at this level. Set assurance
    # as the minimum of reliability and confidence. Keep increasing
    # failures, i.e., keep reducing reliability which will increase
    # the confidence. Keep doing this while the assurance keeps
    # increasing.
    # Return that assurance
    
    max_assurance = 0
    max_reli = 0
    max_conf = 0
    total_samples = n+m
    for f2 in range(m+1):
        r = 1 - (f+f2) / total_samples
        c2, r2 = conf_fin(n, f, r, m)
        assurance = min([r2, c2])
        if assurance > max_assurance:
            max_assurance = assurance
            max_reli = r2
            max_conf = c2
    return (max_assurance, max_reli, max_conf)
