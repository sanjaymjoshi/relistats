import scipy.optimize as opt
import scipy.stats as stats
from typing import Optional

def confidence_in_quantile(k:int, n:int, q:float) -> float:
    """ Confidence that quantile q (0 < q < 1) is at less than k'th index out of sorted n samples
    """
    return 1 - stats.binom.sf(k, n, q) or 0


def _assurance_quantile_fn(x: float, k: int, n: int) -> float:
    """Function to find roots of x = confidence_in_quantile(n, f, x)"""
    x_hat = confidence_in_quantile(k, n, x) or 0
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
    if n <= 1 or k <= 0 or k > n-1:
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

def index_at_quantile(n: int, q: float, c: Optional[float]=None) -> Optional[int]:
    """Find minimum index (0-based) out of n samples, such that the confidence in
    quantile level q is at least c.

    If c is left as None, c = q is assumed 
    """
    # sourcery skip: use-next
    if c is None:
        c = q
    for k in range(1, n):
        if confidence_in_quantile(k, n, q) > c:
            return k
    return None


if __name__ == "__main__":
    n = 20
    for k in range(1, n):
        print(f"{k}: {assurance_in_quantile(k,n)}")
    q = 0.5
    print(f"index at q={q}: {index_at_quantile(n, q)} out of {n}")
    c = 0.95
    print(f"At c={c}: {index_at_quantile(n, q, c)} out of {n}")