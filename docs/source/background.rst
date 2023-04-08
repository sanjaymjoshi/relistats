Background
==========

Computation of the concepts described in `README <README.rst>`_ depend on
the properties of underlying random variables.

- For samples with binary results, such as pass/fail or success/failure, we
  typically assume *binomial distribution*.

Binomial Distribution
----------------------

If in :math:`n` samples, we find :math:`f` failures, then the confidence in
reliability :math:`r, 0 \le r \le 1` is

.. math::
    c = 1 - \sum_{k=0}^f \binom{n}{k}  (1-r)^k r^{n-k},

where :math:`\binom{n}{k} = \frac{n!}{k!(n-k)!}` is the binomial coefficient.

Calculating reliability means solving the above equation for :math:`r`. Yes,
it is complicated. That is why we have :meth:`relistats.binomial.reliability`
method that uses
numerical optimization! There are closed-form approximations and this library
implements the 'Wilson Score Interval with Continuity Correction' method via
:meth:`relistats.binomial.reliability_closed`.

Calculating assurance means setting :math:`r = c = a` in the above equation
to get

.. math::
    a = 1 - \sum_{k=0}^f \binom{n}{k}  (1-a)^k a^{n-k}

Solving the above equation for :math:`a` is not trivial. This library
offers a numerical method :meth:`relistats.binomial.assurance` that allows
tuning desired accuracy level.

See paper `Computation of Reliability Statistics for Success-Failure Experiments
<https://doi.org/10.48550/arXiv.2303.03167>`_ for more information.

