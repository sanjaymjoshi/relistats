Background
==========

Computation of the concepts described in `README <../../../README.rst>`_ depend on
the properties of underlying random variables.

For samples with binary results, such as pass/fail or success/failure, we
typically assume *binomial distribution*.

Infinite population size
------------------------

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

Finite population size
----------------------

For finite population size, after :math:`n`` samples are tested, we already know
how many additional samples, :math:`m` remain. The acceptable number of failures
in these to reach desired reliability level and the associated confidence level
can be computed assuming infinite population size. This technique is implemented
in :meth:`relistats.binom_fin.conf_fin`.

Using the confidence method, reliability and assurance levels can be computed as
shown in :mod:`relistats.binom_fin`.

See references in `README <../../../README.rst>`_ for more information.

