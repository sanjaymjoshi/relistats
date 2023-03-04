Background
==========

Reliability engineering deals with estimating parameters or qualities of a product or
process or experiment. For simplicity, we assume that all units of a product or results
of an experiment are random variables. Collectively, let's call them samples.
We assume that the samples are independent (one sample has no effect on another
sample) and identically distributed (the reliability or properties of underlying
random variable stay the same for each sample).

Concepts
--------
- Reliability is probability of success. The math assumes infinite number of samples,
  but we can get access to only a finite number of samples. Therefore, we can compute
  only an estimate of the actual reliability. Based on the number of samples, we
  qualify the quality of this estimate using *confidence*.

- Confidence in reliability is probability that the actual reliability of the
  population is at least the provided reliability level. 
  For example, we can say "If we see zero failures in 10 samples of a success-failure
  experiment, we have 95% confidence that the reliability is at least about 74%".

- Assurance simplifies reliability and confidence by setting both of them the same.
  The result is just one number that is easier to communicate. For example, 90%
  assurance means 90% reliability with 90% confidence. Given the number of samples
  and number of failures, assurance is just one number.

The calculations and concepts change a bit based on the properties of underlying
random variables.

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
<https://arxiv.org/submit/4768869/view>`_ for more information.

