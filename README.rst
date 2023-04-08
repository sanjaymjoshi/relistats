Reliability Statistics
======================

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

This library provides methods to calculate these statistics.

Example usage in a python file:

.. code-block:: python

   from relistats.binomial import assurance
   
   n = 22
   a = assurance(n, 0) or 0
   print(f"Assurance at {n} good samples: {a*100:.1f}%")

See

- Paper: S.M. Joshi, "Computation of Reliability Statistics for Success-Failure Experiments,"
  `arXiv:2303.03167 [stat.ME] <https://doi.org/10.48550/arXiv.2303.03167>`_, March 2023.

- Jupyter notebook showing how to use this library:
  `relistats_notebook <https://github.com/sanjaymjoshi/relistats_notebook>`_

- Interactive online version of the Jupyter notebook on
  `Google Colab <https://colab.research.google.com/github/sanjaymjoshi/relistats_notebook/blob/main/relistats_binomial.ipynb>`_.

Additional documentation:

- `Usage <docs/source/usage.rst>`_ for installation and how to use.

- `Background <docs/source/background.rst>`_ for concepts and mathematical background.

Credits
----------
This package was created with Cookiecutter and the
`sourcery-ai/python-best-practices-cookiecutter
<https://github.com/sourcery-ai/python-best-practices-cookiecutter>`_
project template.
