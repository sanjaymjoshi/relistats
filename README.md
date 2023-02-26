# Reliability Statistics

The relistat library provides methods to compute the reliability engineering statistics
for success-failure experiments. The samples are assumed to be independent and
identically distributed per binomial distributions.

- Reliability is probability of success. For estimating reliability with a finite
  number of samples, we need to provide the level of confidence. This library provides:
  - `binomial.reliability` uses `binomial.reliability_optim` with default accuracy tolerance
  - `binomial.reliability_optim` uses Brent's method to numerically compute the value,
    with tunable accuracy tolerance 
  - `binomial.reliability_closed` uses closed-form approximation using Wilson Score
     Interval with Continuity Correction

- Confidence in reliability is probability that the actual reliability of the
  population is at least the provided reliability level. This library provides:
  - `binomial.confidence` uses closed-form exact expression

- Assurance simplifies reliability and confidence by setting both of them the same.
  This library provides:
  - `binomial.assurance` uses Brent's method to numerically compute the value,
    with tunable accuracy tolerance

See [TBD paper](tbd) for technical details.

See also [relistats_notebook](https://github.com/sanjaymjoshi/relistats_notebook)
project for how to use this library in a notebook, which can be used online
on [Google Colab](https://colab.research.google.com/github/sanjaymjoshi/relistats_notebook/blob/main/relistats_binomial.ipynb).

## Setup
```sh
# Install dependencies
pipenv install --dev

# Setup pre-commit and pre-push hooks
pipenv run pre-commit install -t pre-commit
pipenv run pre-commit install -t pre-push
```

## Credits
This package was created with Cookiecutter and the [sourcery-ai/python-best-practices-cookiecutter](https://github.com/sourcery-ai/python-best-practices-cookiecutter) project template.
