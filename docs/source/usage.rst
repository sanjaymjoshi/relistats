Usage
=====
Installation
------------

To use relistats, first install it using pip:

.. code-block:: console

   (.venv) $ pip install relistats

How to use
----------

In a python file:

.. code-block:: python

   from relistats.binomial import assurance
   
   n = 22
   a = assurance(n, 0) or 0
   print(f"Assurance at {n} good samples: {a*100:.1f}%")


See also `relistats_notebook <https://github.com/sanjaymjoshi/relistats_notebook>`_
project for how to use this library in a notebook, which can be used online
on `Google Colab
<https://colab.research.google.com/github/sanjaymjoshi/relistats_notebook/blob/main/relistats_binomial.ipynb>`_.
