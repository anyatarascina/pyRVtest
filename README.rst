pyRVtest
========

.. docs-start

This code was written to perform the procedure for testing firm conduct developed in `"Testing Firm Conduct" <https://arxiv.org/abs/2301.06720>`_ by Marco Duarte, Lorenzo Magnolfi, Mikkel Sølvsten, and Christopher Sullivan. It builds on the PyBLP source code (see `Conlon and Gortmaker (2020) <https://onlinelibrary.wiley.com/doi/full/10.1111/1756-2171.12352>`_) - to do so.

The code implements the following features:

* Computes `Rivers and Vuong (2002) <https://onlinelibrary.wiley.com/doi/full/10.1111/1368-423X.t01-1-00071>`_ (RV) test statistics to test a menu of two or more models of firm conduct allowing for the possibility that firms or consumers face per unit or ad-valorem taxes.
* Implements the RV test using the variance estimator of `Duarte, Magnolfi, Sølvsten, and Sullivan (2023) <https://arxiv.org/abs/2301.06720>`_, including options to adjust for demand estimation error and clustering
* Computes the effective F-statistic proposed in `Duarte, Magnolfi, Sølvsten, and Sullivan (2023) <https://arxiv.org/abs/2301.06720>`_ to diagnose instrument strength with respect to worst-case size and best-case power of the test, and reports appropriate critical values
* Reports `Hansen, Lunde, and Nason (2011) <https://www.jstor.org/stable/41057463?seq=1#metadata_info_tab_contents>`_ MCS p-values for testing more than two models
* Compatible with PyBLP `Conlon and Gortmaker (2020) <https://onlinelibrary.wiley.com/doi/full/10.1111/1756-2171.12352>`_, so that demand can be estimated with PyBLP, and the estimates are an input to the test for conduct

For a full list of references, see the references in `Duarte, Magnolfi, Sølvsten, and Sullivan (2023) <https://arxiv.org/abs/2301.06720>`_.


Install
_______

First, you will need to download and install python, which you can do from this `link <https://www.python.org/>`_.

You will also need to make sure that you have all package dependencies installed.

To install the pyRVtest package, use pip:

.. code-block::

    pip install pyRVtest

This should automatically install the python packages on which pyRVtest depends: numpy, pandas, statsmodels, pyblp

To update to a newer version of the package use:


.. code-block::

    pip install --upgrade pyRVtest


Using the package
_________________

For a detailed tutorial about how to set up and run the testing procedure, see `Tutorial <https://pyrvtest.readthedocs.io/en/stable/tutorial.html>`_.


Citing the package
__________________

When using the package, please include the following citation:

Duarte, M., L. Magnolfi, M. Sølvsten, C. Sullivan, and A. Tarascina
(2023): “pyRVtest: A Python package for testing firm conduct,” https://github.com/anyatarascina/pyRVtest.

@misc{
pyrvtest,
author={Marco Duarte and Lorenzo Magnolfi and Mikkel S{\o}lvsten and Christopher Sullivan and Anya Tarascina},
title={\texttt{pyRVtest}: A Python package for testing firm conduct},
howpublished={\url{https://github.com/anyatarascina/pyRVtest}},
year={2023}

Bugs and Requests
_________________

Please use the `GitHub issue tracker <https://github.com/anyatarascina/pyRVtest/issues>`_ to submit bugs or to request features.
