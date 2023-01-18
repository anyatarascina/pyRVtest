pyRVtest
========

.. docs-start

.. note::
    This package is currently in the process of being updated. The updated tutorial no longer applies to older
    versions of the package code. An updated version of the package will be available for download by end of January 2023.

This code was written to perform the procedure for testing firm conduct developed in :ref:`references: Duarte, Magnolfi, Sølvsten, and Sullivan (2023)`.  It builds on the PyBLP source code (see `Conlon and Gortmaker (2020) <https://onlinelibrary.wiley.com/doi/full/10.1111/1756-2171.12352>`_) - to do so.

The code implements the following features:

* Computes :ref:`references:Rivers, D., & Vuong, Q. (2002)` (RV) test statistics to test a menu of two or more models of firm conduct allowing for the possibility that firms or consumers face per unit or ad-valorem taxes.
* Implements the RV test using the variance estimator of :ref:`references: Duarte, Magnolfi, Sølvsten, and Sullivan (2023)`, including options to adjust for demand estimation error and clustering.
* Computes the effective F-statistic proposed in :ref:`references: Duarte, Magnolfi, Sølvsten, and Sullivan (2023)` to diagnose instrument strength with respect to worst-case size and best-case power of the test, and reports appropriate critical values.
* Reports :ref:`references: Hansen, Lunde, and Nason (2011)` MCS p-values for testing more than two models.
* Compatible with PyBLP (:ref:`references: Conlon and Gortmaker (2020)`), so that demand can be estimated with PyBLP, and the estimates are an input to the test for conduct.

For a full list of references, see the references in `references: Duarte, Magnolfi, Sølvsten, and Sullivan (2023)`.


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

For a detailed tutorial about how to set up and run the testing procedure, see [link to tutorial notebook]


Citing the package
__________________

When using the package, please include the following citation:

Duarte, M., L. Magnolfi, M. Sølvsten, C. Sullivan, and A. Tarascina
(2022): “pyRVtest: A Python package for testing firm conduct,” https://github.com/anyatarascina/pyRVtest.

@misc{
    pyrvtest,
    author={Marco Duarte and Lorenzo Magnolfi and Mikkel S{\o}lvsten and Christopher Sullivan and Anya Tarascina},
    title={\texttt{pyRVtest}: A Python package for testing firm conduct},
    howpublished={\url{https://github.com/anyatarascina/pyRVtest}},
    year={2022}
}
