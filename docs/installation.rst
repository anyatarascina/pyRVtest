Installation
============

``pyRVtest`` is distributed on PyPI:

.. code-block:: bash

    pip install pyRVtest

To upgrade:

.. code-block:: bash

    pip install --upgrade pyRVtest

This installs the dependencies ``numpy``, ``pandas``, ``statsmodels``,
and ``pyblp`` automatically. pyRVtest supports both ``numpy<2`` (with
``pyblp<1.2``) and ``numpy>=2`` (with ``pyblp>=1.2``); pip resolves a
compatible pair on a fresh install.


Development install
___________________

To install from source for development:

.. code-block:: bash

    git clone git@github.com:anyatarascina/pyRVtest.git
    cd pyRVtest
    pip install -e .

See :doc:`agent_guide` and ``CONTRIBUTING.md`` for the contributor
architecture overview, lint / test / docs commands, and the
conventions for adding new conduct models or demand backends.
