pyRVtest
========

.. docs-start

This code was written to perform the procedure for testing firm conduct developed in `"Testing Firm Conduct" <https://arxiv.org/abs/2301.06720>`_ by Marco Duarte, Lorenzo Magnolfi, Mikkel Sølvsten, and Christopher Sullivan. It builds on the PyBLP source code (see `Conlon and Gortmaker (2020) <https://onlinelibrary.wiley.com/doi/full/10.1111/1756-2171.12352>`_) - to do so.

The code implements the following features:

* Computes `Rivers and Vuong (2002) <https://onlinelibrary.wiley.com/doi/full/10.1111/1368-423X.t01-1-00071>`_ (RV) test statistics to test a menu of two or more models of firm conduct allowing for the possibility that firms or consumers face per-unit or ad-valorem taxes.
* Implements the RV test using the variance estimator of `Duarte, Magnolfi, Sølvsten, and Sullivan (2023) <https://arxiv.org/abs/2301.06720>`_, including options to adjust for demand estimation error and clustering.
* Computes the effective F-statistic proposed in `Duarte, Magnolfi, Sølvsten, and Sullivan (2023) <https://arxiv.org/abs/2301.06720>`_ to diagnose instrument strength with respect to worst-case size and best-case power of the test, and reports appropriate critical values.
* Reports `Hansen, Lunde, and Nason (2011) <https://www.jstor.org/stable/41057463?seq=1#metadata_info_tab_contents>`_ MCS p-values for testing more than two models.
* Ships a class-based ``ConductModel`` API (``Bertrand``, ``Cournot``, ``Monopoly``, ``PerfectCompetition``, ``MixCournotBertrand``, ``Vertical``) plus Dearing, Magnolfi, Quint, Sullivan, and Waldfogel (2026) simple-markup models (``RuleOfThumb(phi)``, ``ConstantMarkup(markup)``).
* Exposes pass-through diagnostics on ``ProblemResults`` (``passthrough_comparison``, ``passthrough_matrix``) implementing the Dearing et al. (2026) distinguishability conditions.
* **Experimental** — Supports labor-side conduct testing via ``Problem(market_side='labor')`` with ``Monopsony``, ``BertrandWages``, and ``CournotEmployment`` model classes. The labor API is considered experimental in v0.4: the sign convention, column-name defaults, and validation behavior may adjust based on coauthor review (``pyRVtest/models/labor.py`` is flagged for a labor-market-conduct-manuscript sign check). ``NashBargaining`` raises ``NotImplementedError`` and the full ``LaborSupplyBackend`` math (Jacobian, Hessian, demand-adjustment participation) is deferred to v0.5.
* Provides instrument construction helpers (``pyRVtest.instruments.product``: BLP, differentiation IVs, rival sums; ``pyRVtest.instruments.labor``: Hausman, Bartik).
* Compatible with PyBLP `Conlon and Gortmaker (2020) <https://onlinelibrary.wiley.com/doi/full/10.1111/1756-2171.12352>`_, so that demand can be estimated with PyBLP, and the estimates are an input to the test for conduct. A ``DemandBackend`` protocol also supports user-supplied demand systems; see ``docs/custom_demand.rst``.

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


Reader's guide
______________

Where to go next, by audience:

* **End users** running the test on their own data — `Tutorial <https://pyrvtest.readthedocs.io/en/stable/tutorial.html>`_ (worked notebooks).
* **Migrating from v0.3** — ``docs/migrating_to_v0.4.rst``. The class-based ``ConductModel`` API, ``demand_params=dict(rho=...)``, and Problem-level ``unit_tax`` / ``advalorem_tax`` kwargs are all new in v0.4; the legacy per-model ``ModelFormulation``, ``sigma`` alias, and per-model tax kwargs continue to work for one or two releases with ``DeprecationWarning``.
* **Custom demand backend** — ``docs/custom_demand.rst`` covers the ``DemandBackend`` protocol.
* **AI coding assistants and contributors** — ``AGENTS.md`` at the repo root is the living contract for code state, layout, and conventions. ``docs/agent_guide.rst`` is the longer architecture walkthrough (also surfaced via ``pyRVtest.show_agent_guide()``).


Citing the package
__________________

When using ``pyRVtest`` in research, please cite the package itself plus the methodology papers your usage exercises.

**Package:**

Duarte, M., L. Magnolfi, M. Sølvsten, C. Sullivan, and A. Tarascina (2023): “pyRVtest: A Python package for testing firm conduct,” https://github.com/anyatarascina/pyRVtest.

**Methodology papers:**

* For the Rivers-Vuong test, F-statistic, MCS p-values, and demand-adjustment correction:

  Duarte, M., L. Magnolfi, M. Sølvsten, and C. Sullivan (2023): `“Testing Firm Conduct,” <https://arxiv.org/abs/2301.06720>`_ Working paper.

* For pass-through diagnostics, simple-markup models (``RuleOfThumb``, ``ConstantMarkup``), and the instrument-relevance / falsification framework:

  Dearing, A., L. Magnolfi, D. Quint, C. Sullivan, and S. Waldfogel (2026): `“Learning Firm Conduct: Pass-Through as a Foundation for Instrument Relevance,” <https://www.nber.org/papers/w32863>`_ NBER Working Paper No. 32863.

* For the endogenous-cost-component first-stage correction (non-linear cost):

  Duarte, M., L. Magnolfi, D. Quint, M. Sølvsten, and C. Sullivan (2026): “Testing Firm Conduct with Non-Linear Cost,” Working paper.

BibTeX:

.. code-block:: bibtex

    @misc{pyrvtest,
        author={Marco Duarte and Lorenzo Magnolfi and Mikkel S{\o}lvsten and Christopher Sullivan and Anya Tarascina},
        title={\texttt{pyRVtest}: A Python package for testing firm conduct},
        howpublished={\url{https://github.com/anyatarascina/pyRVtest}},
        year={2023}
    }

    @article{dmss2023,
        author={Marco Duarte and Lorenzo Magnolfi and Mikkel S{\o}lvsten and Christopher Sullivan},
        title={Testing Firm Conduct},
        howpublished={\url{https://arxiv.org/abs/2301.06720}},
        year={2023}
    }

    @techreport{dmqsw2026,
        author={Adam Dearing and Lorenzo Magnolfi and Daniel Quint and Christopher Sullivan and Sarah Waldfogel},
        title={Learning Firm Conduct: Pass-Through as a Foundation for Instrument Relevance},
        institution={National Bureau of Economic Research},
        type={NBER Working Paper},
        number={32863},
        year={2026}
    }

    @article{dmqss2026,
        author={Marco Duarte and Lorenzo Magnolfi and Daniel Quint and Mikkel S{\o}lvsten and Christopher Sullivan},
        title={Testing Firm Conduct with Non-Linear Cost},
        year={2026}
    }

Bugs and Requests
_________________

Please use the `GitHub issue tracker <https://github.com/anyatarascina/pyRVtest/issues>`_ to submit bugs or to request features.
