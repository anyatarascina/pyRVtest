pyRVtest
========

.. docs-start

This code was written to perform the procedure for testing firm conduct developed in `"Testing Firm Conduct" <https://arxiv.org/abs/2301.06720>`_ by Marco Duarte, Lorenzo Magnolfi, Mikkel Sølvsten, and Christopher Sullivan. It builds on the PyBLP source code (see `Conlon and Gortmaker (2020) <https://onlinelibrary.wiley.com/doi/full/10.1111/1756-2171.12352>`_) - to do so.

The code implements the following features:

* Computes `Rivers and Vuong (2002) <https://onlinelibrary.wiley.com/doi/full/10.1111/1368-423X.t01-1-00071>`_ (RV) test statistics to test a menu of two or more models of firm conduct allowing for the possibility that firms or consumers face per-unit or ad-valorem taxes.
* Implements the RV test using the variance estimator of `Duarte, Magnolfi, Sølvsten, and Sullivan (2023) <https://arxiv.org/abs/2301.06720>`_, including options to adjust for demand estimation error and clustering.
* Computes the effective F-statistic proposed in `Duarte, Magnolfi, Sølvsten, and Sullivan (2023) <https://arxiv.org/abs/2301.06720>`_ to diagnose instrument strength with respect to worst-case size and best-case power of the test, and reports appropriate critical values.
* Reports `Hansen, Lunde, and Nason (2011) <https://www.jstor.org/stable/41057463?seq=1#metadata_info_tab_contents>`_ MCS p-values for testing more than two models.
* Ships a class-based ``ConductModel`` API: standard oligopoly (``Bertrand``, ``Cournot``, ``Monopoly``, ``PerfectCompetition``, ``MixCournotBertrand``), generalizations (``PartialCollusion``, ``Vertical``), Dearing, Magnolfi, Quint, Sullivan, and Waldfogel (2024) simple-markup models (``RuleOfThumb(phi)``, ``ConstantMarkup(markup)``), and customization escape hatches (``UserSuppliedMarkups`` for precomputed markup columns, ``CustomConductModel`` for arbitrary markup callables).
* Ships the full Dearing, Magnolfi, Quint, Sullivan, and Waldfogel (2024) pass-through framework as a diagnostic suite on ``Problem`` and ``ProblemResults``: ``passthrough_summary`` (pre-solve γ-free pair-by-pair structural-feature distances against four DMQSW-keyed metrics), ``passthrough_matrix`` (raw per-candidate pass-through matrix, computed numerically for every conduct class with analytical fast paths for ``Vertical`` and trivial conducts), and ``instrument_channels`` (post-solve channel decomposition for one chosen IV column). Under non-constant marginal cost (``endogenous_cost_component`` set), ``instrument_channels`` automatically applies DMQSS Appendix B's z^e residualization, producing a single unified diagnostic that collapses the Dearing condition and the DMQSS Appendix A.4 distinctness check.
* Supports ``endogenous_cost_component`` as either a single column name (single endogenous cost variable, the original v0.4 case) or a list of column names (multi-endogenous variables — quadratic cost ``['q', 'q_sq']``, scale + scope ``['log_q', 'log_Q_minus']``, etc., per DMQSS Appendix A.4). Combinable with ``demand_adjustment=True`` and ``costs_type='log'``.
* **Experimental** — Supports labor-side conduct testing via ``Problem(market_side='labor')`` with ``Monopsony``, ``BertrandWages``, and ``CournotEmployment`` model classes. The labor API is considered experimental in v0.4: the sign convention, column-name defaults, and validation behavior may adjust based on coauthor review (``pyRVtest/models/labor.py`` is flagged for a labor-market-conduct-manuscript sign check). ``NashBargaining`` raises ``NotImplementedError`` and the full ``LaborSupplyBackend`` math (Jacobian, Hessian, demand-adjustment participation) is deferred to v0.5.
* Provides instrument construction helpers (``pyRVtest.instruments.product``: BLP, differentiation IVs, rival sums; ``pyRVtest.instruments.labor``: Hausman, Bartik).
* Compatible with PyBLP `Conlon and Gortmaker (2020) <https://onlinelibrary.wiley.com/doi/full/10.1111/1756-2171.12352>`_, so that demand can be estimated with PyBLP, and the estimates are an input to the test for conduct. A ``DemandBackend`` protocol also supports user-supplied demand systems; see ``docs/custom_demand.rst``.

For a full list of references, see the references in `Duarte, Magnolfi, Sølvsten, and Sullivan (2023) <https://arxiv.org/abs/2301.06720>`_.


Install
_______

First, you will need to download and install python, which you can do from this `link <https://www.python.org/>`_.
pyRVtest v0.4 requires Python ``>=3.9``.

You will also need to make sure that you have all package dependencies installed.

**Release candidate (v0.4):** PyPI still serves the older v0.3 line; the v0.4 release candidate is installed from the GitHub tag:

.. code-block::

    pip install git+https://github.com/anyatarascina/pyRVtest@v0.4.0rc8

The v0.4 series will be uploaded to PyPI once it leaves release-candidate status.

**Older v0.3 release:** still available from PyPI for users on the
prior public API:

.. code-block::

    pip install pyRVtest

To update to a newer version of the package use:


.. code-block::

    pip install --upgrade pyRVtest

Dependencies (auto-installed): ``numpy``, ``pandas``, ``statsmodels``,
``pyblp``, ``patsy``, ``scipy``, ``sympy``, ``jinja2``. Note: ``numpy
>= 2`` requires ``pyblp >= 1.2``; pyRVtest enforces this at import
time with a clean ``ImportError`` if the resolved combination is
inconsistent.


Quick start
___________

A complete pyRVtest run on a synthetic dataset shipped with the package
(2 single-product firms × 3000 markets, simulated under perfect
competition with logit demand). Tests four candidate conduct models
(Bertrand, Cournot, Monopoly, Perfect Competition) using rival cost
shifters as testing instruments per :ref:`references: Dearing, Magnolfi, Quint, Sullivan, and Waldfogel (2024)`.

.. code-block:: python

    import pyRVtest

    data = pyRVtest.data.load_example()
    results = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + z1 + z2'),
        instrument_formulation=pyRVtest.Formulation('0 + rival_z1 + rival_z2'),
        models=[
            pyRVtest.Bertrand(ownership='firm_ids'),
            pyRVtest.Cournot(ownership='firm_ids'),
            pyRVtest.Monopoly(),
            pyRVtest.PerfectCompetition(),
        ],
        product_data=data,
        demand_params={'estimate': 'logit',
                       'formulation_X': pyRVtest.Formulation('1 + x1'),
                       'formulation_Z': pyRVtest.Formulation('0 + z1')},
    ).solve(demand_adjustment=False)
    print(results)

Output::

    Testing Results - Instruments z0:
    ===============================================================================================================
      TRV:                                 |   F-stats:                                   |    MCS:
    --------  ---  -----  -------  ------  |  ----------  ---  -------  -------  -------  |  --------  ------------
     models    0     1       2       3     |    models     0      1        2        3     |   models   MCS p-values
    --------  ---  -----  -------  ------  |  ----------  ---  -------  -------  -------  |  --------  ------------
       0      nan  6.894  -9.103   6.771   |      0       nan   92.8     170.2    2.6     |     0          0.0
                    ***     ***     ***    |                   ††† ^^^  ††† ^^^  ††† ^^^  |
       1      nan   nan   -10.555  0.419   |      1       nan    nan     178.4    0.0     |     1         0.675
                            ***            |                            ††† ^^^   †††     |
       2      nan   nan     nan    10.593  |      2       nan    nan      nan      1.4    |     2          0.0
                                    ***    |                                     ††† ^^^  |
       3      nan   nan     nan     nan    |      3       nan    nan      nan      nan    |     3          1.0
    ===============================================================================================================

Models 0/1/2/3 correspond to Bertrand / Cournot / Monopoly / Perfect
Competition (in the order passed to ``models=``). Each block has rows
and columns indexed by model number; cell ``[i, j]`` shows the pairwise
statistic for model ``i`` vs. model ``j``. The diagonal is ``nan`` (a
model is not compared to itself); off-diagonals carry significance
markers (``*`` 10%, ``**`` 5%, ``***`` 1% for TRV; ``†``/``^`` for
F-stat size/power thresholds). The MCS column is per-model.

The truth in this dataset is Perfect Competition. Reading the output:

* **Bertrand** (model 0) and **Monopoly** (model 2) are cleanly
  rejected: their MCS p-values are both 0.0, and every pairwise TRV
  involving them is significant at 1% (``***``).
* **Perfect Competition** (model 3) has the highest MCS p-value (1.0).
* **Cournot** (model 1) has MCS p-value 0.675 — surviving the
  confidence set despite not being the truth. The TRV(Cournot, PC) cell
  is the only insignificant comparison (0.419), and its F-stat is
  effectively zero. This is the Dearing et al. (2024) degeneracy
  result: under logit demand, both Cournot and PC have diagonal
  pass-through matrices (zero off-diagonal pass-through of rival
  costs), so rival cost shifters cannot distinguish them. To falsify
  Cournot in favor of PC here, the researcher would need a different
  instrument such as own or rival product characteristics.

Reader's guide
______________

Where to go next, by audience:

* **End users** running the test on their own data — `Tutorial <https://pyrvtest.readthedocs.io/en/stable/tutorial.html>`_ (worked notebooks).
* **Migrating from v0.3** — ``docs/migrating_to_v0.4.rst``. The class-based ``ConductModel`` API, ``demand_params=dict(rho=...)``, and Problem-level ``unit_tax`` / ``advalorem_tax`` kwargs are all new in v0.4; the legacy per-model ``ModelFormulation``, ``sigma`` alias, and per-model tax kwargs continue to work for one or two releases with ``DeprecationWarning``.
* **Custom demand backend** — ``docs/custom_demand.rst`` covers the ``DemandBackend`` protocol.
* **AI coding assistants and contributors** — ``AGENTS.md`` at the repo root is the living contract for code state, layout, and conventions. ``docs/agent_guide.rst`` is the longer architecture walkthrough (also surfaced via ``pyRVtest.show_agent_guide()``). ``CONTRIBUTING.md`` covers dev environment setup, running tests / lint / docs locally, and the conventions for adding new conduct models or demand backends.


Citing the package
__________________

When using ``pyRVtest`` in research, please cite the package itself plus the methodology papers your usage exercises.

**Package:**

Duarte, M., L. Magnolfi, M. Sølvsten, C. Sullivan, and A. Tarascina (2023): “pyRVtest: A Python package for testing firm conduct,” https://github.com/anyatarascina/pyRVtest.

**Methodology papers:**

* For the Rivers-Vuong test, F-statistic, MCS p-values, and demand-adjustment correction:

  Duarte, M., L. Magnolfi, M. Sølvsten, and C. Sullivan (2023): `“Testing Firm Conduct,” <https://arxiv.org/abs/2301.06720>`_ Working paper.

* For pass-through diagnostics, simple-markup models (``RuleOfThumb``, ``ConstantMarkup``), and the instrument-relevance / falsification framework:

  Dearing, A., L. Magnolfi, D. Quint, C. Sullivan, and S. Waldfogel (2024): `“Learning Firm Conduct: Pass-Through as a Foundation for Instrument Relevance,” <https://www.nber.org/papers/w32863>`_ NBER Working Paper No. 32863, August 2024.

* For the endogenous-cost-component first-stage correction (non-linear cost):

  Duarte, M., L. Magnolfi, D. Quint, M. Sølvsten, and C. Sullivan (2026): “Conduct and Scale Economies: Evaluating Tariffs in the US Automobile Market,” Working paper.

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

    @techreport{dmqsw2024,
        author={Adam Dearing and Lorenzo Magnolfi and Daniel Quint and Christopher Sullivan and Sarah Waldfogel},
        title={Learning Firm Conduct: Pass-Through as a Foundation for Instrument Relevance},
        institution={National Bureau of Economic Research},
        type={NBER Working Paper},
        number={32863},
        month={August},
        year={2024}
    }

    @article{dmqss2026,
        author={Marco Duarte and Lorenzo Magnolfi and Daniel Quint and Mikkel S{\o}lvsten and Christopher Sullivan},
        title={Conduct and Scale Economies: Evaluating Tariffs in the US Automobile Market},
        year={2026}
    }

Bugs and Requests
_________________

Please use the `GitHub issue tracker <https://github.com/anyatarascina/pyRVtest/issues>`_ to submit bugs or to request features.
