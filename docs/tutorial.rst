.. _tutorial_reference:

Tutorial
========

This page walks through a complete ``pyRVtest`` run end-to-end on the
synthetic example dataset shipped with the package, then points to the
deeper tutorials for specific topics.

Step-by-step walkthrough
------------------------

The shipped example
:func:`pyRVtest.data.load_example` provides 3000 markets of duopoly
data simulated under perfect-competition pricing with logit demand.
See :func:`pyRVtest.data.simulate_example` for the full DGP.

Step 1 — Load the data
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pyRVtest

   data = pyRVtest.data.load_example()
   data.head()

Output (first five rows of 6000)::

      market_ids  firm_ids    shares    prices       x1        z1        z2  rival_z1  rival_z2
   0           0         0  0.609383  ...        ...                                            ...
   1           0         1  0.119941  ...
   2           1         0  0.307389  ...
   3           1         1  0.441629  ...
   4           2         0  0.409602  ...

The columns are: market and firm identifiers; equilibrium shares and
prices; one demand-side characteristic (``x1``); two own cost shifters
(``z1``, ``z2``); and the rival sums of those shifters (``rival_z1``,
``rival_z2``) for use as testing instruments.

Step 2 — Estimate demand
~~~~~~~~~~~~~~~~~~~~~~~~

The DGP is plain logit, so :class:`~pyRVtest.LogitEstimator` is the
right choice. It runs Berry inversion plus 2SLS using ``z1`` as the
excluded instrument for price.

.. code-block:: python

   estimator = pyRVtest.LogitEstimator(
       product_data=data,
       formulation_X=pyRVtest.Formulation('1 + x1'),
       formulation_Z=pyRVtest.Formulation('0 + z1'),
   )
   demand_params = estimator.solve()
   print(demand_params['alpha'], demand_params['beta'])

Output::

   -0.9986 [2.0149 0.9955]

The DGP truth is :math:`(\alpha, \beta_0, \beta_1) = (-1, 2, 1)`; the
estimator recovers each within standard 2SLS noise. ``demand_params``
is the populated dict that
:class:`~pyRVtest.Problem` consumes in step 3.

(For random-coefficients / BLP-style demand, you would estimate with
``pyblp`` and pass ``demand_results=`` to :class:`~pyRVtest.Problem`
instead. See :doc:`_notebooks/testing_firm_conduct` for a worked
example using PyBLP-estimated demand on the Nevo cereal data.)

Step 3 — Construct a Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hand the demand parameters, the cost-shifter formulation, the testing-
instrument formulation, and a list of candidate conduct models to
:class:`~pyRVtest.Problem`.

.. code-block:: python

   problem = pyRVtest.Problem(
       cost_formulation=pyRVtest.Formulation('1 + z1 + z2'),
       instrument_formulation=pyRVtest.Formulation('0 + rival_z1 + rival_z2'),
       models=[
           pyRVtest.Bertrand(ownership='firm_ids'),
           pyRVtest.Cournot(ownership='firm_ids'),
           pyRVtest.PerfectCompetition(),
       ],
       product_data=data,
       demand_params=demand_params,
   )

What each argument does:

* ``cost_formulation`` — exogenous variables that explain marginal
  cost. ``Problem`` residualizes prices, markups, and instruments on
  these before computing the test statistics.
* ``instrument_formulation`` — the testing instruments, here the rival
  cost shifters from the dataset (a Dearing et al. 2024 falsification
  instrument set). Pass a list of :class:`~pyRVtest.Formulation` if
  you want the test run on multiple instrument sets.
* ``models`` — the list of candidate :class:`~pyRVtest.ConductModel`
  instances to compare. Each implies its own markup formula.
* ``demand_params`` — the dict from step 2 (or a hand-built one, or
  the inline-estimator shortcut described in
  :doc:`in_package_demand`).

Step 4 — Solve and read the results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   results = problem.solve(demand_adjustment=False)
   print(results)

The output table reports, per instrument set and per pair of candidate
models: the Rivers-Vuong test statistic ``TRV``, the DMSS scaled
F-statistic, and the Hansen-Lunde-Nason MCS p-values. Numerical
example for this run::

   TRV[0]:                          MCS p-values[0]:
       0     1     2                    0    0.000
   0   nan   6.89  6.77                 1    0.675
   1   nan   nan   0.42                 2    1.000
   2   nan   nan   nan

(Models 0 = Bertrand, 1 = Cournot, 2 = PerfectCompetition. Diagonal
``nan`` is by design; cells ``[i, j]`` with ``i > j`` are filled in
the upper triangle.)

Reading the result:

* PerfectCompetition (model 2) — the truth — has MCS p-value 1.0
  and rejects Bertrand (TRV = 6.77) at every conventional level.
* Bertrand (model 0) is rejected: MCS p-value 0.0.
* Cournot (model 1) survives the MCS at p = 0.675 even though it is
  not the truth. The TRV(Cournot, PC) cell is insignificant at 0.42.
  This is the Dearing et al. (2024) degeneracy: under logit demand,
  both Cournot and PerfectCompetition have diagonal pass-through
  matrices, so rival cost shifters cannot distinguish them. To falsify
  Cournot relative to PC here, the researcher would need a different
  testing instrument such as own or rival product characteristics.

Step 5 — Inspect the diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~pyRVtest.ProblemResults` object holds more than the
printed table. Three diagnostic methods cover the DMSS reliability
view and the DMQSW pass-through framework:

.. code-block:: python

   # DMSS empirical reliability
   results.reliability_summary()              # per-cell F, rho^2, size/power CVs

   # DMQSW pass-through framework
   results.passthrough_summary()              # pre-/post-solve γ-free pair distances
   results.passthrough_matrix(0, market_id=0) # raw P_m for one (model, market)
   results.instrument_channels(column='rival_z2', instrument='rival_cost')

   # Implied quantities
   results.markups        # implied markups per candidate model
   results.marginal_cost  # implied marginal costs per candidate model
   results.to_dataframe() # long-form DataFrame for export

The DMQSW framework view in
:meth:`~pyRVtest.ProblemResults.passthrough_summary` identifies the
(Cournot, PerfectCompetition) degenerate pair from Step 4 *ex-ante*::

   Per-pair pass-through-feature distances (median across 3000 markets):

                             pair  offdiag_ratio  full_pass  row_sum  level_adj
              (Bertrand, Cournot)   1.267916e-01   0.173728 0.207784   0.343116
             (Bertrand, Monopoly)   2.025077e+00   0.800202 0.813289   0.661342
   (Bertrand, PerfectCompetition)   1.267916e-01   0.685104 0.378482   1.924238
              (Cournot, Monopoly)   1.799031e+00   0.755743 0.361467   0.489735
    (Cournot, PerfectCompetition)   1.335208e-09   0.837912 0.837912   2.162204
   (Monopoly, PerfectCompetition)   1.799031e+00   0.958163 1.134994   2.252656

The ``(Cournot, PerfectCompetition)`` row's ``offdiag_ratio`` is
``1.3e-9``, numerical zero. This is the headline DMQSW result: under
logit demand, both Cournot and PerfectCompetition have diagonal
pass-through matrices, so rival cost shifters cannot distinguish them
— no matter how much data you collect. The other three columns
(``full_pass``, ``row_sum``, ``level_adj``) are nonzero for the same
pair, telling you which other instrument types *could* break the
degeneracy: own-and-rival cost, per-unit taxes, ad-valorem taxes.

See :ref:`advanced-passthrough` for the post-solve cross-read against
``reliability_summary`` and the per-pair channel decomposition via
``instrument_channels``.

See :doc:`api` for the full attribute / method reference and
:doc:`math` for the formulas behind ``TRV``, ``F``, ``MCS``, and the
pass-through framework.

Tutorials by topic
------------------

Beyond the walkthrough above, three tutorials cover specific entry
points in more depth.

Testing firm conduct on real data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :doc:`_notebooks/testing_firm_conduct` notebook walks through a
complete conduct test on the Nevo (2000, 2001) cereal data. It
estimates BLP demand with ``pyblp``, hands the demand results to
:class:`~pyRVtest.Problem`, runs the test for a Bertrand-vs-Cournot
comparison under several instrument sets, and shows how to read the
significance markers and reliability footers in the printed table.
Read this when you have your own demand estimates and want to see the
full PyBLP-driven workflow on a published dataset.

In-package demand estimation (logit and nested logit)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :doc:`in_package_demand` page covers the
:class:`~pyRVtest.LogitEstimator` and
:class:`~pyRVtest.NestedLogitEstimator` interfaces, including the
inline shortcut form ``demand_params={'estimate': 'logit', ...}`` and
the auto-constructed within-nest IV for nested logit. Read this when
your demand system fits the linear-2SLS scope and you want to skip
running ``pyblp``.

The conduct-model library
~~~~~~~~~~~~~~~~~~~~~~~~~

The :doc:`_notebooks/model_library` notebook is a reference catalogue
of every conduct model class that ships with ``pyRVtest``: standard
oligopoly (:class:`~pyRVtest.Bertrand`, :class:`~pyRVtest.Cournot`,
:class:`~pyRVtest.Monopoly`, :class:`~pyRVtest.PerfectCompetition`,
:class:`~pyRVtest.MixCournotBertrand`,
:class:`~pyRVtest.PartialCollusion`), the Dearing et al. (2024)
simple-markup models (:class:`~pyRVtest.RuleOfThumb`,
:class:`~pyRVtest.ConstantMarkup`),
:class:`~pyRVtest.UserSuppliedMarkups` for hand-supplied markup
columns, and :class:`~pyRVtest.Vertical` for downstream-upstream
bilateral oligopoly. Read this when you need to know what each class
expects, what its FOC looks like, and how to choose between similar
options.

Pre-test framework reasoning per DMQSW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :ref:`advanced-passthrough` section of :doc:`advanced_features`
walks through the Dearing, Magnolfi, Quint, Sullivan, and Waldfogel
(2026) pre-test framework on the synthetic example. The narrative
covers:

* Building the :class:`~pyRVtest.Problem` and inspecting
  :meth:`~pyRVtest.Problem.passthrough_summary` *before* calling
  ``solve``, to identify pairs that are structurally indistinguishable
  under the chosen IV bundle (the (Cournot, PerfectCompetition)
  ``offdiag_ratio = 0`` result of Step 4 above).
* Cross-reading the framework view against
  :meth:`~pyRVtest.ProblemResults.reliability_summary` post-solve to
  diagnose structural-vs-empirical weakness.
* Decomposing the per-pair difference in candidates' implied causal
  effects of one IV column via
  :meth:`~pyRVtest.Problem.instrument_channels`.

Read this when you want to choose testing instruments deliberately
based on which structural feature each candidate pair makes salient,
rather than running the test against an arbitrary IV bundle and
post-rationalizing.

Beyond the tutorials
--------------------

* :doc:`migrating_to_v0.4` — for users coming from v0.3.
* :doc:`custom_demand` — protocol for plugging in a demand system
  pyRVtest does not estimate natively (anything that is not plain
  logit, one-level nested logit, or pyblp).
* :doc:`faq` — common errors, output interpretation, and known caveats.
* :doc:`api` — full API reference.

.. toctree::
   :hidden:

   _notebooks/testing_firm_conduct.ipynb
   _notebooks/model_library.ipynb
