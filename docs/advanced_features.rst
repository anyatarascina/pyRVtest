Advanced features
=================

A tour of v0.4 features that go beyond the basic four-stage workflow
in the :doc:`tutorial`. Each section is self-contained and uses the
shipped example dataset (:func:`pyRVtest.data.load_example`); pick the
sections relevant to your application.

The features covered:

#. :ref:`Multiple instrument sets <advanced-multi-iv>` — running the
   test under several testing-IV bundles in one call.
#. :ref:`Demand adjustment and clustering <advanced-da-cluster>` —
   first-stage and clustering corrections to the test variance.
#. :ref:`F-stat reliability inspection <advanced-f-reliability>` —
   confidence intervals on the F-statistic and the per-cell verdict.
#. :ref:`Problem-level taxes <advanced-taxes>` — unit and ad-valorem
   taxes that affect every candidate model's FOC.
#. :ref:`Pass-through diagnostics <advanced-passthrough>` — the
   Villas-Boas matrix and the DMQSW Remark 4 distinguishability
   metric (``Vertical``-only in v0.4).
#. :ref:`Endogenous cost components <advanced-endog-cost>` — the DMQSS
   first-stage correction when marginal cost depends on quantity (or
   another endogenous variable).


.. _advanced-multi-iv:

Multiple instrument sets
------------------------

Pass ``instrument_formulation`` as a *list* of
:class:`~pyRVtest.Formulation` objects. The test runs once per
instrument set and the results arrays are indexed by the position of
each set in the list.

.. code-block:: python

   import pyRVtest

   data = pyRVtest.data.load_example()

   results = pyRVtest.Problem(
       cost_formulation=pyRVtest.Formulation('1 + z1 + z2'),
       instrument_formulation=[
           pyRVtest.Formulation('0 + rival_z1 + rival_z2'),
           pyRVtest.Formulation('0 + rival_z1'),
       ],
       models=[
           pyRVtest.Bertrand(ownership='firm_ids'),
           pyRVtest.Cournot(ownership='firm_ids'),
           pyRVtest.PerfectCompetition(),
       ],
       product_data=data,
       demand_params={'estimate': 'logit',
                      'formulation_X': pyRVtest.Formulation('1 + x1'),
                      'formulation_Z': pyRVtest.Formulation('0 + z1')},
   ).solve(demand_adjustment=False)
   print(results)

The printed output now contains two tables, labeled ``z0`` and ``z1``,
one per instrument set::

   Testing Results - Instruments z0:
   ============================================================================================
     TRV:                       |   F-stats:                          |    MCS:
     ...                                  ...                                  ...
   ============================================================================================

   Testing Results - Instruments z1:
   ==========================================================================================
     TRV:                       |   F-stats:                        |    MCS:
     ...                                  ...                                  ...
   ==========================================================================================

Programmatic access mirrors the indexing: ``results.TRV[0]`` is the
TRV matrix for instrument set 0 (``rival_z1 + rival_z2``);
``results.TRV[1]`` is for set 1 (``rival_z1`` alone). The same indexing
applies to ``results.F``, ``results.MCS_pvalues``, and the
:meth:`~pyRVtest.ProblemResults.F_reliability_summary` columns.

When to use multiple instrument sets:

* Comparing the test's verdict under different bundles of testing
  instruments (e.g., rival cost shifters vs. BLP-style rival
  characteristics) per :ref:`references: Dearing, Magnolfi, Quint, Sullivan, and Waldfogel (2026)`.
* Sensitivity to which excluded instruments are dropped.
* Combining a "narrow" (high-power) set with a "wide" (more robust to
  weak-instrument concerns) set in the same run.


.. _advanced-da-cluster:

Demand adjustment and clustering
--------------------------------

When the demand parameters are estimated rather than fixed, the
moment-difference variance underlying the RV statistic and the F-stat
should account for first-stage estimation uncertainty. Following
:ref:`references: Duarte, Magnolfi, Sølvsten, and Sullivan (2023)`
Appendix C (linear cost) and DMQSS Appendix B (non-linear cost), the
demand-adjusted variance adds a term capturing
:math:`\partial\bar g_m / \partial\theta` evaluated at the demand
parameter estimates. Activate it with ``demand_adjustment=True``.

The clustering correction multiplies the variance estimate by the
standard cluster-robust factor; specify the cluster column on the
product data via the ``clustering_ids`` column name and pass
``clustering_adjustment=True``.

.. code-block:: python

   import pyRVtest

   data = pyRVtest.data.load_example()
   data['clustering_ids'] = data['market_ids']  # cluster at market level

   results = pyRVtest.Problem(
       cost_formulation=pyRVtest.Formulation('1 + z1 + z2'),
       instrument_formulation=pyRVtest.Formulation('0 + rival_z1 + rival_z2'),
       models=[
           pyRVtest.Bertrand(ownership='firm_ids'),
           pyRVtest.Cournot(ownership='firm_ids'),
           pyRVtest.PerfectCompetition(),
       ],
       product_data=data,
       demand_params={'estimate': 'logit',
                      'formulation_X': pyRVtest.Formulation('1 + x1'),
                      'formulation_Z': pyRVtest.Formulation('0 + z1')},
   ).solve(demand_adjustment=True, clustering_adjustment=True)

The TRV statistics shift modestly relative to the uncorrected output:
in this example the (Bertrand, Cournot) cell moves from
:math:`T_{RV} = 6.894` (no corrections) to :math:`7.013` (with both),
reflecting the small downward adjustment to the moment-difference
variance once first-stage uncertainty and within-market correlation
are accounted for. The MCS p-values and significance markers are
materially unchanged on this dataset.

When to use:

* ``demand_adjustment=True`` whenever you estimate demand and care
  about valid inference. The flag is off by default to keep the
  baseline output light, but for any application paper you should
  turn it on.
* ``clustering_adjustment=True`` if your data has within-cluster
  dependence (markets, time periods, geographies). Set the
  ``clustering_ids`` column on the product data to whatever level you
  want to cluster at.


.. _advanced-f-reliability:

F-stat reliability inspection
-----------------------------

The ``F`` matrix in the printed output is one number per (model pair,
instrument set). The DMSS scaled F-statistic has an asymptotic
distribution at the implied non-centrality parameter, which the
package uses to compute a 95% confidence interval and a per-cell
*verdict* about whether the F-stat is robust to sampling noise
(``robust``) or borderline (``borderline``).
:meth:`~pyRVtest.ProblemResults.F_reliability_summary` returns a
long-form ``pandas.DataFrame`` with one row per pair of candidate
models for each instrument set:

.. code-block:: python

   results = ...  # from any Problem.solve() call
   df = results.F_reliability_summary()
   keep = ['model_i_label', 'model_j_label',
           'F', 'rho_squared', 'F_ci_low', 'F_ci_high', 'verdict']
   print(df[keep].to_string(index=False))

Output on the shipped-example three-model run::

   model_i_label       model_j_label         F  rho_squared  F_ci_low  F_ci_high verdict
        bertrand             cournot 92.848681     0.327266 77.386264 108.311098  robust
        bertrand perfect_competition  2.551670     0.947320  1.836781   3.266559  robust
         cournot perfect_competition  0.006054     0.975397 -0.028044   0.040151  robust

What the columns mean:

* ``F`` — the DMSS scaled F-statistic.
* ``rho_squared`` — :math:`\hat\rho^2`, the moment-pair correlation
  whose proximity to 1 (the Cauchy-Schwarz boundary) drives numerical
  fragility.
* ``F_ci_low`` / ``F_ci_high`` — 95% asymptotic CI for the population F
  at the implied non-centrality.
* ``F_high_precision`` — populated when the package detects
  :math:`\hat\rho^2` close to 1 and recomputes :math:`F` with mpmath
  at extra precision (NaN otherwise; a footnote in the printed table
  flags affected pairs).
* ``strongest_claim_size`` / ``strongest_claim_power`` — the strongest
  worst-case size (or best-case power) claim the F passes, e.g.
  ``"worst-case size <= 7.5%"``.
* ``worst_case_cv_size`` / ``worst_case_cv_power`` — the critical
  values used for the size and power decisions, indexed by
  :math:`K_{\text{eff}}` and :math:`\hat\rho^2`.
* ``verdict`` — ``robust`` if the CI on the F-stat lies entirely on
  one side of the relevant CV; ``borderline`` if the CI straddles a
  decision boundary.

Read the F-stat alongside its CI: a point F that just clears a CV may
be borderline once sampling noise is accounted for. The
:meth:`~pyRVtest.ProblemResults.F_reliability_summary` DataFrame is
ordered the same way as the printed table — model pairs in the upper
triangle :math:`(i, j)` with :math:`i < j`, broken out by instrument
set in the ``instrument_set`` column.


.. _advanced-taxes:

Problem-level taxes
-------------------

When firms or consumers face per-unit or ad-valorem taxes, the
candidate model's FOC must be adjusted to map prices to marginal cost
correctly (DMQSW eq 3 with the retention factor :math:`\nu` and unit
tax :math:`\tau`). pyRVtest accepts tax columns at the
:class:`~pyRVtest.Problem` level: every candidate model inherits them
unless an individual model opts out via a ``salience`` flag.

.. code-block:: python

   import pyRVtest

   data = pyRVtest.data.load_example()
   data['unit_tax'] = 0.5    # constant per-unit tax in this example

   results = pyRVtest.Problem(
       cost_formulation=pyRVtest.Formulation('1 + z1 + z2'),
       instrument_formulation=pyRVtest.Formulation('0 + rival_z1 + rival_z2'),
       models=[
           pyRVtest.Bertrand(ownership='firm_ids'),
           pyRVtest.PerfectCompetition(),
       ],
       product_data=data,
       demand_params={'estimate': 'logit',
                      'formulation_X': pyRVtest.Formulation('1 + x1'),
                      'formulation_Z': pyRVtest.Formulation('0 + z1')},
       unit_tax='unit_tax',           # column name on product_data
   ).solve(demand_adjustment=False)

Both ``unit_tax`` and ``advalorem_tax`` are accepted at the
:class:`~pyRVtest.Problem` level. ``advalorem_tax`` requires a paired
``advalorem_payer`` argument set to ``'firm'`` or ``'consumer'``.
Ad-valorem rates are fractions, not percentages.

**Salience tests.** Set ``unit_tax_salient=False`` on a candidate
model to test the hypothesis that *that model* ignores the tax (firms
or consumers do not respond as if the tax shifts the relevant price):

.. code-block:: python

   models = [
       pyRVtest.Bertrand(ownership='firm_ids'),                       # salient by default
       pyRVtest.Bertrand(ownership='firm_ids', unit_tax_salient=False),  # tax-blind
   ]

The same flag exists for ``advalorem_tax_salient``. The Problem-level
tax columns supply the data; the per-model salience flags choose which
candidates use them in their FOC.

Legacy v0.3 code passed ``unit_tax='col'`` / ``advalorem_tax='col'``
on individual ``ModelFormulation`` instances. That pattern still works
in v0.4 with a ``DeprecationWarning`` and is scheduled for removal in
v0.6. See :doc:`migrating_to_v0.4` for the rewrite recipe.


.. _advanced-passthrough:

Pass-through diagnostics (Vertical models, v0.4)
------------------------------------------------

DMQSW's identification framework rests on the model-implied cost
pass-through matrix
:math:`\mathcal{P}_{mt} = \partial p_t / \partial mc_t`: rival cost
shifters target the off-diagonal-to-diagonal ratio,
product characteristics target the full matrix, and tax instruments
target row sums. Two ProblemResults methods expose pass-through:

* :meth:`~pyRVtest.ProblemResults.passthrough_matrix` — the
  Villas-Boas (2007) :math:`\mathcal{P}_{mt}` for a single model and
  market.
* :meth:`~pyRVtest.ProblemResults.passthrough_comparison` — the DMQSW
  Remark 4 pairwise distance between two candidate models'
  pass-through matrices (Frobenius, max-element, or row-sum).

.. note::

   In v0.4, both methods require **all candidate models in the
   :class:`~pyRVtest.Problem` to be :class:`~pyRVtest.Vertical`
   instances** — the closed-form Villas-Boas matrix the package
   computes is for the downstream-upstream Bertrand-Bertrand case.
   Calling either method on a non-Vertical model raises a clear
   ``ValueError`` / ``NotImplementedError``. Numerical pass-through
   for Bertrand / Cournot / RuleOfThumb / ConstantMarkup is on the
   v0.5 roadmap; in the meantime the closed-form expressions for these
   models are documented in :doc:`math` and in DMQSW.

The shipped-example data has only ``Bertrand`` / ``Cournot`` /
``PerfectCompetition`` candidates, so the methods are not directly
demonstrable here. See the
:func:`pyRVtest.construct_passthrough_matrix` standalone helper if you
want to compute :math:`\mathcal{P}_{mt}` directly from your own
inputs (ownership matrix, price-share Jacobian).


.. _advanced-endog-cost:

Endogenous cost components
--------------------------

When marginal cost depends on an endogenous variable — most commonly
quantity, under scale economies — fitting the cost formulation with
ordinary least squares produces biased coefficients. The DMQSS
extension addresses this with a two-step IV correction: estimate the
endogenous cost coefficient by 2SLS, then fold the correction into
the moment / variance computation. Activate it on
:class:`~pyRVtest.Problem` with the ``endogenous_cost_component``
argument:

.. code-block:: python

   problem = pyRVtest.Problem(
       cost_formulation=pyRVtest.Formulation('1 + z1 + z2 + log_quantity'),
       instrument_formulation=pyRVtest.Formulation('0 + rival_z1 + rival_z2'),
       models=[...],
       product_data=data,
       demand_params=...,
       endogenous_cost_component='log_quantity',  # column name
   )
   results = problem.solve(demand_adjustment=False)

The named column must already appear in the cost formulation; the
package treats it as endogenous, runs the first-stage IV correction
(DMQSS Appendix B), and returns the standard
:class:`~pyRVtest.ProblemResults` with the corrected moments.

The shipped example does not include a quantity-driven cost component
(prices in :func:`pyRVtest.data.load_example` are simulated under
perfect competition with linear costs in ``z1`` / ``z2``), so the
feature has no working demonstration on the shipped data. For
applications, see DMQSS for the full theory and the working
``endogenous_cost_component`` examples in
``tests/test_analytical.py`` and ``tests/test_demand_adjustment.py``
for end-to-end fixtures with ``log_quantity`` as the endogenous
component.

.. note::

   Combining ``endogenous_cost_component`` with
   ``demand_adjustment=True`` is silently allowed in v0.4 but
   produces a biased demand-adjusted variance: the gradient
   computation does not currently capture the
   :math:`\partial\gamma_m / \partial\theta` channel that flows
   through the IV correction. Tracked as a v0.5 follow-up. Run with
   ``demand_adjustment=False`` when ``endogenous_cost_component`` is
   active, or apply the IV correction once externally and pass the
   corrected marginal cost.


Where to go from here
---------------------

* :doc:`api` — full attribute and method reference.
* :doc:`math` — formulas for the RV statistic, the F-stat with the
  :math:`K_{\text{eff}}` rank adjustment, the MCS elimination, the
  Villas-Boas matrix, and the DMQSS first-stage correction.
* :doc:`faq` — common errors, output footnotes (``recomputed with
  extra precision``, ``indistinguishable``), and known caveats.
* :doc:`references` — the methodology papers (DMSS, DMQSW, DMQSS).
