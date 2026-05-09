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
   per-cell DMSS critical values and the strongest reliability claim
   :math:`F` supports.
#. :ref:`Problem-level taxes <advanced-taxes>` — unit and ad-valorem
   taxes that affect every candidate model's FOC.
#. :ref:`Pass-through diagnostics <advanced-passthrough>` — the
   DMQSW (2026) framework: pre-solve per-pair feature distances
   (:meth:`~pyRVtest.Problem.passthrough_summary`), the raw
   pass-through matrix
   (:meth:`~pyRVtest.ProblemResults.passthrough_matrix`), and
   post-solve channel decomposition for one IV column
   (:meth:`~pyRVtest.Problem.instrument_channels`).
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
:meth:`~pyRVtest.ProblemResults.reliability_summary` columns.

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
instrument set). DMSS-style robustness reduces to a comparison of
``F`` against the published critical values at the cell's plug-in
:math:`\hat\rho^2`, with a worst-rho variant for users who want a
robustness margin on the :math:`\hat\rho^2` estimate.
:meth:`~pyRVtest.ProblemResults.reliability_summary` returns a
long-form ``pandas.DataFrame`` with one row per pair of candidate
models for each instrument set:

.. code-block:: python

   results = ...  # from any Problem.solve() call
   df = results.reliability_summary()
   keep = ['model_i_label', 'model_j_label',
           'F', 'rho_squared', 'size_cv_075', 'power_cv_095',
           'strongest_claim_size']
   print(df[keep].to_string(index=False))

What the columns mean:

* ``F`` — the DMSS scaled F-statistic.
* ``rho_squared`` — :math:`\hat\rho^2`, the moment-pair correlation
  whose proximity to 1 (the Cauchy-Schwarz boundary) drives numerical
  fragility.
* ``F_high_precision`` — populated when the package detects
  :math:`\hat\rho^2` close to 1 and recomputes :math:`F` with mpmath
  at extra precision (NaN otherwise; a footnote in the printed table
  flags affected pairs).
* ``strongest_claim_size`` / ``strongest_claim_power`` — the strongest
  worst-case size (or best-case power) claim the F passes, e.g.
  ``"worst-case size <= 7.5%"``.
* ``size_cv_075`` / ``size_cv_100`` / ``size_cv_125`` — worst-rho
  size CVs at the 7.5% / 10% / 12.5% levels (max CV across
  :math:`\rho^2 \in [0, 0.99]` at this :math:`K`). Clearing
  ``size_cv_075`` is the strongest worst-rho size claim.
* ``power_cv_050`` / ``power_cv_075`` / ``power_cv_095`` — worst-rho
  power CVs at the 50% / 75% / 95% levels. Clearing ``power_cv_095``
  is the strongest worst-rho power claim.
* ``size_cv_075_emp`` / ``size_cv_100_emp`` / ``size_cv_125_emp`` and
  ``power_cv_050_emp`` / ``power_cv_075_emp`` / ``power_cv_095_emp``
   — the empirical-rho counterparts (CVs at the cell's plug-in
  :math:`\hat\rho^2`). The internal verdict uses these; they take
  :math:`\hat\rho^2` noise as given (no robustness margin) and are
  typically less conservative than the worst-rho set.

The :meth:`~pyRVtest.ProblemResults.reliability_summary` DataFrame is
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

Pass-through diagnostics (DMQSW framework)
------------------------------------------

The DMQSW (2026) identification framework derives instrument relevance
for the conduct test from candidate models' implied pass-through
matrices :math:`P_m = (I - \partial \Delta_m / \partial p)^{-1}`.
Different testing-instrument types target different
*pass-through-feature distances* between candidate matrices, and the
test has zero asymptotic power for an instrument-candidate combination
whose feature distance is structurally zero (DMQSW Remarks 1, 2, 4, 5).

pyRVtest exposes three diagnostic methods that surface the framework:

* :meth:`~pyRVtest.Problem.passthrough_summary` — pre-solve γ-free
  pair-by-pair structural feature distances (four DMQSW-keyed metrics
  plus an optional per-model structural block).
* :meth:`~pyRVtest.ProblemResults.passthrough_matrix` — raw
  :math:`P_m` for one candidate in one market (inspection / debugging).
* :meth:`~pyRVtest.Problem.instrument_channels` — post-solve channel
  decomposition for one chosen IV column, separating the
  pass-through-mediated indirect channel from the markup-derivative
  direct channel.

Pass-through matrices are computed *numerically* via central-
difference perturbation through each candidate's markup function,
with an analytical fast path for :class:`~pyRVtest.Vertical`
(Villas-Boas 2007) and short-circuit identity / :math:`\varphi I`
for trivial conducts (:class:`~pyRVtest.PerfectCompetition`,
:class:`~pyRVtest.ConstantMarkup`,
:class:`~pyRVtest.UserSuppliedMarkups`,
:class:`~pyRVtest.RuleOfThumb`). The methodology footer of every
diagnostic output reflects which paths fired for the candidate set;
see :doc:`math`.

Pre-solve: ``passthrough_summary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct the :class:`~pyRVtest.Problem` (do not solve yet), then
inspect the pair-by-pair feature distances. The four metrics correspond
to DMQSW Remarks 1, 2, 4, and 5; a structural distance of zero under a
remark rules out the corresponding instrument type for the pair ex-ante.

.. code-block:: python

   import pyRVtest

   data = pyRVtest.data.load_example()
   problem = pyRVtest.Problem(
       cost_formulation=pyRVtest.Formulation('1 + z1 + z2'),
       instrument_formulation=pyRVtest.Formulation('0 + rival_z1 + rival_z2'),
       models=[
           pyRVtest.Bertrand(ownership='firm_ids'),
           pyRVtest.Cournot(ownership='firm_ids'),
           pyRVtest.Monopoly(ownership='firm_ids'),
           pyRVtest.PerfectCompetition(),
       ],
       product_data=data,
       demand_params={'estimate': 'logit',
                      'formulation_X': pyRVtest.Formulation('1 + x1'),
                      'formulation_Z': pyRVtest.Formulation('0 + z1')},
   )
   print(problem.passthrough_summary())

Output (median across 3000 markets)::

   Per-pair pass-through-feature distances (median across 3000 markets):

                             pair  offdiag_ratio  full_pass  row_sum  level_adj
              (Bertrand, Cournot)   1.267916e-01   0.173728 0.207784   0.343116
             (Bertrand, Monopoly)   2.025077e+00   0.800202 0.813289   0.661342
   (Bertrand, PerfectCompetition)   1.267916e-01   0.685104 0.378482   1.924238
              (Cournot, Monopoly)   1.799031e+00   0.755743 0.361467   0.489735
    (Cournot, PerfectCompetition)   1.335208e-09   0.837912 0.837912   2.162204
   (Monopoly, PerfectCompetition)   1.799031e+00   0.958163 1.134994   2.252656

   Per-feature notes:
     offdiag_ratio (rival cost shifters): γ-free; column ratios. Zero ⇒
       structural degeneracy. Magnitude doesn't predict power.
     full_pass (own+rival cost; product chars under linear-index demand):
       γ_known scaled or γ=0 by rival exclusion; full pass-through difference.
     row_sum (unit tax): row sums of pass-through. ν observed; fully computable.
     level_adj (ad valorem tax): ‖P_m·(p−Δ_m) − P_m'·(p−Δ_m')‖. ν, p observed.

   Methodology — pass-through: central-difference numerical, delta=1e-7
   (Bertrand / Cournot / Monopoly / PartialCollusion / MixCournotBertrand
   / CustomConductModel); exact via short-circuit (PerfectCompetition /
   ConstantMarkup / UserSuppliedMarkups / RuleOfThumb).
   See passthrough_summary() docstring and docs/math.rst.

Reading the table: the ``(Cournot, PerfectCompetition)`` row shows
``offdiag_ratio ≈ 1.3e-9``, which is numerical zero. This is the DMQSW
headline result: under logit demand, both Cournot and Perfect
Competition have diagonal pass-through matrices, so the off-diagonal
column features that rival cost shifters target are identical. *Rival
cost shifters cannot distinguish this pair, period* — even with
infinite data.

The other three features (``full_pass``, ``row_sum``, ``level_adj``)
are nonzero for the same pair, indicating that own-and-rival cost
shifters, product characteristics under linear-index demand, per-unit
taxes, or ad-valorem taxes *could* distinguish the pair (subject to
empirical sample variation, which the framework cannot predict
ex-ante).

The ``with_models=True`` form prepends a per-model structural block
showing each candidate's median diagonal, signed-max off-diagonal,
and median row sum — useful for diagnosing which model is driving a
small pair distance:

.. code-block:: python

   print(problem.passthrough_summary(with_models=True))

Cross-reading the per-model and per-pair blocks shows that Cournot's
``max_offdiag = 0.000`` and PerfectCompetition's ``max_offdiag =
0.000`` are what drives the ``(Cournot, PerfectCompetition)``
``offdiag_ratio`` to zero — they are the only two candidates with
diagonal pass-through matrices.

Post-solve: ``reliability_summary`` cross-read
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After ``problem.solve(...)``, the pass-through framework view should
be cross-read against the empirical reliability table:

.. code-block:: python

   results = problem.solve(demand_adjustment=False)
   df = results.reliability_summary()
   keep = ['model_i_label', 'model_j_label', 'F', 'rho_squared', 'strongest_claim_size']
   print(df[keep].to_string(index=False))

Output::

      model_i_label        model_j_label          F  rho_squared     strongest_claim_size
           bertrand              cournot   92.84868     0.327266  worst-case size <= 7.5%
           bertrand             monopoly  170.20543     0.495062  worst-case size <= 7.5%
           bertrand  perfect_competition    2.55167     0.947320  worst-case size <= 7.5%
            cournot             monopoly  178.43779     0.217697  worst-case size <= 7.5%
            cournot  perfect_competition    0.00605     0.975397  worst-case size <= 7.5%
           monopoly  perfect_competition    1.42474     0.988201  worst-case size <= 7.5%

For the ``(Cournot, perfect_competition)`` pair, the F-statistic is
essentially zero, confirming the structural degeneracy empirically.
The cross-read pattern lets the user diagnose which kind of weakness
is at play:

* ``offdiag_ratio = 0`` **and** ``F`` near zero → structural
  degeneracy under rival cost shifters. No amount of data will fix
  this; switch to a different instrument type.
* ``offdiag_ratio > 0`` **but** ``F`` low → empirical weakness in
  this sample. The pair is structurally distinguishable; the data
  just lack the variation to show it. Consider a richer instrument
  bundle, more markets, or a different cost shifter.
* ``offdiag_ratio = 0`` **and** ``F`` clearing CVs → suspicious;
  re-check the candidate set and the IV bundle for misspecification.

Post-solve: ``instrument_channels``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For one chosen IV column, decompose the per-pair difference in
candidates' implied causal effects into pass-through-mediated and
markup-derivative channels:

.. code-block:: python

   print(results.instrument_channels(column='rival_z2', instrument='rival_cost'))

Output::

   Post-solve instrument-channel decomposition: column 'rival_z2'
   (declared type: 'rival_cost'). γ_m fitted from solve.

   Channel components (combine per instrument type):
     indirect = P_m · (P_m^{-1} − P_m'^{-1}) · (dp_0/dz)
                       └── structural ──┘   └── data ──┘
     direct = β_m − β_m'  (empirical OLS partialling on prices)

   Data-side: empirical effect of z on prices
     ‖dp_0/dz‖_obs = 0.0023     (sample regression slope of p on rival_z2)
     SD(rival_z2)  = 0.9999
     range          = [-3.8878, 3.2002]

   Direct channel: per-candidate β_m (OLS slope of Δ_m on z | p)
                model    beta_m
             Bertrand  0.522183
              Cournot -0.066019
             Monopoly -1.997271
   PerfectCompetition  0.000000

   Per-pair channel components (median across 3000 markets):

                             pair  structural   direct
              (Bertrand, Cournot)    0.764726 0.588201
             (Bertrand, Monopoly)    4.054319 2.519453
   (Bertrand, PerfectCompetition)    2.014690 0.522183
              (Cournot, Monopoly)    3.597123 1.931252
    (Cournot, PerfectCompetition)    3.597123 0.066019
   (Monopoly, PerfectCompetition)    5.087100 1.997271

The output reports γ-free *building blocks* rather than collapsing to
a single per-pair number, because combining them depends on the
instrument's targeting:

* For a **rival cost shifter** :math:`z = w_\ell`, the relevant
  projection is the column-:math:`\ell` slice of :math:`P_m -
  P_{m'}`. The full structural-side magnitude
  :math:`\| P_m^{-1} - P_{m'}^{-1} \|_F` shown in the table is
  γ-free but includes diagonal and other-column components that
  are not selected by rival-:math:`\ell` IV variation; the *off-
  diagonal column* projection is what
  :meth:`~pyRVtest.Problem.passthrough_summary`'s ``offdiag_ratio``
  captures (and which is zero for ``(Cournot, PerfectCompetition)``
  in this example, despite the Frobenius norm being nonzero).
* For a **per-unit tax** instrument, multiply by ``np.ones(J_t)`` to
  pick out row sums.
* For an **ad-valorem tax** instrument, the projection involves
  :math:`P_m (p - \Delta_m)`.
* For a **product-characteristic** instrument that enters demand,
  the direct channel is structurally nonzero and ``β_m`` identifies
  the markup response; the per-pair ``direct`` column is the
  relevant magnitude.

For composite IVs (BLP-style sums, differentiation IVs), the same
conditional regression identifies ``β_m`` uniformly without special
handling, and the data-side and structural-side blocks compute
identically.

Pre-test workflow recap
~~~~~~~~~~~~~~~~~~~~~~~

The diagnostic suite implements a structural pre-test:

1. Build the :class:`~pyRVtest.Problem` with the candidate set you
   want to test and the IV bundle you have available.
2. Run :meth:`~pyRVtest.Problem.passthrough_summary` *before solving*.
   Note any pair with a near-zero feature distance under the
   instrument types you are using.
3. If a degenerate pair shows up, decide whether to (a) switch to a
   different instrument type that the table flags as nonzero, (b)
   accept the degeneracy and report it (e.g. as a robustness comment),
   or (c) drop the offending candidate from the menu.
4. Solve the problem, read
   :meth:`~pyRVtest.ProblemResults.reliability_summary` together with
   the framework view to diagnose structural vs. empirical weakness.
5. For follow-up inspection of one IV column,
   :meth:`~pyRVtest.Problem.instrument_channels` decomposes the
   per-pair causal effect into channel components.

The framework view is *γ-free* — it depends on candidate models and
demand fit but not on the cost-side coefficients :math:`\gamma_m`.
Use it ex-ante for instrument selection without burning post-selection
inference. The empirical reliability view in
:meth:`~pyRVtest.ProblemResults.reliability_summary` is plug-in-
:math:`\hat\rho^2` at the fitted :math:`\gamma_m`; treat it as the
honest finite-sample reliability claim.


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
   ``demand_adjustment=True`` is fully supported. The unified
   :func:`pyRVtest.solve.demand_adjustment.compute_demand_adjustment`
   pipeline computes the
   :math:`\partial\gamma_m / \partial\theta` channel (via finite
   differences over the IV correction evaluated at perturbed demand
   parameters) and folds it into the demand-adjusted variance.
   ``demand_results`` and ``demand_params`` paths produce the same
   TRV / F-stat correction up to analytical-vs-finite-difference
   noise; the
   ``test_option_a_demand_params_matches_demand_results_with_endogenous_cost``
   test in ``tests/test_demand_adjustment.py`` pins this cross-path
   parity.


Where to go from here
---------------------

* :doc:`api` — full attribute and method reference.
* :doc:`math` — formulas for the RV statistic, the F-stat with the
  :math:`K_{\text{eff}}` rank adjustment, the MCS elimination, the
  Villas-Boas matrix, and the DMQSS first-stage correction.
* :doc:`faq` — common errors, output footnotes (``recomputed with
  extra precision``, ``indistinguishable``), and known caveats.
* :doc:`references` — the methodology papers (DMSS, DMQSW, DMQSS).
