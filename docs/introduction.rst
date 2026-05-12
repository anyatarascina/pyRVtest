Introduction
============

.. note::

   v0.4 is the current major release. The legacy v0.3 API continues to
   work with ``DeprecationWarning`` for one or two more releases — see
   :doc:`migrating_to_v0.4`. Please use the `GitHub issue tracker
   <https://github.com/anyatarascina/pyRVtest/issues>`_ to report bugs
   or to request features.


What pyRVtest does
__________________

``pyRVtest`` is a Python package for econometric testing of firm
conduct models in differentiated-products markets. Given demand
estimates, a menu of two or more candidate supply-side conduct
hypotheses (Bertrand-Nash, Cournot, joint-profit maximization, perfect
competition, vertical relationships, rule-of-thumb pricing, ...), and
a set of testing instruments, it reports pairwise Rivers-Vuong tests
of relative fit, the DMSS effective F-statistic for instrument
strength, and Hansen-Lunde-Nason Model Confidence Set p-values for the
surviving set of candidates. Per-unit and ad-valorem taxes are
supported, marginal cost can include an endogenous component (e.g.,
quantity, under scale economies), and corrections for
demand-estimation error and clustering are available.

The package implements the procedure developed in
:ref:`references: Duarte, Magnolfi, Sølvsten, and Sullivan (2024)`
(hereafter DMSS), with later extensions in
:ref:`references: Dearing, Magnolfi, Quint, Sullivan, and Waldfogel (2024)`
(hereafter DMQSW), which characterizes when an instrument set has
power to distinguish given candidate models, and
:ref:`references: Duarte, Magnolfi, Quint, Sølvsten, and Sullivan (2026)`
(hereafter DMQSS), which extends the procedure to settings where
marginal cost includes an endogenous component (e.g., quantity, under
scale economies).


The procedure
_____________

For each pair of candidate conduct models :math:`m, m'`, ``pyRVtest``
computes:

* **The Rivers-Vuong (RV) test statistic** :math:`T_{RV}`. Following
  DMSS, this compares each model's GMM moment value at a fixed
  instrument set, with an asymptotic-normal distribution under the
  null of equal fit. Two-sided thresholds:
  :math:`|T_{RV}| > 1.645,\ 1.96,\ 2.576` for 10%, 5%, 1% significance.

* **The DMSS scaled F-statistic.** A diagnostic for whether the RV
  test itself has acceptable size and power given the instrument set
  and the candidate models being compared. Critical values are
  reported for worst-case size of the test (0.075, 0.10, 0.125) and
  best-case power (0.50, 0.75, 0.95).

* **Model Confidence Set (MCS) p-values.** Following Hansen, Lunde, and
  Nason (2011), the MCS reports the largest set of candidate models
  that cannot be statistically distinguished as jointly best-fitting at
  a given level. Each candidate's MCS p-value is the largest level at
  which it survives.

When demand parameters are estimated rather than fixed, the moments and
their variances are corrected for first-stage uncertainty (DMSS Appendix
C; DMQSS Appendix B for the non-linear-cost case). Clustering
corrections are also available.


Inputs
______

To run a test, the user supplies:

* **Product-level data.** Per-market, per-product observations of
  prices, market shares, ownership identifiers, exogenous demand-side
  characteristics, and cost shifters.
* **Demand estimates.** Either an existing ``pyblp.ProblemResults``
  object, or estimates produced by the in-package logit / nested-logit
  estimators (:class:`~pyRVtest.LogitEstimator`,
  :class:`~pyRVtest.NestedLogitEstimator`).
* **Candidate conduct models.** Specified via the class-based
  ``ConductModel`` API: :class:`~pyRVtest.Bertrand`,
  :class:`~pyRVtest.Cournot`, :class:`~pyRVtest.Monopoly`,
  :class:`~pyRVtest.PerfectCompetition`,
  :class:`~pyRVtest.MixCournotBertrand`,
  :class:`~pyRVtest.PartialCollusion`, :class:`~pyRVtest.Vertical`,
  :class:`~pyRVtest.RuleOfThumb`, :class:`~pyRVtest.ConstantMarkup`,
  :class:`~pyRVtest.UserSuppliedMarkups`,
  :class:`~pyRVtest.CustomConductModel`.
* **Cost shifters.** Variables that explain marginal cost, specified
  via the cost formulation. To handle a marginal cost that depends on
  one or more endogenous components (e.g., own quantity under scale
  economies, own + same-firm-platform output under scale-and-scope, or
  ``q + q²`` under quadratic cost), pass the column name(s) to
  ``endogenous_cost_component=`` — single string for one variable or a
  list of strings for the multi-variable case (DMQSS Appendix A.4).
* **Testing instruments.** Variables that distinguish candidate models
  by their differential implications for prices or markups. DMQSW
  characterizes which instruments have power against which model
  comparisons.


Workflow
________

A typical ``pyRVtest`` session has four stages:

1. **Estimate demand.** Externally with ``pyblp``, or in-package with
   :class:`~pyRVtest.LogitEstimator` or
   :class:`~pyRVtest.NestedLogitEstimator`.
2. **Build the problem.** Pass the demand results, product data, cost
   formulation, instrument formulation(s), and the list of candidate
   conduct models to :class:`~pyRVtest.Problem`.
3. **Solve.** Call ``Problem.solve()``, optionally enabling
   demand-adjustment and clustering corrections. Returns a
   :class:`~pyRVtest.ProblemResults` object.
4. **Inspect.** ``print(results)`` shows the formatted RV / F / MCS
   table per instrument set. The ``results`` object exposes
   ``markups``, ``marginal_cost``, and the v0.4 DMSS + DMQSW + DMQSS
   diagnostic suite:

   * :meth:`~pyRVtest.ProblemResults.reliability_summary` — per-cell
     F-stat with worst-rho and empirical-rho DMSS critical values.
   * :meth:`~pyRVtest.ProblemResults.passthrough_matrix` — raw
     :math:`P_m = (I - \partial \Delta_m / \partial p)^{-1}` for one
     candidate / market.
   * :meth:`~pyRVtest.Problem.passthrough_summary` — pre- or post-solve
     γ-free pair-by-pair structural feature distances (DMQSW
     instrument-relevance framework).
   * :meth:`~pyRVtest.Problem.instrument_channels` — post-solve
     per-pair channel decomposition for one IV column. Under
     non-constant marginal cost
     (``endogenous_cost_component`` set), uses DMQSS Appendix B z^e
     residualization automatically.
   * :meth:`~pyRVtest.ProblemResults.to_dataframe` — long-form export.

See :doc:`tutorial` for a step-by-step walkthrough of each stage on the
synthetic example dataset shipped with the package, with the actual
code and verbatim output for every step.


Where to go next
________________

* :doc:`installation` — pip and dependency setup.
* :doc:`tutorial` — worked examples on the Nevo cereal data and the
  conduct-model library, including a runnable end-to-end quick start
  on the synthetic dataset shipped with the package.
* :doc:`in_package_demand` — built-in logit / nested-logit estimation.
* :doc:`advanced_features` — multi-IV, demand-adjustment, clustering,
  Problem-level taxes, pass-through, endogenous cost.
* :doc:`migrating_to_v0.4` — migration guide for users coming from
  v0.3.
* :doc:`custom_demand` — protocol for custom demand backends.
* :doc:`faq` — common questions.
* :doc:`api` — reference API documentation.
* :doc:`references` — bibliography.
