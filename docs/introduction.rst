Introduction
============

.. note::

   This package is in beta. In future versions, the API may change substantially. Please use the `GitHub issue tracker <https://github.com/anyatarascina/pyRVtest/issues>`_ to report bugs or to request features.


What pyRVtest does
__________________

``pyRVtest`` is a Python package for econometric testing of firm conduct
models in differentiated-products markets. Given demand estimates and a
candidate set of supply-side conduct hypotheses (Bertrand-Nash,
Cournot, joint-profit maximization, perfect competition, vertical
relationships, rule-of-thumb pricing, ...), it computes whether the
data falsify each candidate using moment-based methods that account for
finite-sample uncertainty in instrument relevance.

The package implements the procedure developed in
:ref:`references: Duarte, Magnolfi, Sølvsten, and Sullivan (2023)`
(hereafter DMSS), with later extensions in
:ref:`references: Dearing, Magnolfi, Quint, Sullivan, and Waldfogel (2026)`
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
  :math:`|T_{RV}| > 1.64,\ 1.96,\ 2.58` for 10%, 5%, 1% significance.

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
  :class:`~pyRVtest.CustomConductModel`.
* **Cost shifters.** Variables that explain marginal cost. May be
  exogenous (e.g., input prices) or include an endogenous component
  (e.g., quantity, under scale economies; see DMQSS).
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
   table per instrument set. The ``results`` object also exposes
   ``markups``, ``marginal_cost``,
   :meth:`~pyRVtest.ProblemResults.passthrough_matrix`,
   :meth:`~pyRVtest.ProblemResults.passthrough_comparison` (DMQSW
   Remark 4), and other diagnostics.


Where to go next
________________

* :doc:`tutorial` — worked examples on the Nevo cereal data and the
  conduct-model library.
* :doc:`migrating_to_v0.4` — migration guide for users coming from
  v0.3.
* :doc:`custom_demand` — protocol for custom demand backends.
* :doc:`api` — reference API documentation.
* :doc:`references` — bibliography.

For a minimal end-to-end run on a synthetic dataset shipped with the
package, see **Quick start** below.


.. include:: ../README.rst
    :start-after: docs-start
