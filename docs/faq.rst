FAQ and troubleshooting
=======================

Installation and environment
----------------------------

**``pip install pyRVtest`` succeeds, but ``import pyRVtest`` raises
``AttributeError: module 'numpy' has no attribute 'unicode_'``.**

Your installed ``pyblp`` predates NumPy 2.0 (which dropped
``np.unicode_``). Upgrade ``pyblp`` to a version compatible with
NumPy 2 (``pyblp >= 1.2``), or downgrade to ``numpy < 2``. The two
supported pin combinations are ``numpy < 2 + pyblp < 1.2`` and
``numpy >= 2 + pyblp >= 1.2``; CI exercises both.

**Which Python versions are supported?**

Python 3.7+ at install time. The development environment targets 3.9
(framework Python on macOS); CI runs across the supported range.


Setting up a Problem
--------------------

**Should I pass demand to ``Problem`` via ``demand_results`` or
``demand_params``?**

Use ``demand_results`` when you have a populated
``pyblp.ProblemResults`` (the typical BLP / random-coefficients case).
Use ``demand_params`` when you estimated demand with the in-package
:class:`~pyRVtest.LogitEstimator` /
:class:`~pyRVtest.NestedLogitEstimator`, when you have hand-computed
:math:`(\alpha, \beta, \rho)`, or when you want to use the inline
estimator shortcut (``demand_params={'estimate': 'logit', ...}``).
See :doc:`in_package_demand` for the inline-shortcut form.

**How do I supply more than one set of testing instruments?**

Pass ``instrument_formulation`` as a list of
:class:`~pyRVtest.Formulation` objects. The test runs once per
instrument set and the result arrays are indexed by instrument set
(e.g., ``results.TRV[0]``, ``results.TRV[1]``).

**What's the difference between ``models=`` and ``model_formulations=``?**

``models=`` is the class-based API introduced in v0.4 — pass a list of
:class:`~pyRVtest.ConductModel` instances (e.g.,
``models=[pyRVtest.Bertrand(ownership='firm_ids'), ...]``). It's the
preferred form for new code. ``model_formulations=`` is the legacy
string-based API (``ModelFormulation(model_downstream='bertrand', ...)``)
kept for one or two releases with a ``DeprecationWarning``. The two are
mutually exclusive on a single ``Problem`` call. See
:doc:`migrating_to_v0.4` for one-to-one recipes.

**Can I still use ``Keystone()``?**

No, the alias was dropped before v0.4 release. Write
``RuleOfThumb(phi=2)`` directly.

**My conduct model is unusual (not Bertrand / Cournot / monopoly etc.).
How do I add it?**

Two options. (1) If you can express the markup formula in closed form
given prices and shares, use :class:`~pyRVtest.CustomConductModel`
with a callable. (2) If you have pre-computed markup columns from
elsewhere (a structural estimation, a separate model fit, etc.), pass
them via :class:`~pyRVtest.UserSuppliedMarkups`. The model-library
notebook (:doc:`_notebooks/model_library`) shows both.


Reading the output
------------------

**The ``TRV`` table shows ``nan`` on the diagonal — is that an error?**

No, that's by design. ``TRV[m, m]`` compares model :math:`m` to
itself, which is undefined; the diagonal is filled with ``nan``.

**An off-diagonal ``TRV`` cell is ``nan``, not a number.**

The pair of models is *trivially degenerate*: they produce identical
implied moments at the candidate weighting matrix, so the RV test
statistic is undefined. The output footer flags this case with the
note ``indistinguishable: N pair(s)``.

**The F-stat is ``NaN``, negative, or near zero.**

Three common causes:

* **Degenerate pair.** The pair is trivially degenerate (same as
  above); the F-stat denominator is ill-defined. Look for the
  ``indistinguishable`` footnote.
* **Identically-zero markup.** When one model in the pair has
  identically-zero markup (notably :class:`~pyRVtest.PerfectCompetition`),
  the F-stat denominator structure degenerates and the diagnostic can
  collapse to small or zero values even when the corresponding ``TRV``
  is sharply significant. Tracked as a v0.5 follow-up; the ``TRV`` and
  MCS p-values remain reliable in this regime.
* **Numerical fragility.** When :math:`\hat\rho^2` is very close to 1
  (Cauchy-Schwarz boundary), float64 loses precision in computing the
  F-stat denominator. The package detects this case and recomputes
  with mpmath; the output footer notes ``recomputed with extra
  precision: N pair(s)`` when this happens.

**Why does Cournot survive the MCS when the truth is Perfect
Competition?**

This is the Dearing et al. (2026) degeneracy: under logit demand both
Cournot and Perfect Competition have diagonal pass-through matrices
(zero off-diagonal pass-through of rival costs), so rival cost
shifters cannot distinguish them as testing instruments. The README
quick-start walks through this empirically. To falsify Cournot
relative to PC in this setting, use a different testing instrument
(e.g., own or rival product characteristics).

**The output shows a footnote ``recomputed with extra precision``.
What does that mean?**

For at least one model pair, double-precision float64 lost enough
significant digits in the F-stat denominator that the F-vs-CV decision
could have flipped due to rounding. The package recomputed the
affected F-statistics using mpmath at higher precision and the values
shown are the high-precision ones. This is informational; the test
results are sound.


Common errors and warnings
--------------------------

**``UserWarning: estimated alpha is positive``** (from the
in-package logit estimator).

The price coefficient estimate is positive — almost always a sign of
a misspecified instrument. Check that ``formulation_Z`` includes a
plausible cost shifter, that the cost shifter actually moves prices
in your data, and that you have enough markets to identify
:math:`\alpha`.

**``UserWarning: estimated rho is outside [0, 1)``** (nested-logit
estimator).

The within-nest correlation estimate is outside the theoretically
valid range. The estimator does **not** auto-clip. Inspect the data
and the instrument set; the most common cause is a within-nest IV
that is too weak.

**``DeprecationWarning: ModelFormulation is deprecated``.**

Migrate to the class-based API. ``ModelFormulation`` will be removed
in v0.6. See :doc:`migrating_to_v0.4` for per-case rewrites.

**``UserWarning`` when passing ``costs_type='log'`` together with
``demand_adjustment=True``.**

The two options interact incompletely in v0.4: the package falls back
to ``costs_type='linear'`` and emits a warning. If you genuinely need
log costs with a first-stage demand-adjustment correction, the
combination is tracked as a v0.5 follow-up.

**No warning, but I set ``endogenous_cost_component='shares'`` and
``demand_adjustment=True``.**

This combination is silently allowed in v0.4 but produces a biased
demand-adjusted variance: the gradient computation misses the
``dgamma/dtheta`` channel that flows through the IV correction.
Tracked as a v0.5 follow-up. As a workaround, set
``demand_adjustment=False`` when ``endogenous_cost_component`` is
active, or run the IV correction once externally and pass the
corrected marginal cost.


Performance
-----------

**``Problem.solve`` takes a long time.**

The slowest steps are typically:

* Demand estimation (if you call ``pyblp`` first or use the inline
  shortcut on a large dataset).
* The first-stage demand-adjustment finite differences (when
  ``demand_adjustment=True``); cost scales roughly as
  ``n_theta * 2 * n_instrument_sets``.
* The Model Confidence Set elimination loop (cost grows with the
  square of the number of candidate models).

Set ``demand_adjustment=False`` if you don't need the first-stage SE
correction. Drop unnecessary candidate models to shrink the MCS loop.


Labor-side conduct testing (experimental in v0.4)
-------------------------------------------------

**Why is my labor-side run raising ``NotImplementedError``?**

The labor-side feature is partial in v0.4. :class:`~pyRVtest.Monopsony`,
:class:`~pyRVtest.BertrandWages`, and
:class:`~pyRVtest.CournotEmployment` work for the basic moment
computation, but :class:`~pyRVtest.NashBargaining` is a placeholder
and the full :class:`~pyRVtest.backends.LaborSupplyBackend` math
(Jacobian, Hessian, demand-adjustment participation) is deferred to
v0.5.

**Sign conventions for labor-side models.**

Sign conventions flip relative to product-side models (the labor
supply Jacobian is upward-sloping). The labor implementation in
``pyRVtest/models/labor.py`` is flagged for a sign check against the
labor-market-conduct manuscript; treat the labor results as
provisional until that review lands.


Migration from v0.3
-------------------

For migration questions not covered above, see :doc:`migrating_to_v0.4`.
The most common changes:

* ``ModelFormulation`` → class-based ``ConductModel`` API.
* ``sigma`` keyword → ``rho`` (nested-logit parameter).
* Per-model ``unit_tax`` / ``advalorem_tax`` → Problem-level kwargs.
* ``Problem.solve(mc_correction=...)`` → ``endogenous_cost_component``
  argument at ``Problem`` construction.

All four legacy forms continue to work for one or two releases with a
``DeprecationWarning``.


Where else to look
------------------

* :doc:`introduction` — what the package does and the underlying procedure.
* :doc:`tutorial` — worked examples.
* :doc:`api` — full API reference.
* :doc:`references` — bibliography.
* ``AGENTS.md`` and ``pyRVtest.show_agent_guide()`` — architectural
  walkthrough for AI assistants and contributors.
