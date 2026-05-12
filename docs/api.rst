.. currentmodule:: pyRVtest


API Documentation
=================

This page is the reference index of every public class and function. The
listing follows the order in which a typical run touches them: data
preparation → demand estimation → ``Problem`` construction → ``solve()``
→ inspecting ``ProblemResults``. For a narrative walkthrough, start with
:doc:`introduction` and :doc:`tutorial`. For an end-to-end example,
see the README quick-start.

Top-level imports come from the ``pyRVtest`` namespace; the canonical
list is :py:data:`pyRVtest.__all__`. Subpackages
(``pyRVtest.backends``, ``pyRVtest.estimators``,
``pyRVtest.instruments``, ``pyRVtest.models``, ``pyRVtest.solve``)
host the implementations.


Formulation
-----------

R-style formula classes used throughout the package to specify cost
shifters, instrument sets, and demand-side regressors.

.. autosummary::
   :toctree: _api
   :template: class_without_methods.rst

   Formulation

.. note::

   :class:`ModelFormulation` is the legacy v0.3 string-based conduct
   specifier. It is **deprecated** in v0.4 and scheduled for removal in
   v0.6; emits ``DeprecationWarning`` on construction. New code should
   use the class-based :class:`ConductModel` API listed below
   (:class:`Bertrand`, :class:`Cournot`, etc.). See
   :doc:`migrating_to_v0.4` for one-to-one rewrites.

.. autosummary::
   :toctree: _api
   :template: class_without_methods.rst

   ModelFormulation


Demand estimation
-----------------

Two in-package estimators handle the linear-2SLS cases (plain logit and
one-level nested logit) and return a populated ``demand_params`` dict
that ``Problem`` consumes directly. See :doc:`in_package_demand` for a
walkthrough.

.. autosummary::
   :toctree: _api
   :template: class_with_signature.rst

   LogitEstimator
   NestedLogitEstimator

For random-coefficients / BLP-style demand, estimate externally with
``pyblp`` and pass the resulting ``ProblemResults`` to
:class:`Problem` via ``demand_results=...``. For everything else (a
demand system the package does not natively estimate), use the
:doc:`custom_demand` protocol.


Demand backends
---------------

Internal machinery that adapts a demand specification (PyBLP results, a
``demand_params`` dict, or a user-supplied protocol implementation) to
the share-Jacobian / share-Hessian interface that
:class:`~pyRVtest.Problem` consumes. Most users do not construct these
directly; ``Problem`` builds the right backend from
``demand_results=`` / ``demand_params=`` automatically. The classes are
documented for users implementing :doc:`custom_demand` or extending the
package.

.. autosummary::
   :toctree: _api

   backends.DemandBackend
   backends.SupportsDemandAdjustment
   backends.PyBLPBackend
   backends.LogitBackend
   backends.NestedLogitBackend
   backends.UserSuppliedBackend

Labor-side (experimental in v0.4):

.. autosummary::
   :toctree: _api

   backends.LaborSupplyBackend

Calling math methods on :class:`~pyRVtest.backends.LaborSupplyBackend`
raises ``NotImplementedError`` in v0.4; the full Jacobian / Hessian /
demand-adjustment-participation math is deferred to v0.5. See
:doc:`faq` for the experimental-status caveats.


Conduct models
--------------

The class-based ``ConductModel`` API. Each subclass implements a
specific firm-conduct hypothesis. Pass a list of instances to
``Problem(models=...)``; ``Problem`` computes implied markups from each
candidate and tests them against the data.

.. autosummary::
   :toctree: _api
   :template: class_with_signature.rst

   ConductModel
   Bertrand
   Cournot
   Monopoly
   PerfectCompetition
   MixCournotBertrand
   PartialCollusion
   Vertical
   RuleOfThumb
   ConstantMarkup
   UserSuppliedMarkups
   CustomConductModel

For the model-by-model walkthrough — what each class expects, the
underlying FOC, and when to choose between similar options — see the
``model_library`` notebook in :doc:`tutorial`.

Labor-side conduct models (experimental in v0.4):

.. autosummary::
   :toctree: _api
   :template: class_with_signature.rst

   Monopsony
   BertrandWages
   CournotEmployment
   NashBargaining

See :doc:`faq` for the experimental-status caveats; ``NashBargaining``
in particular raises ``NotImplementedError`` and is deferred to v0.5.


Instrument helpers
------------------

Vectorized constructors for testing instruments. Live in
``pyRVtest.instruments``; thin convenience helpers that do not encode
methodology opinions about which bundles to use together.

Product-side (``pyRVtest.instruments.product``):

.. autosummary::
   :toctree: _api

   instruments.rival_sums
   instruments.differentiation_ivs
   instruments.blp_instruments

Labor-side (``pyRVtest.instruments.labor``):

.. autosummary::
   :toctree: _api

   instruments.hausman
   instruments.bartik

A ``concentration_hhi`` helper is intentionally **not** provided on the
labor side: labor-market HHI is a function of endogenous employment
shares and is not a valid instrument for wages. See the
:mod:`pyRVtest.instruments.labor` module docstring for the rationale.


Problem
-------

The central class. Takes the cost formulation, instrument
formulation(s), candidate models, product data, and demand
specification, then runs the test on ``solve()``.

.. autosummary::
   :toctree: _api
   :template: class_with_signature.rst

   Problem
   Models
   Products

.. autosummary::
   :toctree: _api

   Problem.solve


Results
-------

``Problem.solve()`` returns a :class:`ProblemResults` object holding
the test statistics, F-stats, MCS p-values, markups, marginal cost,
and (optionally) the F-stat reliability diagnostic columns.

.. autosummary::
   :nosignatures:
   :toctree: _api
   :template: class_without_signature.rst

   ProblemResults
   PanelResults

Result inspection methods:

.. autosummary::
   :toctree: _api

   ProblemResults.reliability_summary
   ProblemResults.to_dataframe
   ProblemResults.summary_df
   ProblemResults.taus_dataframe
   ProblemResults.to_latex
   ProblemResults.to_markdown
   ProblemResults.to_pickle

In addition to the printed table, :class:`ProblemResults` exposes the
F-stat reliability diagnostic as both per-cell columns and a
human-readable summary:

* Per-cell array attributes (one entry per instrument set, of shape
  ``(M, M)`` over candidate-model pairs): ``lambda_dmss``, ``verdict``,
  ``F_high_precision``, ``rho_squared_high_precision``,
  ``worst_case_cv_size``, ``worst_case_cv_power``,
  ``strongest_claim_size``, ``strongest_claim_power``.
* :meth:`~ProblemResults.reliability_summary` returns a long-form
  ``pandas.DataFrame`` with one row per model pair, surfacing the
  reliability columns alongside the standard ``F`` and ``rho_squared``.
  Worst-rho CVs (``size_cv_075`` / ``size_cv_100`` / ``size_cv_125`` and
  ``power_cv_050`` / ``power_cv_075`` / ``power_cv_095``) and
  empirical-rho CVs (``size_cv_*_emp``, ``power_cv_*_emp``) are exposed
  as scalar columns. Use this when you want to inspect or filter on the
  diagnostic programmatically. See :doc:`faq` for how to interpret the
  ``recomputed with extra precision`` and ``indistinguishable``
  footers in the printed output.


Pass-through and instrument-relevance diagnostics
-------------------------------------------------

The DMQSW (Dearing, Magnolfi, Quint, Sullivan, and Waldfogel 2024)
framework identifies when an instrument set has *structural* power
to distinguish a pair of candidate conduct models, separately from
finite-sample F-stat strength. Two pre-solve diagnostics report the
key quantities; one post-solve helper exposes the underlying matrix.
The tutorial walk-through lives in
:ref:`Pass-through diagnostics <advanced-passthrough>`;
the underlying math is in :doc:`math`.

**Pre-solve diagnostics on the** :class:`Problem`:

.. autosummary::
   :toctree: _api

   Problem.passthrough_summary
   Problem.instrument_channels

**Post-solve mirrors on** :class:`ProblemResults`:

The same methods are available on :class:`ProblemResults` for users
who want to inspect pass-through *after* ``solve`` rather than as a
pre-solve sanity check. Both signatures and outputs match the
:class:`Problem`-level versions; calling them on
:class:`ProblemResults` is purely an ergonomic alias.

.. autosummary::
   :toctree: _api

   ProblemResults.passthrough_summary
   ProblemResults.instrument_channels
   ProblemResults.passthrough_matrix

**Low-level building block:**

.. autosummary::
   :toctree: _api

   build_passthrough

``build_passthrough(problem, model_index, market_id=None)`` returns
the per-candidate pass-through matrix
:math:`P_m = (I - \partial \Delta_m / \partial p)^{-1}` for one
market (or all markets). For the trivial-closed-form conducts
(``PerfectCompetition`` / ``ConstantMarkup`` / ``UserSuppliedMarkups``
give :math:`P = I`; ``RuleOfThumb(\varphi)`` gives
:math:`P = \varphi I`) the answer is short-circuited; for ``Vertical``
the Villas-Boas closed form is used; everything else goes through a
batched central-difference numerical core
(:func:`pyRVtest.solve.passthrough.compute_passthrough_numerical`).


Convenience functions
---------------------

Helpers for ownership matrices, markups, and reading pickled results.

.. autosummary::
   :toctree: _api

   build_ownership
   build_markups
   construct_passthrough_matrix
   evaluate_first_order_conditions
   read_pickle


Diagnostics and helpers
-----------------------

.. autosummary::
   :toctree: _api

   show_agent_guide

For runtime configuration (default dtype, output verbosity, etc.) see
:py:data:`pyRVtest.options`.


Exceptions
----------

pyRVtest raises a small hierarchy of custom exceptions for common
validation and backend failures. Every custom class subclasses a
Python built-in (``ValueError`` or ``RuntimeError``) so existing
callers using ``except ValueError:`` continue to work unchanged.

Error messages follow an **expected / received / fix** structure: what
the check was looking for, what was actually there, and a concrete
next step. Internal-invariant failures are prefixed with ``pyRVtest
internal error:`` and kept terse.

.. autosummary::
   :toctree: _api
   :template: class_without_methods.rst

   PyRVTestError
   ValidationError
   InstrumentDataError
   BackendError
   DemandBackendError
   HessianUnavailableError
