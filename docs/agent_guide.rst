Agent guide
===========

A tutorial walkthrough for AI coding assistants (Claude, GPT, and similar
tools) and for new human contributors who prefer a narrative over a reference.
If you just want a quick map, read ``AGENTS.md`` at the repo root; this file
goes deeper on the math, the backend protocol, and the deprecation plan.

This page exists because pyRVtest's v0.4 refactor explicitly treats "AI usage
as a first-class design target." The class-based API, the module separation
by concern, and the single-source-of-truth helpers were chosen partly so that
an AI assistant can understand and safely modify the package without
reverse-engineering 25 steps of commit history.

If you are reading this via ``pyRVtest.show_agent_guide()``, you are seeing
the contents of ``AGENTS.md``; this longer ``docs/agent_guide.rst`` is only
available from the rendered documentation or the repository.

What the package does
---------------------

Given a demand estimate (market shares, own- and cross-price derivatives)
and a set of candidate *conduct* hypotheses about how firms set prices
(Bertrand-Nash, Cournot, monopoly, perfect competition, vertical
integration, partial collusion, custom user-supplied markup formulas),
pyRVtest runs the Rivers and Vuong (2002) test and related diagnostics to
statistically compare the models. The output is a table of pairwise RV
test statistics, scaled F-statistics for weak-instrument diagnostics,
and Model Confidence Set p-values in the spirit of
Hansen, Lunde, and Nason (2011). Users then read off which conduct models are
or are not rejected at their chosen significance level.

The key reference for the econometrics is Duarte, Magnolfi, Sølvsten,
and Sullivan (2023) ("Testing Firm Conduct"), Appendix C. pyRVtest implements their
first-stage-correction eq. 77 for demand-side parameter estimation
uncertainty, plus a clustered variance option for market-level
clustering.

Package architecture
--------------------

The v0.4-refactor branch reorganizes the package around a
backend-generic demand interface and a class-based conduct model API.
The directory layout is::

    pyRVtest/
    ├── __init__.py          # Public API re-exports, __all__
    ├── _agent_guide.py      # show_agent_guide() helper (step 23)
    ├── problem.py           # Problem, Models recarray
    ├── products.py          # Products data class (step 2)
    ├── formulation.py       # Formulation (pyblp re-export),
    │                        #   ModelFormulation (deprecated)
    ├── markups.py           # build_markups, build_ownership, etc.
    ├── output.py            # Display helpers
    ├── options.py           # Global package options
    ├── data/                # Critical value CSV tables
    ├── backends/            # Demand-side interface implementations
    │   ├── base.py          # DemandBackend + SupportsDemandAdjustment
    │   ├── pyblp.py         # PyBLPBackend
    │   ├── logit.py         # LogitBackend + analytical helpers
    │   ├── nested_logit.py  # NestedLogitBackend
    │   ├── user.py          # UserSuppliedBackend
    │   └── labor/           # Labor-side backends (skeleton)
    ├── models/              # ConductModel class hierarchy
    │   ├── base.py          # ConductModel abstract base
    │   ├── standard.py      # Bertrand, Cournot, Monopoly, PerfectCompetition
    │   ├── mixed.py         # MixCournotBertrand
    │   ├── collusion.py     # PartialCollusion
    │   ├── custom.py        # CustomConductModel
    │   ├── vertical.py      # Vertical composer
    │   ├── constant.py      # Placeholder (step 12)
    │   ├── labor.py         # Placeholder (step 14)
    │   └── _adapter.py      # Legacy ModelFormulation bridge
    ├── solve/               # Per-phase pipeline stages
    │   ├── demand_adjustment.py  # DMSS 2024 eq. 77 (step 4d)
    │   ├── passthrough.py        # build_passthrough (step 11)
    │   ├── markups.py            # Markup stage (scaffolded, step 8)
    │   ├── orthogonalize.py
    │   ├── endogenous_cost.py
    │   └── test_engine.py
    ├── instruments/         # Instrument-construction helpers
    │   ├── product.py       # BLP / differentiation / rival-sum
    │   └── labor.py         # Bartik / Hausman / HHI
    └── results/
        └── __init__.py      # ProblemResults, Progress

The top-level ``pyRVtest`` namespace re-exports the stable public API
from these submodules. The canonical contents are in
``pyRVtest.__all__`` and are audited by
:mod:`tests.test_public_api_pin` and :mod:`tests.test_import_roundtrip`.

The DMSS framework in three paragraphs
--------------------------------------

Duarte, Magnolfi, Sølvsten, and Sullivan (2023) propose a GMM moment-based framework for testing
conduct. For each candidate model :math:`m` you compute implied marginal
costs :math:`\hat c_m = p - \mu_m(\hat\theta)` where :math:`\mu_m` is the
model's implied markup and :math:`\hat\theta` are the estimated demand
parameters. You then form the moment :math:`g_m = Z' (\hat c_m - W \hat
\gamma_m)` where :math:`W` are cost shifters, :math:`Z` are excluded
instruments, and :math:`\hat\gamma_m` is a model-specific cost-function
coefficient vector estimated by 2SLS. The GMM fit measure is :math:`Q_m
= g_m' \Omega_m^{-1} g_m / N` where :math:`\Omega_m` is a clustered
variance estimator.

The pairwise Rivers-Vuong statistic for models :math:`m_1` and
:math:`m_2` is :math:`T_{RV} = \sqrt{N} (Q_{m_1} - Q_{m_2}) / \hat\sigma`,
where :math:`\hat\sigma` comes from the variance of the moment
difference. The first-stage correction (DMSS Appendix C eq. 77) inflates
this variance to account for the fact that :math:`\hat\theta` was
estimated and the markups :math:`\mu_m(\hat\theta)` therefore carry
parameter uncertainty. The scaled F-statistic reported alongside
:math:`T_{RV}` is a weak-instrument diagnostic derived from the same
moment conditions.

The Model Confidence Set (MCS) procedure of Hansen, Lunde, and Nason (2011)
takes the matrix of pairwise :math:`T_{RV}` values and returns
per-model p-values indicating whether each model survives elimination
at the chosen confidence level. A small MCS p-value on model :math:`m`
means "we can reject the hypothesis that :math:`m` is among the best
models." This is the final headline output of ``Problem.solve``.

The class-based ``ConductModel`` API
------------------------------------

v0.4 replaces the v0.3 string-based ``ModelFormulation`` specifier with
a class hierarchy rooted at ``ConductModel``. Each concrete class
implements two math hooks:

* ``_compute_markup(ownership_matrix, response_matrix, shares)`` returns
  the markup vector implied by the model for one market. For Bertrand
  this is :math:`-(O \odot D)^{-1} s`, where :math:`O` is the ownership
  matrix, :math:`D = \partial s / \partial p` is the response matrix,
  and :math:`s` is the share vector.

* ``_markup_derivative(O, D, dD, s, mu)`` returns the gradient of the
  markup with respect to one demand parameter :math:`\theta_k`, given
  :math:`dD = \partial D / \partial \theta_k`. This closes the
  first-stage-correction chain rule in
  ``solve/demand_adjustment.py``.

Concrete classes:

* :class:`pyRVtest.Bertrand` — price-setting with ownership matrix.
* :class:`pyRVtest.Cournot` — quantity-setting with ownership matrix.
* :class:`pyRVtest.Monopoly` — full collusion; ownership matrix has all
  ones on the block for the monopoly group.
* :class:`pyRVtest.PerfectCompetition` — structurally-zero markup. Useful
  also as a sentinel when supplying ``user_supplied_markups``.
* :class:`pyRVtest.MixCournotBertrand` — mixed market with a per-product
  Cournot/Bertrand indicator column (``mix_flag``).
* :class:`pyRVtest.PartialCollusion` — Bertrand with a kappa-modified
  ownership matrix. Signals collusion at the call site by class name.
* :class:`pyRVtest.CustomConductModel` — wraps a user-supplied markup
  function. Bypasses the internal math entirely.
* :class:`pyRVtest.Vertical` — composer that combines a downstream and
  upstream :class:`ConductModel` into a bilateral oligopoly. Carries
  the shared config (``vertical_integration``, taxes) at the wrapper
  level; inner conducts carry only their own ownership / kappa / mix.

A typical v0.4 problem setup::

    import pyRVtest

    problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
        product_data=product_data,
        demand_results=pyblp_results,
        models=[
            pyRVtest.Bertrand(ownership='firm_ids'),
            pyRVtest.Cournot(ownership='firm_ids'),
            pyRVtest.Monopoly(ownership='firm_ids'),
            pyRVtest.PerfectCompetition(),
            pyRVtest.Vertical(
                downstream=pyRVtest.Bertrand(ownership='firm_ids'),
                upstream=pyRVtest.Monopoly(ownership='manufacturer_ids'),
                vertical_integration='vi_col',
            ),
        ],
    )
    results = problem.solve()
    print(results)

The ``DemandBackend`` protocol
------------------------------

The demand side is factored out behind two PEP 544 structural Protocols
declared in ``pyRVtest/backends/base.py``:

1. :class:`~pyRVtest.backends.base.DemandBackend` — the core interface.
   Every backend implements this. Provides:

   * ``n_parameters`` (int) and ``theta_names`` (list of str) for
     parameter enumeration.
   * ``compute_jacobian(market_id=None)`` returning :math:`\partial s /
     \partial p` for one market or (stacked, NaN-padded) across all
     markets.
   * ``compute_hessian(market_id=None)`` returning :math:`\partial^2 s /
     \partial p^2` for one market (required only for vertical
     passthrough).
   * ``perturbed(theta_index, delta)`` — context manager yielding a
     new backend with parameter :math:`\theta_k` shifted by ``delta``.
     Used by the finite-difference demand adjustment path; analytical
     backends may override with closed-form derivatives.

2. :class:`~pyRVtest.backends.base.SupportsDemandAdjustment` — optional
   mixin for backends that can participate in the DMSS eq. 77
   first-stage correction. Requires:

   * ``demand_moments()`` returning :math:`g_D = Z_D' \xi` where
     :math:`Z_D` are demand-side excluded instruments and :math:`\xi`
     is the unobserved demand residual.
   * ``xi_gradient(theta_index)`` returning :math:`\partial\xi /
     \partial\theta_k`.
   * ``jacobian_gradient(theta_index)`` returning :math:`\partial D /
     \partial\theta_k`.

The testing engine uses ``isinstance(backend, SupportsDemandAdjustment)``
to decide whether to invoke the first-stage correction at all. The split
(a v0.4 §2 design goal) lets :class:`~pyRVtest.backends.UserSuppliedBackend`
implement just the core protocol — users who supply a bare Jacobian
don't have to also supply :math:`\xi`, :math:`Z_D`, :math:`W_D`, and
the BLP contraction.

Concrete backends:

* :class:`~pyRVtest.backends.PyBLPBackend` — wraps
  ``pyblp.ProblemResults``. Only place in the package that touches
  ``pyblp.ProblemResults._sigma / _pi / _beta / _rho / _delta``;
  all other access sites go through the public protocol.

* :class:`~pyRVtest.backends.LogitBackend` — analytical logit. Closed-form
  Jacobian and Hessian from the logit share formula; no finite-differencing.

* :class:`~pyRVtest.backends.NestedLogitBackend` — analytical nested
  logit following the AFSSZ L-level convention. Closed-form Hessian
  added in v0.4 (replacing the finite-difference path that v0.3 used
  for vertical + demand_params).

* :class:`~pyRVtest.backends.UserSuppliedBackend` — bring-your-own-Jacobian.
  Requires at minimum a stacked :math:`(N, J_{\max})` NaN-padded
  Jacobian and per-product market ids. Optional: a per-market Hessian
  callable (required for vertical models), parameter names, and a
  perturb callback (required for finite-difference demand adjustment).
  See :doc:`custom_demand` for a worked example.

Pipeline stages in ``solve/``
-----------------------------

``Problem.solve()`` runs a sequence of pipeline stages. Each stage has
an explicit input/output contract and lives in its own module under
``pyRVtest/solve/``:

* ``solve/demand_adjustment.py`` — the DMSS 2024 Appendix C eq. 77
  first-stage correction. v0.4 step 4d unified the two previously
  parallel paths (the ``demand_results`` / PyBLP path and the
  ``demand_params`` / analytical path) into a single implementation
  generic over any ``SupportsDemandAdjustment`` backend. Exports
  :func:`~pyRVtest.solve.demand_adjustment.compute_demand_adjustment`
  and the shared 2SLS profile-out helper ``_residualize_on_xd``.

* ``solve/passthrough.py`` — exports
  :func:`~pyRVtest.solve.passthrough.build_passthrough`, the
  Villas-Boas (2007) passthrough matrix construction, surfaced as a
  standalone diagnostic so users can compute it without running
  ``Problem.solve``.

* ``solve/markups.py`` — per-model markup stage. Scaffolded in v0.4
  step 1; the actual extraction from ``Problem.solve`` lands in step 8.

* ``solve/orthogonalize.py`` — cost-shifter orthogonalization stage.

* ``solve/endogenous_cost.py`` — endogenous-cost :math:`\gamma`
  correction.

* ``solve/test_engine.py`` — final RV / F / MCS computation stage.

Instrument helpers in ``instruments/``
--------------------------------------

Constructing testing instruments :math:`Z` is often the most
project-specific step in a conduct-testing exercise. pyRVtest ships
with common constructions:

* ``pyRVtest.instruments.product`` — BLP-style instruments,
  differentiation IVs (Gandhi and Houde, 2019), rival-sum instruments, and
  Hausman-style cost-shifter constructions for the product side.

* ``pyRVtest.instruments.labor`` — Bartik instruments,
  concentration-based HHI instruments, and Hausman-style wage
  instruments for labor-side applications.

These landed in v0.4 step 13 (single commit). Use them as a starting
point for your own Z matrix; glue them into a
:class:`pyRVtest.Formulation` via ordinary numpy/pandas manipulation.

Deprecation plan (detailed)
---------------------------

Three backward-compatibility surfaces coexist on the v0.4 branch:

1. **``ModelFormulation``.** Deprecated; use the class-based API
   (:class:`Bertrand`, :class:`Cournot`, ...). Implementation: v0.4
   step 5c adds a translation adapter in ``pyRVtest/models/_adapter.py``
   so :class:`ModelFormulation` instances construct the corresponding
   :class:`ConductModel` internally and the rest of the pipeline sees
   only the class-based side. The class-level deprecation flag lives
   on :class:`ModelFormulation` itself; the warning fires once per
   Python session.

2. **``Problem(model_formulations=...)``.** The legacy kwarg accepts a
   sequence of :class:`ModelFormulation` objects. The new ``models=``
   kwarg takes a list of :class:`ConductModel` instances. Passing both
   raises ``TypeError``. Passing only the old kwarg emits a
   deprecation warning and routes through the adapter.

3. **``demand_params['sigma']``.** Alias for ``demand_params['rho']``.
   Step 6b. Passing both in the same dict raises ``TypeError``.
   The alias is user-facing only; the
   :class:`~pyRVtest.backends.NestedLogitBackend` class constructor
   keeps ``sigma=[...]`` because its internal math follows the AFSSZ
   L-level convention where ``sigma_l`` is a per-level parameter.

**Warning-emission hygiene.** Each deprecation site uses a
class-level-or-module-level flag to fire the ``DeprecationWarning``
once per Python session. We deliberately do not call
``warnings.simplefilter('once', DeprecationWarning)`` at import time
because that would mutate the user's global filter state and
suppress unrelated deprecations from other libraries. If you are
adding a new deprecation, follow the same flag pattern — see
:mod:`pyRVtest.formulation` for an example implementation.

**Timeline.**

* **v0.4 (current):** class-based API lands; legacy surfaces continue
  to work with once-per-session warnings.
* **v0.5:** same as v0.4; continued migration window.
* **v0.6:** legacy surfaces removed. Only the class-based API works.

Testing invariants
------------------

The test suite encodes a handful of invariants that must not be violated
without explicit coordination:

1. **Snapshot reproducibility.** ``tests/snapshots/*.json`` contains
   per-fixture JSON with exact TRV / F / MCS / markups / g / Q.
   ``tests/test_snapshots.py`` asserts every current run reproduces
   these. The decision rule:

   * Changes at ``atol <= 1e-12``: floating-point noise, regenerate freely.
   * Changes above ``atol <= 1e-7``: require numerical justification,
     coauthor approval, and a handover note.
   * Grey zone in between: investigate before regenerating.

   Regenerate with
   ``REGENERATE_SNAPSHOTS=1 python -m pytest tests/test_snapshots.py``.

2. **First-stage correction equivalence.**
   ``tests/test_first_stage_correction.py`` directly compares the
   ``demand_results`` path with the ``demand_params`` path on matched
   DGP. If you touch ``solve/demand_adjustment.py``, run these tests
   first.

3. **Public API pinning.** ``tests/test_public_api_pin.py`` audits
   every module that declares ``__all__``:

   * Every non-underscore attribute defined in the module is in
     ``__all__`` (no orphan public names).
   * Every name in ``__all__`` resolves to a real attribute (no
     dead names).

4. **Import roundtrip.** ``tests/test_import_roundtrip.py`` enumerates
   the expected ``__all__`` contents for every package and subpackage.
   Adding a new public symbol requires updating the expected list
   there as well as the ``__all__`` declaration.

5. **Deprecation coverage.** ``tests/test_model_formulation_deprecation.py``
   and ``tests/test_demand_params_rho_alias.py`` lock the warning
   emission logic.

Logging layout
--------------

v0.4 step 18 switched pyRVtest from the legacy pyblp ``output()`` helper
to the standard :mod:`logging` module. Every module that emits
progress/diagnostic messages defines its own logger at the top:
``logger = logging.getLogger(__name__)``. This gives users per-subsystem
control over verbosity without any pyRVtest-specific configuration.

Loggers currently in use:

* ``pyRVtest.problem`` — ``Problem.__init__`` and ``Problem.solve``
  progress ("Initializing the problem ...", "Solving the problem ...",
  "Computing Markups ...", "Absorbing cost-side fixed effects ...",
  plus the rendered dimension / formulation / results tables).
* ``pyRVtest.backends.logit`` — nested-logit column inference ("Inferred
  nesting order ...") when the user relies on auto-detection rather than
  passing ``nesting_ids_columns`` explicitly.
* ``pyRVtest.output`` — the legacy ``output()`` compatibility shim
  (deprecated in v0.4; removed in v0.6). Both its deprecation warnings
  and its forwarded message go through this logger.

Typical user recipes::

    import logging

    # See all pyRVtest progress messages on stderr:
    logging.basicConfig(level=logging.INFO)

    # Keep the general info stream on, but silence just ``Problem.solve``:
    logging.getLogger("pyRVtest.problem").setLevel(logging.WARNING)

    # Only see warnings (fallbacks, unusual recoveries) and above:
    logging.getLogger("pyRVtest").setLevel(logging.WARNING)

    # Route pyRVtest output into a file, separate from stdout:
    fh = logging.FileHandler("pyrvtest.log")
    fh.setLevel(logging.DEBUG)
    logging.getLogger("pyRVtest").addHandler(fh)

pyRVtest does not install its own handlers — following the standard
library guidance, the package only emits records and leaves the handler
configuration to the application. A library with no handler attached
behaves as if :class:`logging.NullHandler` were installed (messages are
silently dropped). Call :func:`logging.basicConfig` in your script or
notebook to actually see the records.

Level assignments in pyRVtest:

* ``logger.info(...)`` — normal progress messages (most call sites).
* ``logger.debug(...)`` — reserved for verbose per-iteration
  diagnostics.
* ``logger.warning(...)`` — unusual but recoverable behaviour (e.g.
  fallback paths).
* ``logger.error(...)`` — reserved for error-context logging before
  raising an exception; in practice the exception message itself is
  sufficient.

Two call sites remain on bare ``print()`` by design:
``pyRVtest.show_agent_guide()`` and its fallback path in
``pyRVtest._agent_guide``, because the whole contract of that helper is
to dump the guide onto stdout for a human/LLM user to read, and the
test suite captures stdout to assert on the content.

Workflows for common tasks
--------------------------

Add a new conduct model
^^^^^^^^^^^^^^^^^^^^^^^

1. Create a new file under ``pyRVtest/models/`` (or add a class to an
   existing thematic file).

2. Inherit from :class:`~pyRVtest.ConductModel`. Implement the two
   abstract math hooks (``_compute_markup``, ``_markup_derivative``).
   Use vectorized numpy; avoid Python-level per-product loops.

3. Export the class from ``pyRVtest/models/__init__.py`` (both the
   import line and the ``__all__`` list).

4. Re-export at the package top level in ``pyRVtest/__init__.py``
   (add to the ``from .models import (...)`` line and the
   ``__all__`` list).

5. Update ``tests/test_import_roundtrip.py`` so the new class name
   is in the expected ``__all__`` entries for both
   ``'pyRVtest.models'`` and (if added to the top level) the
   top-level ``pyRVtest`` ``__all__``.

6. Add a unit test in ``tests/test_models.py`` following the
   existing patterns (ownership validation, markup shape, gradient
   signs).

7. If the new model produces numerically-distinct markups on any
   existing fixture, regenerate the relevant snapshot with an
   explicit justification.

Add a new backend
^^^^^^^^^^^^^^^^^

1. Create a new file under ``pyRVtest/backends/``.

2. Implement the :class:`~pyRVtest.backends.base.DemandBackend`
   protocol: ``n_parameters``, ``theta_names``, ``compute_jacobian``,
   ``compute_hessian``, and the ``perturbed`` context manager.

3. If your backend can supply the DMSS eq. 77 quantities, also
   implement :class:`~pyRVtest.backends.base.SupportsDemandAdjustment`:
   ``demand_moments``, ``xi_gradient``, ``jacobian_gradient``. The
   testing engine uses ``isinstance`` to detect capability.

4. Export the class from ``pyRVtest/backends/__init__.py`` and add
   it to ``__all__``. Add the corresponding entry to
   ``tests/test_import_roundtrip.py``.

5. Add a test under ``tests/test_backends.py`` exercising the
   protocol surface. If the backend implements
   :class:`SupportsDemandAdjustment`, add an equivalence test against
   at least one other backend on a shared DGP.

Add a new pipeline stage
^^^^^^^^^^^^^^^^^^^^^^^^

1. Place the stage under ``pyRVtest/solve/`` with a clear
   ``compute_<stage>(...)`` entry point.

2. Give it explicit input/output types. The goal is that each stage
   can be understood in isolation.

3. Wire it into ``Problem.solve`` at the correct position. Do not
   inline stage logic into ``Problem.solve`` — that is exactly the
   structural problem the refactor is addressing.

4. Add unit tests at ``tests/test_<stage>.py`` that exercise the
   stage in isolation.

5. If the stage is user-facing, re-export at the package top
   level and add to ``__all__``.

Further reading
---------------

* :doc:`migrating_to_v0.4` — the user-facing migration guide.
* :doc:`custom_demand` — worked example of
  :class:`~pyRVtest.backends.UserSuppliedBackend`.
* :doc:`api` — complete API reference generated from docstrings.
* The ``.claude/plans/v0.4-refactor.md`` file in the repository is the
  authoritative source for architectural decisions. The 25 migration
  steps are numbered and cross-referenced throughout the code in
  docstrings and commit messages (``step 1``, ``step 4d``, ``step 11``,
  etc.).
* The ``.claude/handovers/`` directory holds session-by-session
  handover notes. The most recent one is a good starting point for
  catching up on what has changed since this guide was last updated.
