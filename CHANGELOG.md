# Changelog

All notable changes to pyRVtest are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the
project roughly follows [Semantic Versioning](https://semver.org/).

## [0.4.0rc1] — 2026-04-20

Release candidate for v0.4.0. Version 0.4 is a substantial refactor of
the package architecture. The primary motivations are (a) encapsulating
PyBLP private-attribute access behind a `DemandBackend` protocol, (b)
merging the two parallel demand-adjustment paths (PyBLP results vs.
`demand_params`) into a single dispatch, and (c) setting up hooks for
labor-side conduct testing. See `.claude/plans/v0.4-refactor.md` for
the full design document.

Step 16 (AFSSZ dogfood on a real 910-market-year panel) is still
outstanding and may introduce additional changes before v0.4.0 final is
tagged. Step 16 is data-blocked (~1 week lead time on the AFSSZ panel).

### rc1 fixes (coauthor break-it pass, 2026-04-18)

- **`UserSuppliedMarkups` class.** Pre-computed markup columns are now a
  first-class conduct model:
  ```python
  pyRVtest.UserSuppliedMarkups(markups='mkup_col', ownership='firm_ids')
  ```
  The legacy pattern
  `ModelFormulation(user_supplied_markups='col', ownership_downstream='firm_ids')`
  (no `model_downstream`) used to crash with an `AssertionError` in the
  adapter; it now translates to `UserSuppliedMarkups` and emits the
  standard `ModelFormulation` deprecation warning. Fixes Lorenzo's P0
  regression against carRV's production `conduct_test.py`.
- **K > 30 critical-value warning.** Instrument counts above 30 previously
  fell back to the K=30 critical values silently. A `UserWarning` is now
  emitted once per instrument set when K exceeds the tabulated range.
- **`options.digits` wired through result formatters.** The global
  `pyRVtest.options.digits` setting now controls numeric precision in
  `to_markdown()` / `to_latex()` / `summary_df()` output (previously
  hardcoded to 6 significant figures). Default value changed from 7 to
  6 to match the prior `_dataframe_to_github_markdown` hardcoded format
  (no user-visible change vs. pre-rc1 output at the default setting).
- **`options.verbose` deprecated.** Reading the attribute emits a
  `DeprecationWarning` pointing at the `logging.getLogger('pyRVtest')`
  API that superseded it in v0.4. Assignment stays silent so the
  widely-used `pyRVtest.options.verbose = False` pattern keeps working.
  Removal scheduled for v0.6.
- **`pyproject.toml` added.** Minimal build-system declaration so
  `pip install -e .` works under modern pip (>=23) without falling back
  to legacy setuptools.

### Deferred to v0.4.0 final (tracked, not rc1-blocking)

- numpy 2.x / pyblp 1.1.2 test-suite failures (7 failed / 31 errors on
  Windows + numpy 2.4, including a 2.5% shift in the `analytical_scale`
  snapshot). Needs investigation of pyblp-version vs numpy-version
  attribution before committing to a pin or a code fix.
- `PanelResults` roster-hash validation (audit B1).
- `Problem(demand_backend=...)` public kwarg (audit B3; already flagged
  as future work in `docs/custom_demand.rst`).
- Per-model tax `DeprecationWarning` firing at construction time.

### Migration from v0.3.x

Four user-visible break points. Most emit a `DeprecationWarning` for
one release (slated for removal in v0.6). The per-model tax kwargs get
two releases (removed in v0.7) because they appear in user code much
more frequently. See `docs/migrating_to_v0.4.rst` for the full
deprecation timeline.

1. **Conduct-model specification.** Prefer the new class-based API over
   `ModelFormulation`:
   ```python
   # v0.3
   from pyRVtest import ModelFormulation
   models = [
       ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_id'),
       ModelFormulation(model_downstream='cournot',  ownership_downstream='firm_id'),
   ]

   # v0.4
   from pyRVtest import Bertrand, Cournot
   models = [
       Bertrand(ownership='firm_id'),
       Cournot(ownership='firm_id'),
   ]
   ```
   `ModelFormulation(...)` still works but emits `DeprecationWarning`.
   See `docs/migrating_to_v0.4.rst` for per-model mappings, including
   `Vertical`, `Monopoly`, `PerfectCompetition`, and
   `MixedCournotBertrand`.

2. **`demand_params` parameter name.** `rho` is the canonical nesting
   parameter key; `sigma` is retained as a deprecated alias:
   ```python
   # v0.3
   problem.solve(demand_params=dict(alpha=-1.2, sigma=0.5))

   # v0.4
   problem.solve(demand_params=dict(alpha=-1.2, rho=0.5))
   ```
   Passing `sigma=` raises `DeprecationWarning` and continues to work.

3. **Custom-demand users.** If you previously monkey-patched PyBLP
   internals to plug in a non-PyBLP demand system, subclass
   `pyRVtest.backends.UserSuppliedBackend` instead. Supply
   `compute_jacobian`, `compute_hessian`, and (optionally) the
   `SupportsDemandAdjustment` hooks. See `docs/custom_demand.rst` for a
   worked linear-demand example.

4. **Tax specification: model-level → Problem-level.** Per-model
   `unit_tax='col'` / `advalorem_tax='col'` / `advalorem_payer=...`
   kwargs on `ConductModel` / `ModelFormulation` / `Vertical` are
   deprecated. Specify the tax once on `Problem`; opt individual
   models out via `unit_tax_salient=False` / `advalorem_tax_salient=False`
   (salience-testing pattern). Removal scheduled for **v0.7** (one
   release later than other v0.4 deprecations, since the per-model tax
   pattern is common in existing user code).

Existing v0.3 scripts without custom demand code should run unchanged on
v0.4 modulo one-line deprecation warnings.

### Added

- **`DemandBackend` protocol and implementations** (steps 1–4). Core
  protocol in `backends/base.py` with `PyBLPBackend`, `LogitBackend`,
  `NestedLogitBackend`, and `UserSuppliedBackend` implementations.
  Optional `SupportsDemandAdjustment` mixin declares which backends can
  supply first-stage correction inputs (`xi`, `Z_D`, `W_D`, gradients).
  Encapsulates previously-scattered `pyblp_results._*` accesses behind a
  single documented interface.
- **Class-based `ConductModel` API** (step 5). `Bertrand`, `Cournot`,
  `Monopoly`, `PerfectCompetition`, `MixedCournotBertrand`, and
  `Vertical` classes each own their `compute_markup()`. `ModelFormulation`
  kept as a thin backward-compat bridge that raises `DeprecationWarning`.
  Full migration guide in `docs/migrating_to_v0.4.rst` with one-to-one
  recipes for every `ModelFormulation` shape.
- **Dearing et al. (2026) simple-markup conduct models** (step 12).
  `RuleOfThumb(phi)` and `Keystone()` (the `phi=2` shorthand) implement
  the Dearing Example 1 rule :math:`p = \varphi \cdot mc` as an
  ergonomic wrapper over the existing `cost_scaling` machinery, which
  v0.4 step 12a extends to accept a numeric scalar in addition to a
  column name. `ConstantMarkup(markup)` implements Example 7's fixed
  per-product dollar markup via a new additive-markup plumbing path
  threaded through the `Models` recarray and `evaluate_first_order_conditions`.
  All three classes re-exported at the package level
  (`pyRVtest.RuleOfThumb`, `pyRVtest.Keystone`, `pyRVtest.ConstantMarkup`).
  Backward compatibility: the legacy `PerfectCompetition(cost_scaling='lmbda_col')`
  pattern still works unchanged.
- **Analytical nested-logit Hessian** (step 7). Closed-form
  `compute_analytical_hessian` in `backends/logit.py` for plain logit
  and single-scalar-rho nested logit. Per-nest rho (Cardell-Nevo),
  multi-level nesting, and BLP continue to use the pyblp
  finite-difference path. AFSSZ-style specifications with per-nest rho
  therefore do not benefit from the O(ε²) Hessian improvement; plan
  accordingly when routing analytical vs finite-difference.
  Validated against finite-diff, PyBLP's own `compute_demand_hessians`, and
  Clairaut symmetry (8 parametrized tests plus a nested-logit-vertical
  snapshot).
- **`pyRVtest.build_passthrough`** (step 11). Stand-alone helper that
  returns the Villas-Boas passthrough matrix per vertical model, either
  for a single market or as a dict across all markets. Clear errors for
  non-vertical models, invalid `market_id`, and missing `hessian_fn` on
  `UserSuppliedBackend`.
- **`ProblemResults.passthrough_comparison`** and
  **`ProblemResults.passthrough_matrix`** (OQ 15). Dearing-style
  pass-through diagnostics surfaced on `ProblemResults`.
  `passthrough_comparison` returns a pandas DataFrame with one row per
  `(market_id, unordered model pair)` and a scalar pairwise distance
  between pass-through matrices; three metrics are supported —
  `'frobenius'` (default), `'offdiag_frobenius'` (implements the Dearing
  et al. (2026) Remark 4 distinguishability condition — invariant to
  diagonal-only differences in pass-through), and `'max_abs'`. The
  chosen metric is recorded on `frame.attrs['metric']`.
  `passthrough_matrix` is a thin ergonomic wrapper over
  `build_passthrough`. Both methods currently require every candidate
  model to be `Vertical`; a non-Vertical candidate raises
  `NotImplementedError` with a pointer to the v0.5 scope item for
  per-model closed-form pass-through (Bertrand / Cournot / RuleOfThumb /
  ConstantMarkup / PerfectCompetition).
- **Instrument construction helpers** (step 13). New
  `pyRVtest.instruments.product` module with `rival_sums`,
  `differentiation_ivs`, `blp_instruments`. New
  `pyRVtest.instruments.labor` with `hausman` and `bartik`. All five
  accept DataFrame, structured recarray, or dict-like `product_data`.
  A labor-side `concentration_hhi` helper was prototyped and then
  deliberately removed before release: labor-market HHI is endogenous
  in wages (shares respond to the variable being tested), so it is not
  a valid wage instrument even though the product-side analogue is
  sometimes defensible. See `pyRVtest/instruments/labor.py` for the
  rationale and references.
- **Worked `UserSuppliedBackend` example** (step 15). New
  `docs/custom_demand.rst` with an end-to-end linear-demand DGP and
  accompanying test.
- **`ProblemResults` export methods** (step 9). `to_dataframe()`,
  `summary_df(alpha=0.05)`, `to_latex(...)`, and `to_markdown(...)` on
  `ProblemResults` for friction-free reporting.
- **Stable `reject` column + `alpha` in `DataFrame.attrs`**
  (Open-Question-11 resolution). `ProblemResults.summary_df` emits a
  constant `reject` column name (not `reject_at_{alpha:g}`) and records
  the critical level on `DataFrame.attrs['alpha']`. Downstream
  aggregators can read alpha without re-deriving the column name;
  `PanelResults.rejection_rates` and `PanelResults.summary_df` consume
  and propagate it accordingly.
- **`PanelResults`** (step 10). Multi-problem aggregation class for
  panels of market-years. Mapping-like API (`keys`, `__getitem__`,
  `__iter__`, `__len__`, `__contains__`), plus `to_dataframe()`,
  `rejection_rates(alpha)`, `summary_df`, `to_latex`, `to_markdown`.
  Constructor validates non-empty mapping, `ProblemResults` values, and
  homogeneous candidate-model count across the panel.
- **Structured logging** (step 18). Per-module loggers
  (`pyRVtest.problem`, `pyRVtest.backends.logit`, …) replace the
  in-house `output()` / `print()` sites. Users can silence a subsystem
  with `logging.getLogger("pyRVtest.problem").setLevel(logging.WARNING)`;
  no handlers are installed at import time. `tests/test_logging.py`
  covers emission, silencing, and the shim's deprecation warning. See
  the new "Logging layout" section of `docs/agent_guide.rst`.
- **Custom exception hierarchy** (step 19). New `pyRVtest/exceptions.py`
  with `PyRVTestError`, `ValidationError` (+ `InstrumentDataError`),
  and `BackendError` (+ `DemandBackendError`, `HessianUnavailableError`).
  Every class multi-inherits from the appropriate built-in
  (`ValueError` / `RuntimeError`) so existing `except ValueError:` and
  `pytest.raises(ValueError, ...)` callers work unchanged. Re-exported
  from `pyRVtest.__init__`; documented in `docs/api.rst` under
  "Exceptions".
- **Doctest coverage on public API** (step 21). 47 runnable docstring
  examples across 28 modules, with 9 intentionally skipped blocks that
  require a fitted PyBLP results object. `pytest --doctest-modules`
  wired into CI; `tox -e doctest` available for local runs.
- **Property-based tests** (step 20). Market-partition invariance of the
  raw moment, FWL identity for the GMM moment, Bertrand markup
  homogeneity of degree −1 in α (Hypothesis-driven).
- **Public API pin** (step 22). Dynamic `__all__` audit across every
  pyRVtest module (65 parametrized assertions across 32 modules).
- **`AGENTS.md` + `docs/agent_guide.rst` + `pyRVtest.show_agent_guide()`**
  (step 23). Architecture tour, deprecation policy, and pointers for
  AI-assisted contributors and users.
- **Snapshot regression suite** (step 0b) + golden DMSS-yogurt
  scaffolding (step 0d, data pending Lorenzo).
- **Minimal CI** (step 24.5). `.github/workflows/ci.yml` runs pytest on
  Ubuntu + Python 3.11 on every push and PR. `mypy --strict` and
  `pytest --doctest-modules` steps are active following steps 17 and 21.
- **Problem-level taxes + per-model salience flags** (OQ 14). Pass
  `unit_tax='col'`, `advalorem_tax='col'`, and
  `advalorem_payer='firm'|'consumer'` directly on `Problem(...)` so the
  tax lives on the DGP (where it belongs) rather than being repeated on
  every candidate model. Individual models opt out via
  `Bertrand(..., unit_tax_salient=False)` /
  `advalorem_tax_salient=False` — the mechanism for salience tests
  (e.g., comparing a salient-tax Bertrand to a non-salient-tax Bertrand
  under the same Problem-level tax). Salience flags default to `True`;
  if no Problem-level tax is set the flag is a no-op. The legacy
  per-model `unit_tax` / `advalorem_tax` path still works and wins by
  precedence when both are set (each emits a once-per-session
  `DeprecationWarning` plus a separate conflict-warning when Problem-
  level and model-level values disagree). 18 tests in
  `tests/test_problem_level_taxes.py`.
- **Known-coefficient cost shifters on `Formulation`** (OQ 14). The
  `cost_formulation` now accepts a
  `known_coefficients={'col': gamma, ...}` dict of cost shifters with
  researcher-supplied (non-estimated) coefficients. They enter the
  effective-price line in `Problem.solve`:
  `prices_effective = advalorem_tax_adj * p / (1 + cost_scaling) -
  unit_tax - sum(gamma_k * x_k)`, applied uniformly to every model
  (these are DGP-level primitives, not behavioral choices). Per-unit
  taxes are the leading special case; Dearing et al. (2026) work with
  a broader class of such shifters. Validation at Formulation
  construction time (dict type, finite numeric coefficients, no
  overlap with the formula); column-existence check at
  `Problem.__init__`. 16 tests in `tests/test_known_coefficients.py`.

### Changed

- **Unified demand-adjustment path** (step 4). The `demand_params` branch
  and the PyBLP-results branch now share a single
  `compute_demand_adjustment` function. Two ~200-line duplicate methods
  on `Problem` deleted; auto-routing sends plain logit and
  single-scalar-rho nested-logit cases to the analytical path by
  default. Per-nest rho (Cardell-Nevo), multi-level nesting, and BLP
  continue through the pyblp finite-difference fallback.
- **`sigma` → `rho` in `demand_params`** (step 6b). `rho` is the
  canonical key; `sigma` still accepted as a deprecated alias for one
  release (raises `DeprecationWarning`).
- **Error messages follow an expected / received / fix structure**
  (step 19). All 120 `raise` sites in `pyRVtest/` rewritten. User-facing
  validation errors now state what was checked, what was actually
  received, and a concrete fix. For example, from `Products`:
  ```
  Expected the 'market_ids' column to be one-dimensional. Received
  shape (200, 2). Fix: pass a single vector of market identifiers, not
  a multi-column array.
  ```
  Internal-invariant failures are prefixed `"pyRVtest internal error:"`
  and kept terse.
- **Labor-side conduct testing hooks** (step 14). New `market_side`
  parameter on `Problem` (default `'product'`; set to `'labor'` for
  monopsony / wage-setting conduct tests). Four new labor conduct model
  classes in `pyRVtest.models.labor`: `Monopsony`, `BertrandWages`,
  `CournotEmployment` (all with real sign-flipped markup formulas),
  plus `NashBargaining` as a v0.5 stub. Skeleton `LaborSupplyBackend`
  in `pyRVtest.backends.labor.nested_logit_labor` honors the
  `DemandBackend` protocol; real math deferred to v0.5 when labor data
  arrives. Labor-mode column-name defaults are `'wages'` (in place of
  `'prices'`) and `'employment_share'` (in place of `'shares'`); the
  default advertises units because the canonical `shares` column is
  treated as a share in `[0, 1]`, so users with raw employment counts
  must normalize first. Sign-convention validation rejects non-positive
  wages or employment shares with rich error messages.
  `ProblemResults.__str__` swaps the header banner under labor mode
  ("markdown / MRP / wage" instead of "markup / MC / price"). Labor-
  side models cannot be mixed with product-side models. `PerfectCompetition`
  stays side-neutral (zero markup has no sign convention).
  `CustomConductModel` now requires an explicit `side='labor'` opt-in
  when used under labor mode (and rejects `side='labor'` under the
  default product mode) because the user-supplied `markup_fn`
  implicitly picks a sign convention and silent acceptance on either
  side would let a product-side formula leak into a labor problem
  unnoticed. The symmetric cross-side validator also now rejects
  labor-side conduct classes (Monopsony, BertrandWages, etc.) under the
  default `market_side='product'`. 39 tests in
  `tests/test_labor_mode.py`.
- **`Problem.solve` split into staged pipeline** (step 8). The ~200-line
  monolithic `solve()` method is now a thin orchestrator that calls
  staged modules under `pyRVtest/solve/`: `markups.compute`,
  `orthogonalize.residualize`, `endogenous_cost.iv_correct`,
  `demand_adjustment.apply`, and `test_engine.compute`. `problem.py`
  shrank from 1733 to 1328 lines (−23%). Each stage is a pure function
  with its own logger (`pyRVtest.solve.markups`,
  `pyRVtest.solve.orthogonalize`, etc.). User-facing behavior is
  bit-identical: 471 tests + snapshot suite pass unchanged.
- **Module split of `problem.py`** (internal). `Products` extracted to
  `pyRVtest/products.py` (step 2). `ModelFormulation` bridge lives in
  `pyRVtest/models/_adapter.py`; standard model classes live in
  `pyRVtest/models/standard.py`.
- **`mypy --strict` coverage** (step 17, internal). Eight modules
  strict-clean: `output`, `data`, `formulation`, `models._adapter`,
  `models.standard`, `results`, `solve.demand_adjustment`,
  `solve.passthrough`. `problem.py` and `markups.py` remain lax with
  narrow `disable_error_code` lists pending the step-8 split.

### Fixed

- **`Dict_K` / `Dict_Z_formulation` shared class state** (step 6a).
  Previously class attributes, so two concurrent `Problem` instances
  could accumulate each other's state. Now per-instance. Three
  regression tests pin the fix.
- **First-stage correction weight-matrix and sign bugs** (`b3b08a3`,
  pre-v0.4). Carried over from CClean-fixes.

### Deprecated

- `ModelFormulation(...)` now raises `DeprecationWarning`; use the
  class-based `ConductModel` API (`Bertrand`, `Cournot`, `Vertical`, …)
  directly. See the "Migration from v0.3.x" section above.
- `demand_params=dict(sigma=…)` raises `DeprecationWarning`; use
  `demand_params=dict(rho=…)` instead.
- `pyRVtest.output.output()` is a logging-backed compatibility shim and
  emits a once-per-session `DeprecationWarning`. Use
  `logging.getLogger("your.module").info(...)` in new code.
- **Per-model `unit_tax` / `advalorem_tax` / `advalorem_payer`** on
  `ConductModel`, `Vertical`, and `ModelFormulation` are deprecated in
  favor of the Problem-level kwargs. The model-level fields still work
  (and win by legacy precedence when both are set) but emit a once-
  per-session `DeprecationWarning`. **Removal scheduled for v0.7**
  (one release later than other v0.4 deprecations — the per-model tax
  pattern is common enough in existing user code to warrant an extra
  release of runway). Migrate by moving the tax column to
  `Problem(..., unit_tax='col', ...)` and using `unit_tax_salient=False`
  on individual models for salience-test opt-outs.

### Notes for coauthors

- See `MEMO_coauthor_updates.md` for a running, behavior-focused ledger
  of v0.4 changes that affect downstream code.
- The v0.4 test suite is at 388 passed + 3 skipped as of `2cdf2d0`, and
  continues to grow; steps 18 and 19 each add a dedicated test module
  (`tests/test_logging.py`, `tests/test_error_messages.py`).
- Data-dependent regression tests for DMSS yogurt (`step 0d`) and the
  Dearing `LearningFirmConduct` reference (`step 12`) remain blocked on
  external inputs.

## [0.3.2] — prior

See `git log v0.3.2` for pre-v0.4 history. Notable line items from the
CClean-fixes branch merged before the v0.4 refactor started:

- 12 correctness fixes (first-stage correction, sign conventions,
  ownership handling).
- 900× clustering speedup.
- `demand_params` feature for passing demand parameters directly.
- `endogenous_cost_component` support for non-constant marginal cost.
