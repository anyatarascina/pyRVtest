# Changelog

All notable changes to pyRVtest are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the
project roughly follows [Semantic Versioning](https://semver.org/).

## [Unreleased] — v0.4.0 (work in progress)

Version 0.4 is a substantial refactor of the package architecture. The
primary motivations are (a) encapsulating PyBLP private-attribute access
behind a `DemandBackend` protocol, (b) merging the two parallel
demand-adjustment paths (PyBLP results vs. `demand_params`) into a single
dispatch, and (c) setting up hooks for labor-side conduct testing.
See `.claude/plans/v0.4-refactor.md` for the full design document.

Steps 14 (labor hooks) and 16 (AFSSZ dogfood) are still outstanding and
may introduce additional changes before v0.4.0 is tagged. Step 16 is
data-blocked (~1 week lead time on the AFSSZ panel).

### Migration from v0.3.x

Three user-visible break points. Each deprecation emits a
`DeprecationWarning` for one release before removal:

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
- **Analytical nested-logit Hessian** (step 7). Closed-form
  `compute_analytical_hessian` in `backends/logit.py` for plain logit and
  1-level nested logit; multi-level nesting falls back to finite-difference.
  Validated against finite-diff, PyBLP's own `compute_demand_hessians`, and
  Clairaut symmetry (8 parametrized tests plus a nested-logit-vertical
  snapshot).
- **`pyRVtest.build_passthrough`** (step 11). Stand-alone helper that
  returns the Villas-Boas passthrough matrix per vertical model, either
  for a single market or as a dict across all markets. Clear errors for
  non-vertical models, invalid `market_id`, and missing `hessian_fn` on
  `UserSuppliedBackend`.
- **Instrument construction helpers** (step 13). New
  `pyRVtest.instruments.product` module with `rival_sums`,
  `differentiation_ivs`, `blp_instruments`. New
  `pyRVtest.instruments.labor` with `hausman`, `bartik`,
  `concentration_hhi`. All six accept DataFrame, structured recarray,
  or dict-like `product_data`.
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

### Changed

- **Unified demand-adjustment path** (step 4). The `demand_params` branch
  and the PyBLP-results branch now share a single
  `compute_demand_adjustment` function. Two ~200-line duplicate methods
  on `Problem` deleted; auto-routing sends PyBLP logit and single-`rho`
  nested-logit cases to the analytical path by default, with
  finite-difference as the fallback for everything else.
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
