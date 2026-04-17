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

Steps 8, 9, 10, 14, 16, 18, 19, 21, 24, and 25 are still outstanding
and may introduce additional changes before v0.4.0 is tagged.

### Added

- **`DemandBackend` protocol and implementations** (steps 1–4). Core
  protocol in `backends/base.py` with `PyBLPBackend`, `LogitBackend`,
  `NestedLogitBackend`, and `UserSuppliedBackend` implementations.
  Optional `SupportsDemandAdjustment` mixin declares which backends can
  supply first-stage correction inputs (`xi`, `Z_D`, `W_D`, gradients).
- **Class-based `ConductModel` API** (step 5). `Bertrand`, `Cournot`,
  `Monopoly`, `PerfectCompetition`, `MixedCournotBertrand`, and
  `Vertical` classes each own their `compute_markup()`. `ModelFormulation`
  kept as a thin backward-compat bridge that raises `DeprecationWarning`.
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
- **`AGENTS.md` + `docs/agent_guide.rst` + `pyRVtest.show_agent_guide()`**
  (step 23). Architecture tour, deprecation policy, and pointers for
  AI-assisted contributors and users.
- **Property-based tests** (step 20). Market-partition invariance of the
  raw moment, FWL identity for the GMM moment, Bertrand markup
  homogeneity of degree −1 in α (Hypothesis-driven).
- **Snapshot regression suite** (step 0b) + golden DMSS-yogurt
  scaffolding (step 0d, data pending Lorenzo).
- **Public API pin** (step 22). Dynamic `__all__` audit across every
  pyRVtest module (65 parametrized assertions across 32 modules).
- **Minimal CI** (step 24.5). `.github/workflows/ci.yml` runs pytest on
  Ubuntu + Python 3.11 on every push and PR. `mypy --strict` and
  `--doctest-modules` steps stubbed pending steps 17 completion and 21.

### Changed

- **Unified demand-adjustment path** (step 4). The `demand_params` branch
  and the PyBLP-results branch now share a single
  `compute_demand_adjustment` function. Two ~200-line duplicate methods
  on `Problem` deleted.
- **`sigma` → `rho` in `demand_params`** (step 6b). `rho` is the
  canonical key; `sigma` still accepted as a deprecated alias for one
  release (raises `DeprecationWarning`).
- **Module split of `problem.py`**. `Products` extracted to
  `pyRVtest/products.py` (step 2). `ModelFormulation` bridge lives in
  `pyRVtest/models/_adapter.py`. Standard model classes live in
  `pyRVtest/models/standard.py`.
- **`mypy --strict` coverage** (step 17). Eight modules now strict-clean:
  `output`, `data`, `formulation`, `models._adapter`, `models.standard`,
  `results`, `solve.demand_adjustment`, `solve.passthrough`. `problem.py`
  and `markups.py` remain lax with narrow `disable_error_code` lists
  pending the step-8 split.
- **Auto-routing** (pre-step-4 refactor work). PyBLP logit and
  single-`rho` nested logit now route to the analytical demand-adjustment
  path by default; other cases fall back to finite-difference.

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
  directly.
- `demand_params=dict(sigma=…)` raises `DeprecationWarning`; use
  `demand_params=dict(rho=…)` instead.

### Notes for coauthors

- See `MEMO_coauthor_updates.md` for a running, behavior-focused ledger
  of v0.4 changes that affect downstream code.
- The v0.4 test suite is at 388 passed + 3 skipped as of `2cdf2d0`.
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
