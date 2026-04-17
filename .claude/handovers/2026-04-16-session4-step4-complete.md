# Handover: v0.4 step 4 complete (session 4, 2026-04-16)

**Date:** 2026-04-16 (fourth session of the day, continuing from `2026-04-16-session3-step0-complete.md`)
**Branch:** `v0.4-refactor` at `b7a3f12` on origin (`anyatarascina/pyRVtest`)
**Tag:** `v0.3.3-stable` (annotated, at `47b4457`) still the baseline-protection anchor
**Status:** Step 4 (unify demand adjustment behind the backend protocol) fully landed with an extra correctness fix (4i). Test suite **187 passed + 3 skipped** (was 155 + 3 at end of session 3). Next work item: step 5 (`ConductModel` class-based API, Option B).

## TL;DR

1. **Step 4 landed in 11 commits (4a–4i + 2 correctness follow-ups).** The two inline demand-adjustment methods in `problem.py` (~520 lines combined) are deleted. `Problem.solve`'s `demand_adjustment=True` branch now routes through a single `compute_demand_adjustment` function in `pyRVtest/solve/demand_adjustment.py`, generic over any `SupportsDemandAdjustment` backend.

2. **Closed a silent capability gap.** Pre-v0.4 the analytical path (demand_params) returned `gradient_gamma_per_instrument=None`, silently disabling the endogenous-cost gamma correction. Post-v0.4 both paths compute it.

3. **Closed a latent correctness bug (step 4i).** Pre-v0.4 the analytical path did NOT apply `advalorem_tax_adj / (1 + cost_scaling)` to `gradient_markups`. PyBLP path did. Step 4d initially matched the analytical behavior — regressing PyBLP users with nontrivial taxes. Step 4i applies the factor uniformly in both paths. Zero snapshot drift (no existing fixture has taxes) but it's a behavioral fix for demand_params users who had taxes — worth flagging in v0.4 CHANGELOG.

4. **Auto-routing from `demand_results` to analytical backend** when the pyblp estimation is pure logit (K2=0, rho.size=0) or single-scalar-rho nested logit (K2=0, rho.size=1). Per-nest rho (Cardell-Nevo) and BLP stay on `PyBLPBackend` with finite-diff (the AFSSZ L-level formulation can't represent per-nest rho, and there's no analytical form through the BLP contraction).

5. **Validated `_nested_logit_jacobian_derivative`** against finite-diff — 8 parametrized tests covering 1-level (5 configs) and 2-level (3 configs × 2 derivatives) nested logit. Hand-derived formula was correct but un-validated before this session; confidence now high.

6. **Independent math validation** via Option B tests (`TestOptionBHandDerivedGroundTruth`): Berry inversion xi, 2SLS-residualized H, and Bertrand `d(markup)/d(alpha) = -mu/alpha` identity all checked against hand-derived ground truth on a minimal fixture. Catches the class of bugs the missing DMSS yogurt replication test would catch, on synthetic data instead of real data.

---

## Commits this session

All on `v0.4-refactor`, all pushed to `origin` (`anyatarascina/pyRVtest`).

| Commit | Scope |
|--------|-------|
| `4312a4c` | 4a — thread `backend.compute_hessian` through vertical Villas-Boas passthrough in `_compute_markups` |
| `783c6f7` | 4b — extract `_residualize_on_xd` helper (2SLS profile-out) to `solve/demand_adjustment.py`; refactor `PyBLPBackend.xi_gradient` to call it |
| `911ca57` | 4c — add `SupportsDemandAdjustment` methods (`demand_moments`, `xi_gradient`, `jacobian_gradient`) to `LogitBackend`; `NestedLogitBackend` inherits them by overriding `self._sigma` |
| `8bc0df4` | 4d — land unified `compute_demand_adjustment(backend, problem, ...)` in `solve/demand_adjustment.py`; not yet wired into `Problem.solve` |
| `6cf0f38` | Option A + Option B pre-4e gates — cross-path parity test with endogenous_cost (xfailed pre-4e) + hand-derived Berry / H / Bertrand identity tests |
| `7345bb9` | 4e — wire `Problem.__init__` to construct `self._demand_backend`; `Problem.solve` calls unified function; removes Option A xfail; updates `first_stage_pyblp_path` snapshot (analytical implicit-diff vs finite-diff, deliberate per §5 rule) |
| `25d2c24` | A+C — 8 parametrized finite-diff validation tests for `_nested_logit_jacobian_derivative`; accurate PyBLPBackend docstring on finite-diff precision (the "ULP" claim was wrong; it's ~1e-9 for logit, O(eps²) for BLP) |
| `139a23c` | B — `_backend_from_demand_results` auto-routes plain logit and single-scalar-rho nested logit to analytical backends; updates `first_stage_pyblp_path` snapshot again |
| `c357c7d` | 4f — delete `_compute_analytical_demand_adjustment` (295 lines) + `_compute_demand_adjustment_gradient` (189 lines) + `_compute_first_difference_markups` + `_compute_perturbation` (4 dead methods, 519 lines total); update test_first_stage_correction.py's gradient-markups test to call unified function via backend |
| `98e26c9` | 4g — delete `pyRVtest/demand_jacobian.py` shim; drop obsolete kwargs `demand_jacobian`/`demand_alpha`/`demand_sigma`/`demand_nesting` from `_compute_markups`; wrap `shares_t` with `np.asarray` for DataFrame robustness; new regression test; route `_perturb_and_build_markups` through `self._demand_backend` |
| `b41a82b` | 4h — Hypothesis-driven row-permutation invariance test for `demand_adjustment=True` path (25 examples) |
| `b7a3f12` | 4i — apply `advalorem_tax_adj[m] / (1 + cost_scaling[m])` factor to `gradient_markups[m]` at end of `compute_demand_adjustment`; per-observation broadcasting; new regression test |

---

## Architectural state

### `pyRVtest/solve/demand_adjustment.py`

Public API:

- `compute_demand_adjustment(backend, problem, M, N, markups, advalorem_tax_adj, cost_scaling, marginal_cost_base=None)` → 6-tuple `(gradient_markups, H_prime_wd, H, h_i, h, gradient_gamma_per_instrument)`.
- `_residualize_on_xd(dxi_dtheta, X_D, Z_D, W_D)` — shared 2SLS profile-out helper used by backends and the unified function.

Implementation structure:

1. Extract `xi, Z_D, W_D = backend.demand_moments()` and `partial_xi_theta = backend.xi_gradient()`.
2. Build `H`, `H_prime_wd`, `h_i`, `h`.
3. Markup gradient: closed-form implicit differentiation of FOC for Bertrand / Cournot / Monopoly / MixCournotBertrand (uses `backend.compute_jacobian(t)` + `backend.jacobian_gradient(t)`); finite-diff via `backend.perturbed(k, ±eps/2)` for vertical / custom models.
4. Residualize `gradient_markups` on cost shifters (FWL).
5. Apply tax / cost-scaling factor `advalorem_tax_adj[m] / (1 + cost_scaling[m])` to each model's gradient (4i).
6. Gamma gradient (endogenous cost) via finite-diff through `_compute_iv_correction`, with tax adjustment on perturbed markups.

`UserSuppliedBackend` without `SupportsDemandAdjustment` raises `TypeError` citing the missing protocol.

### `pyRVtest/backends/`

- `base.py` — `DemandBackend` and `SupportsDemandAdjustment` Protocol classes (unchanged since session 3).
- `pyblp.py` — `PyBLPBackend` wraps `pyblp.ProblemResults`. `jacobian_gradient` finite-diff via `perturbed` context manager. Docstring (as of 25d2c24) documents precision analysis: ~1e-9 for plain logit (D linear in alpha at restored shares), O(eps²) for nested logit / BLP (D nonlinear).
- `logit.py` — module-level analytical functions (`compute_analytical_jacobian`, `_nested_logit_jacobian`, `_nested_logit_jacobian_derivative`, etc.) and `LogitBackend` class. Implements `SupportsDemandAdjustment` (4c); keyed on `self._sigma` which is `[]` for plain logit.
- `nested_logit.py` — `NestedLogitBackend(LogitBackend)`. Delegates to `super().__init__` and overrides `self._sigma` (filtered nonzero) + `self._nesting_ids_columns`. Inherits `demand_moments`/`xi_gradient`/`jacobian_gradient` unchanged.
- `user.py` — `UserSuppliedBackend`. Core `DemandBackend` only; does NOT implement `SupportsDemandAdjustment`.

### `Problem.__init__` routing

`_construct_demand_backend` (problem.py) dispatches based on what the user supplied:

```
demand_results is not None -> _backend_from_demand_results(r):
    K2 > 0 (BLP)                       -> PyBLPBackend
    K2 = 0 and r.rho.size == 0          -> LogitBackend (analytical, precision gain modest)
    K2 = 0 and r.rho.size == 1          -> NestedLogitBackend(sigma=[rho])  (analytical, material gain)
    K2 = 0 and r.rho.size > 1           -> PyBLPBackend (per-nest Cardell-Nevo; AFSSZ L-level can't represent)

demand_params is not None:
    [s for s in sigma if s > 0] non-empty -> NestedLogitBackend
    else                                  -> LogitBackend
```

Falls back to `PyBLPBackend` if the raw product_data is missing columns pyblp packed into X1/ZD (e.g., fixed-effect dummies).

pyblp does NOT support multi-level nesting natively (verified at `pyblp/primitives.py:179-180`): `nesting_ids` must be 1-dimensional. 2-level nested logit can only enter pyRVtest through the `demand_params` path with `sigma=[rho_1, rho_2]` and `nesting_ids_columns=['col_1', 'col_2']`. The AFSSZ multi-level formulas handle it cleanly and are validated.

### `Problem.solve` demand_adjustment block

Now a single call:

```python
if demand_adjustment:
    from .solve.demand_adjustment import compute_demand_adjustment
    mc_base = marginal_cost_base if self.endogenous_cost_component is not None else None
    gradient_markups, H_prime_wd, H, h_i, h, gradient_gamma_per_instrument = compute_demand_adjustment(
        self._demand_backend, self, M, N, markups,
        advalorem_tax_adj, cost_scaling, mc_base,
    )
```

---

## Test coverage

Total: **187 passed + 3 skipped**.

Key test files and what they cover:

- `tests/test_backends.py` — backend protocol conformance (`PyBLPBackend`, `LogitBackend`, `NestedLogitBackend`, `UserSuppliedBackend`); backend-equivalence in `_compute_markups`; vertical Hessian threading (4a); DataFrame regression (4g).
- `tests/test_demand_adjustment.py` — `_residualize_on_xd` unit tests (4b); backend routing tests (B); Option A cross-path parity with endogenous cost (post-4e sanity); Option B hand-derived ground truth (Berry xi, H, Bertrand identity); `TestNestedLogitJacobianDerivative` (A); `TestGradientMarkupsAppliesTaxFactor` (4i).
- `tests/test_properties.py` — row-permutation invariance without demand_adjustment (pre-existing) and with demand_adjustment (4h, 25 Hypothesis examples).
- `tests/test_first_stage_correction.py` — two-path TRV/F/MCS equivalence at atol=1e-4 (pre-existing, 9 tests); `test_gradient_markups_match` rewritten in 4f to use unified function via backends.
- `tests/test_snapshots.py` — 6 Step 0 snapshots at atol=1e-10. `first_stage_pyblp_path` updated twice in this session (4e for inline→unified; 4g-adjacent for PyBLP→Logit auto-routing).
- `tests/replication/test_dmss_yogurt.py` — 3 skipped placeholders awaiting Lorenzo's input (Step 0d).

### Known limitations flagged in docstrings / comments

1. **BLP demand (K2>0)**: no analytical dD/d(theta); finite-diff via `backend.perturbed` with O(eps²) error. Documented in `PyBLPBackend.jacobian_gradient` docstring.
2. **Per-nest rho nested logit (K2=0, rho.size>1)**: stays on `PyBLPBackend` finite-diff. AFSSZ L-level formulation assumes one sigma per level; Cardell-Nevo has one rho per nest. Genuinely different math. Documented.
3. **Tax adjustment for analytical path was broken pre-v0.4** — fixed in 4i. Silent bug affected demand_params users with nontrivial `advalorem_tax` or `cost_scaling` columns. Worth flagging in CHANGELOG.

---

## Outstanding items for Chris (end of session)

1. **DMSS yogurt golden file (Step 0d).** Still blocked on Lorenzo's data + pinned TRV/F/MCS. Scaffold at `tests/replication/test_dmss_yogurt.py`; Option A/B substitutes cover math correctness but not paper-replication validation.
2. **Coauthor memo.** Should be updated with this session's work — big changes (capability fix, tax-adjustment fix, inline methods deleted) affect how coauthors would characterize pre-v0.4 vs v0.4 output.
3. **v0.4 CHANGELOG drafting.** When step 25 rolls around, the capability fixes flagged in 4d commit message and step 4i commit message need user-facing notes.
4. **`v0.3.3-stable` tag placement.** Unchanged since session 3. If step 0d materially changes any output when Lorenzo's inputs arrive, create `v0.3.4-stable` rather than moving the tag.

---

## Uncertainty flags for next Claude

1. **Two "I deferred correctness" mistakes this session, both caught by Chris.** (a) After 4d I characterized the missing tax_adj on gradient_markups as "a known limitation under nonzero taxes; test fixtures currently have no taxes so byte-level equivalence is preserved" — implying deferral was fine. Chris asked "do we need this?" and the correct answer was yes. Fix was a one-line multiplication plus a regression test (step 4i). (b) On pyblp nested-logit routing I initially asserted "per-nest rho ≠ AFSSZ L-level sigma" when in fact single-scalar rho IS the L=1 case and fully supported. Chris pushed back; I rechecked pyblp source and updated the routing. **Calibration note for future work:** when my instinct is to defer something with "edge case, no existing test," stop and reconsider whether the edge users would actually be wrong — that's exactly when attention is warranted.
2. **Mixed-models and vertical analytical derivatives remain finite-diff.** The unified function falls back to `backend.perturbed` for vertical (`model_upstream is not None`) and custom (`custom_model_specification is not None`) models. Deriving analytical d(markup)/d(theta) through the Villas-Boas passthrough is out of v0.4 scope; possible future work.
3. **`_compute_markups` retains the `pyblp_results`-direct fallback branch.** `build_markups(product_data, pyblp_results)` (public user API) doesn't go through a backend. If a user calls `build_markups` directly they hit `pyblp_results.compute_demand_jacobians()` / `construct_passthrough_matrix` rather than the backend path. Behavior is preserved pre/post-v0.4. If step 8 restructures `solve/` more aggressively this may need revisiting.
4. **Option A's cross-path parity test** is now post-4e at atol=5e-9 (not a tight pre-gate). Its diagnostic power is limited: it catches gross correction-omission bugs but not subtle formula errors.
5. **187 passed is mostly green** but ~300 deprecation warnings from `float(pyblp_results.beta[...])` in test files. NumPy 1.25 will error on these. Pre-existing; worth a separate cleanup PR.

---

## Files changed this session

**Added:**
- `pyRVtest/solve/demand_adjustment.py` — populated from placeholder (4b/4d/4i)
- `tests/test_demand_adjustment.py` — new file for helper + backend SupportsDemandAdjustment + compute_demand_adjustment + routing + Option A/B/C + tax-factor tests

**Modified (major):**
- `pyRVtest/problem.py` — −519 lines (dead methods); +~140 lines (`_construct_demand_backend`, `_backend_from_demand_results`, `_augment_with_intercept_column`, helpers); `Problem.solve` block collapsed to single call
- `pyRVtest/backends/logit.py` — `LogitBackend.__init__` accepts demand-adjustment state; new `SupportsDemandAdjustment` methods
- `pyRVtest/backends/nested_logit.py` — `__init__` delegates to super; filter nonzero sigmas at construction
- `pyRVtest/backends/pyblp.py` — `xi_gradient` calls `_residualize_on_xd` helper; accurate finite-diff precision docstring
- `pyRVtest/markups.py` — drop obsolete kwargs; `np.asarray(shares_t)` robustness; remove obsolete `demand_jacobian` vertical branch
- `tests/test_backends.py` — 4a vertical Hessian tests; DataFrame robustness regression; delete 2 obsolete equivalence tests
- `tests/test_properties.py` — +4h permutation-with-demand_adjustment test
- `tests/test_first_stage_correction.py` — gradient-markups test rewritten to use unified function via backends
- `tests/test_import_roundtrip.py` — track `pyRVtest.solve.demand_adjustment.__all__` growth
- `tests/snapshots/first_stage_pyblp_path.json` — updated twice (4e and B)

**Deleted:**
- `pyRVtest/demand_jacobian.py` (the step-3c shim)

---

## Key files for the next Claude to read first

1. `.claude/plans/v0.4-refactor.md` — the full refactor plan. §5.1 documents step 4 sub-commit table (all landed); step 5 is the next row in §5. §5.1 should probably be archived or annotated now that step 4 is done; the sub-commit planning for step 5 will need its own §5.2 if it needs the same level of detail.
2. This handover.
3. `pyRVtest/solve/demand_adjustment.py` — the unified function. 400+ lines; the heart of what step 4 produced.
4. `pyRVtest/problem.py` — `_construct_demand_backend` and `_backend_from_demand_results` are the new routing logic worth understanding.
5. `tests/test_demand_adjustment.py` — the canonical source for what coverage step 4 built; useful when step 5 adds ConductModel and we want to keep parity.
6. `MEMO_coauthor_updates.md` — needs updating to reflect this session.

---

## Next work item — Step 5: class-based `ConductModel` API

Per the migration table in `.claude/plans/v0.4-refactor.md` §5:

> Step 5 | Class-based `ConductModel` API (Option B). Add **mechanical** model classes (Bertrand, Cournot, Monopoly, PerfectCompetition, MixCournotBertrand, PartialCollusion). `ModelFormulation` becomes deprecation-warning alias (with `once-per-session` hygiene). New `Problem(models=[...])` parameter. `ConstantMarkup`, `RuleOfThumb`, `CostPlus` are **not** added here — they depend on Dearing notation verification (step 12). | All step-0 tests + backward-compat test for ModelFormulation alias

Key design constraints (from Decisions Log):

- **Option B (class-based) chosen over Option A (string-based).** Polymorphism, type safety, and AI navigability outweigh migration cost (small coauthor base).
- **Mechanical classes only in step 5.** `ConstantMarkup` / `RuleOfThumb` / `CostPlus` parametrizations require reading `LearningFirmConduct.pdf` — that's step 12.
- **Deprecation-warning hygiene** — each site fires once per Python session via `warnings.simplefilter('once', DeprecationWarning)` in the package `__init__`.
- **`ModelFormulation` alias must stay working.** Existing user code that does `pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids')` must continue producing the same markups and TRV. Deprecation warning fires on construction but the object still works.

Suggested sub-commit structure (draft; can refine on entering step 5):

- 5a — introduce `ConductModel` base class + `Bertrand` / `Cournot` / `Monopoly` / `PerfectCompetition` / `MixCournotBertrand` / `PartialCollusion` subclasses; unit tests asserting each class produces the same markup (`compute_markup`) as the existing string-based dispatch in `evaluate_first_order_conditions`.
- 5b — new `Problem(models=[Bertrand(...), ...])` parameter; internal conversion from class list to the existing `Models` recarray format that the rest of the pipeline consumes. `models=` and `model_formulations=` are mutually exclusive (TypeError if both).
- 5c — `ModelFormulation` alias: constructor emits deprecation warning, delegates to the class-based construction. Existing tests that use `ModelFormulation` keep passing and now emit a warning; pytest `filterwarnings` handles them.
- 5d — incremental `mypy --strict` on the new classes + `__all__` in `pyRVtest/models/__init__.py`.
- 5e — property tests: class-based API must produce byte-identical markups / TRV / F / MCS as the string-based `ModelFormulation` path on all existing fixtures.

Step 5 touches user-facing API more than step 4 did. Plan to keep the `ModelFormulation` -> ConductModel bridge small and fully tested before deleting anything (mirror the 4a–4g staging discipline).

Snapshot risk: **near zero** if the class-based API is a pure re-dispatch to the same math. No expected changes to `tests/snapshots/*.json`. Will verify as soon as 5a lands.

---

**Push status:** all commits on `origin/v0.4-refactor` at `b7a3f12`. No uncommitted changes.
