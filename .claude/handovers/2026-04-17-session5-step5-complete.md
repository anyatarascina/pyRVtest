# Handover: v0.4 steps 4 and 5 complete (session 5, 2026-04-17)

**Date:** 2026-04-17 (continuing from `2026-04-16-session4-step4-complete.md`; that earlier handover covers step 4 in detail)
**Branch:** `v0.4-refactor` at `ebb8779` on origin (`anyatarascina/pyRVtest`)
**Tag:** `v0.3.3-stable` (annotated, at `47b4457`) remains the baseline-protection anchor
**Status:** Step 5 (class-based `ConductModel` API with ramp) fully landed. Test suite **248 passed + 3 skipped** (started step 5 at 186 + 3; +62 tests across six sub-commits). Next: step 6 (`Dict_K` class→instance fix + rho-canonical naming with sigma deprecation alias).

## TL;DR

1. **Step 5 landed in 6 sub-commits (5a–5e + a mid-course 5b' refactor).** Users can now write the class-based API:

   ```python
   pyRVtest.Problem(
       ...,
       models=[
           pyRVtest.Bertrand(ownership='firm_ids'),
           pyRVtest.PerfectCompetition(),
       ],
   )
   ```

   Old string-based `ModelFormulation` + `model_formulations=` keeps working with a once-per-session `DeprecationWarning`. Migration guide lives at `docs/migrating_to_v0.4.rst`. Scheduled removal: v0.6.

2. **`ConductModel` / `Vertical` is the canonical internal representation** (after 5b'). `ModelFormulation` is a translated input alias; it's converted to `ConductModel` via `from_model_formulation` at `Problem.__init__`, and the internal pipeline (`Models.__new__`, `_compute_markups`, `compute_demand_adjustment`) operates on the class-based form throughout.

3. **Byte-identical parity** between the old and new APIs on every Step 0 snapshot fixture — the 6-scenario parity test in `tests/test_models_step_0_parity.py` asserts markups, TRV, F, MCS_pvalues, and cost_param all agree at atol=1e-14.

4. **`Vertical` wrapper** — bilateral-oligopoly models use an explicit composer class:

   ```python
   pyRVtest.Vertical(
       downstream=pyRVtest.Bertrand(ownership='firm_ids'),
       upstream=pyRVtest.Monopoly(ownership='manufacturer_ids'),
       vertical_integration='vi_col',
   )
   ```

   Tax / `vertical_integration` / `cost_scaling` / `user_supplied_markups` live on the wrapper (apply to combined model); each inner class carries its own `ownership` / `kappa_specification` / `mix_flag`. `Vertical` validates that those "combined-model" fields are NOT set on inner conducts.

5. **`PartialCollusion` is its own class** — inherits from `Bertrand` mathematically but requires `kappa_specification` at construction. Signals intent at call site.

6. **`CustomConductModel`** — class-based replacement for `model_downstream='other'` + `custom_model_specification={name: fn}`. Takes a `markup_fn` callable and a `name` kwarg.

---

## Commits this session

All on `v0.4-refactor`, all pushed to `origin`.

| Commit | Scope |
|--------|-------|
| `171ded9` | 5a — ConductModel base + 7 subclasses + Vertical composer + 41 unit tests |
| `74b6755` | 5b — `Problem(models=[...])` kwarg via `to_model_formulations` forward adapter; 10 integration tests |
| `0e4c9a6` | 5b' — Models.__new__ rewritten to consume ConductModel/Vertical directly; `from_model_formulation` reverse adapter; ModelFormulation no longer on the internal pipeline |
| `1123f04` | 5c — ModelFormulation emits DeprecationWarning on first construction per session (class-level flag, no global warnings state mutation); 4 new tests |
| `89e4c77` | 5d — migration guide `docs/migrating_to_v0.4.rst` with before/after examples for every case (simple conduct, partial collusion, mix, custom callable, vertical, taxes, user-supplied markups) |
| `ebb8779` | 5e — Step 0 fixture parity: byte-identical markups/TRV/F/MCS/cost_param between legacy and class-based APIs on all 6 snapshot scenarios |

Plus the session 4 handover `6100ab0` that documented step 4 completion.

---

## Architectural state

### `pyRVtest/models/` package

```
pyRVtest/models/
├── __init__.py        # re-exports all public classes
├── base.py            # ConductModel abstract base
├── standard.py        # Bertrand, Cournot, Monopoly, PerfectCompetition
├── mixed.py           # MixCournotBertrand (per-market mix_flag variants)
├── collusion.py       # PartialCollusion (inherits Bertrand, requires kappa)
├── custom.py          # CustomConductModel (user-supplied callable)
├── vertical.py        # Vertical composer
├── _adapter.py        # from_model_formulation / from_model_formulations (reverse adapter)
├── constant.py        # placeholder for step 12 (ConstantMarkup, RuleOfThumb, CostPlus)
└── labor.py           # placeholder for step 14 (Monopsony, BertrandWages, etc.)
```

### `ConductModel` base class

Abstract methods: `_compute_markup(O, D, s) -> (J, 1)` and `_markup_derivative(O, D, dD, s, mu) -> (J,)`.

Shared config attributes (all optional, set by subclass `__init__` or the base `__init__`):
- `ownership`, `kappa_specification`, `user_supplied_markups`
- `unit_tax`, `advalorem_tax`, `advalorem_payer`, `cost_scaling`
- `vertical_integration`, `mix_flag`

For non-vertical models these attributes live on the bare conduct instance. For `Vertical` models the "combined-model" fields (`vertical_integration`, taxes, `cost_scaling`, `user_supplied_markups`) live on the `Vertical` wrapper instead. `Vertical.__init__` validates this split.

### `Models` recarray construction (post-5b')

`Models.__new__(models, product_data)` accepts a sequence of:
- `ConductModel` / `Vertical` instances (preferred, canonical form).
- `ModelFormulation` instances (translated inline via `from_model_formulation`).

Produces the same structured recarray the downstream pipeline has always consumed. No changes to `_compute_markups`, `compute_demand_adjustment`, `_analytical_markup_derivative`, or `ProblemResults`.

### `Problem.__init__` flow

```
user input:
  models=[Bertrand(...)]  OR  model_formulations=(ModelFormulation(...),)
    │
    ▼
  convert ModelFormulation → ConductModel/Vertical (if legacy path)
    │
    ▼
  Problem._models : List[ConductModel | Vertical]   (canonical)
  Problem.model_formulations : tuple or None         (back-compat attr)
    │
    ▼
  Models.__new__(Problem._models, product_data)
    │
    ▼
  Problem.models : RecArray  (downstream pipeline's input)
```

---

## Test coverage (248 passed + 3 skipped)

New tests landed during step 5:

| File | Tests | What it covers |
|------|-------|----------------|
| `tests/test_models.py` | 41 | `_compute_markup` and `_markup_derivative` on each class against the legacy string dispatch (`evaluate_first_order_conditions`, `_analytical_markup_derivative`) on parametrized random market fixtures at atol=1e-14. Plus guards: MixCournotBertrand.mix_flag required, PartialCollusion kappa required, Vertical type checks, Vertical rejects config on inner conducts, advalorem_payer validation. |
| `tests/test_models_integration.py` | 10 | End-to-end parity: `Problem(models=[Bertrand(...), PC()])` vs `Problem(model_formulations=(ModelFormulation(...)))` byte-identical markups + TRV/F/MCS on a shared DGP. Covers Bertrand+PC (with demand_adjustment=True), Cournot+PC, Monopoly+PC, PartialCollusion+PC, MixCournotBertrand+PC, Vertical Bertrand+Monopoly, CustomConductModel matching Bertrand. Plus kwarg-exclusivity validation. |
| `tests/test_model_formulation_deprecation.py` | 4 | `ModelFormulation()` emits DeprecationWarning on first construction; does NOT re-emit on subsequent; new API (Bertrand/Vertical) is silent. |
| `tests/test_models_step_0_parity.py` | 6 | Byte-identical markups/TRV/F/MCS/cost_param on each of the 6 Step 0 snapshot scenarios: analytical_base, analytical_clustering, analytical_base_fe, analytical_scale, first_stage_pyblp_path, first_stage_demand_params_path. |

Plus the step-4 correctness tests from the previous session (185 total pre-step-5, 248 now).

### Known limitations / deliberate choices

1. **MixCournotBertrand has a split API.** `_compute_markup(O, D, s)` raises `NotImplementedError` because it needs the per-market `mix_flag` slice; internal callers use `_compute_markup_with_flag(..., mix_flag_t)`. This keeps the base class's 3-arg signature clean but means MixCournotBertrand doesn't quite fit the polymorphic protocol. Step 8's `solve/` split is a natural time to clean this up (possibly via a `per_market_context` dict argument to `_compute_markup`).

2. **CustomConductModel on the upstream tier of Vertical is not supported.** The `from_model_formulation` reverse adapter raises `NotImplementedError` if it sees this. Can lift if a user needs it; not in scope for v0.4.

3. **ConductModel config fields live on the base class and are inherited by all subclasses.** Even `PerfectCompetition` has an `ownership` attribute that isn't used by its math. This is slightly inelegant but keeps the type hierarchy uniform and lets the base validate shared config once. Considered (and rejected) a mixin pattern that splits math from config.

4. **ModelFormulation's DeprecationWarning is fired via a class-level flag**, not via `warnings.simplefilter('once', ...)` in the package `__init__`. Rationale: modifying global warnings filter state is poor library behavior; the class-level flag achieves once-per-session without side effects.

---

## Step 5 design evolution (honest summary)

Step 5's design went through **four iterations** during this session, each prompted by pushback from Chris:

1. **Started with single-tier** (`Bertrand(ownership_downstream='firm_ids', model_upstream=Monopoly(), ...)`): minimal user-facing change, kwarg names preserved. I defended this as the lowest-friction migration.

2. **Chris: "what about Vertical() instead of Bertrand()?"** — pivoted to the `Vertical(downstream=..., upstream=...)` wrapper. Correctly identified that vertical is a distinct analytic structure and shared config (tax, vi) lives on the combined model, not on the downstream tier. Accepted.

3. **Chris: "why are we changing what users are used to?"** — stopped and articulated the rationale. Reassessed that most of the motivation (polymorphism, AI navigability, dispatch deduplication) could be satisfied *without* changing the user API. Proposed three options:
   - **Option 1:** internal-only refactor, keep ModelFormulation as user API.
   - **Option 2:** thin syntactic sugar (`Bertrand` subclasses `ModelFormulation`).
   - **Option 3:** full redesign with deprecation ramp.

4. **Chris: "can we do Option 3 but with a ramp?"** — landed the four-pillar deprecation ramp: both APIs coexist, legacy translates to new internally, once-per-session warning, migration guide. This is what shipped.

5. **Mid-implementation pivot (5b'):** initial 5b had ConductModel convert forward to ModelFormulation at Problem init. When Chris pushed back ("why is ModelFormulation the internal canonical?") I realized the cleaner end-state has ConductModel canonical and ModelFormulation as the input alias. 5b' reversed the adapter direction.

Calibration note: each pivot improved the design. The recurring pattern is that I default to conservative / minimum-change proposals and Chris's pushback surfaces the cleaner alternative. Two cycles of this in step 4 (tax-factor fix, per-nest-rho routing), two more in step 5 (Vertical wrapper, ConductModel canonical). Worth internalizing for step 6+.

---

## Outstanding items for Chris

1. **DMSS yogurt golden file (Step 0d).** Still blocked on Lorenzo's data + pinned TRV/F/MCS. Scaffold at `tests/replication/test_dmss_yogurt.py` unchanged.

2. **Coauthor memo.** Updated this session (see `MEMO_coauthor_updates.md`). Worth sending to Lorenzo and Marco now that step 5 provides the clean public API.

3. **v0.4 CHANGELOG drafting.** When step 25 rolls around, the capability fixes in step 4 (gamma gradient for demand_params, tax-factor on gradient_markups) plus the ModelFormulation deprecation in step 5 all need user-facing notes.

4. **`v0.3.3-stable` tag placement.** Unchanged since session 3.

---

## Uncertainty flags for next Claude

1. **Same calibration concern from session 4 recurs in step 5 (noted above).** When my instinct is "minimum change," that's often the moment to stop and ask "what's the cleanest end-state?" — Chris's pushback has improved every design this session. Worth leading with that framing.

2. **`MixCournotBertrand._compute_markup` raising is awkward.** If step 8's `solve/` restructuring doesn't clean this up naturally, there's a design debt here. Options: (a) add `per_market_context` dict to the base `_compute_markup` signature; (b) accept the sig mismatch and document it; (c) split `MixCournotBertrand` into two phases (build per-market operators, then apply).

3. **`ConductModel`'s config fields live on all subclasses.** `PerfectCompetition.ownership` exists but is unused by the math. Mixin pattern was considered but would complicate the class hierarchy. Current design is uniform at the cost of some unused fields per instance. Low-impact but worth noting.

4. **`CustomConductModel` on the upstream of Vertical.** Not supported; `from_model_formulation` raises. If a user needs this, extend `from_model_formulation` first, then Models.__new__'s upstream branch, then add a test.

5. **`test_models_step_0_parity.py` duplicates DGP construction.** The `first_stage_fixture` there is a near-copy of the one in `tests/test_snapshots.py`. A tests/conftest.py-level fixture could deduplicate. Low priority.

---

## Files changed this session

**Added (7 files):**
- `pyRVtest/models/base.py`
- `pyRVtest/models/_adapter.py`
- `tests/test_models.py`
- `tests/test_models_integration.py`
- `tests/test_model_formulation_deprecation.py`
- `tests/test_models_step_0_parity.py`
- `docs/migrating_to_v0.4.rst`

**Modified:**
- `pyRVtest/__init__.py` — 9 new names in `__all__` for the class-based API
- `pyRVtest/formulation.py` — ModelFormulation DeprecationWarning (5c)
- `pyRVtest/models/__init__.py` — populated re-exports
- `pyRVtest/models/standard.py`, `mixed.py`, `collusion.py`, `custom.py`, `vertical.py` — populated from placeholders
- `pyRVtest/problem.py` — `Problem.__init__` accepts `models=`; `Models.__new__` rewritten to consume ConductModel/Vertical; helper `_apply_conduct_to_fields`
- `tests/test_import_roundtrip.py` — track populated `__all__` in `pyRVtest/models/*`
- `docs/index.rst` — migration guide added to toctree

---

## Key files for the next Claude to read first

1. `.claude/plans/v0.4-refactor.md` — the refactor plan. Step 5's row is now complete; step 6 is next.
2. This handover.
3. `pyRVtest/models/__init__.py` + the 6 populated class files — the canonical class-based API.
4. `pyRVtest/problem.py` — `_construct_demand_backend` (from step 4) and the `Problem.__init__` models= handling (from 5b/5b') are the two critical intake points.
5. `docs/migrating_to_v0.4.rst` — the user-facing migration guide.
6. `MEMO_coauthor_updates.md` — current state for coauthors.

---

## Next work item — Step 6: `Dict_K` fix + rho-canonical naming

Per `.claude/plans/v0.4-refactor.md` §5:

> Step 6 | Fix `Dict_K` (class → instance). Add `rho`-canonical naming in `demand_params` dict with `sigma` deprecation alias. | All step-0 tests + new test for two concurrent `Problem` instances

Two independent fixes bundled into one step:

1. **`Dict_K` class → instance.** Look for `class Container` (or similar) in `pyRVtest/problem.py`: `Dict_K` and `Dict_Z_formulation` are currently class-level attributes, so two concurrent `Problem` instances share them. That's a long-standing bug; see §4.6 of the plan. Fix: make them instance attributes set in `__init__`. Add a test that constructs two Problems with different L and checks their K values don't bleed.

2. **`rho`-canonical in `demand_params`.** The pyblp nested-logit parameter is `rho`, but `demand_params` currently uses `sigma`. Inside pyRVtest the AFSSZ L-level formulation also uses sigma. PyBLP users are confused by the mismatch. Fix: accept both `demand_params['rho']` (new canonical) and `demand_params['sigma']` (deprecated alias with once-per-session warning, same pattern as 5c). Internal code keeps using `self._sigma` on the backend classes (they're AFSSZ L-level; sigma is correct there).

Step 6 is small — probably 2 sub-commits (6a Dict_K, 6b rho). Low risk. No snapshot changes expected.

---

**Push status:** all commits on `origin/v0.4-refactor` at `ebb8779`. No uncommitted changes.
