# Monte Carlo Example — Session Handover

**Branch:** `CClean-Marco`  
**Date:** 2026-04-20  
**Task:** Update `docs/notebooks/monte_carlo_example.py` to v0.4 API and demonstrate new features.

## What has been done

The file has been fully rewritten. Here is a summary of every change:

### 1. v0.4 class-based API (all ModelFormulation deprecated patterns removed)
- `ModelFormulation(model_downstream='bertrand', ...)` → `pyRVtest.Bertrand(ownership='firm_ids')`
- `ModelFormulation(model_downstream='cournot', ...)` → `pyRVtest.Cournot(ownership='firm_ids')`
- `ModelFormulation(model_downstream='monopoly')` → `pyRVtest.Monopoly()`
- `ModelFormulation(model_downstream='perfect_competition')` → `pyRVtest.PerfectCompetition()`
- kappa models → `pyRVtest.PartialCollusion(...)` / `Bertrand(..., kappa_specification=...)`
- Vertical models → `pyRVtest.Vertical(downstream=..., upstream=..., vertical_integration=...)`
- Mixed → `pyRVtest.MixCournotBertrand(mix_flag='mix_flag', ownership='firm_ids')`
- Custom → `pyRVtest.CustomConductModel(markup_fn=..., ownership=..., name=...)`
- `Problem(..., model_formulations=...)` → `Problem(..., models=...)`

### 2. pyRVtest.instruments helpers replace raw pyblp calls
- `from pyRVtest.instruments import blp_instruments, differentiation_ivs`
- BLP instruments: `blp_instruments(data, columns=['z', 'x1', 'x2'])`
- Differentiation IVs: `differentiation_ivs(data, char)` in a loop
- Columns prefixed `test_iv0_*` and `test_iv1_*` respectively

### 3. Dearing et al. (2026) simple-markup models (v0.4 step 12)
- Added as a separate list `dearing_models`: `Keystone()`, `RuleOfThumb(phi=3.0)`, `ConstantMarkup(markup=0.3)`
- Shown in markup comparison table (full `model_library = model_core + dearing_models`)
- Tested against Bertrand in a dedicated `dearing_problem` (separate from main test)
- **Important:** `RuleOfThumb(phi=1.5)` was avoided because it is numerically identical
  to `PerfectCompetition(cost_scaling='cost_scaling_col')` (both produce `cost_scaling=0.5`);
  using phi=3.0 avoids the collinearity.

### 4. UserSuppliedMarkups sanity check
- Pre-computed Bertrand markups stored in `data['mkup_bertrand']`
- Tested via `usm_problem` with `demand_adjustment=False` (required when using UserSuppliedMarkups)
- **Important:** UserSuppliedMarkups cannot be mixed with `demand_adjustment=True` — the
  pipeline needs a demand system to perturb for the first-stage correction.

### 5. Analytical (demand_params) code path
- `analytical_problem` built with `demand_params={'beta': [...], 'sigma': [...]}` instead of `demand_results=pyblp_results`
- Tests the analytical first-stage correction equivalence (same as `tests/test_first_stage_correction.py`)

### 6. Passthrough diagnostics (Dearing Remark 4)
- `vertical_problem` built with only the two Vertical models
- `vertical_results.passthrough_comparison(metric='offdiag_frobenius')` called
- Grouped by model pair labels and printed

### 7. Labor-side example (v0.4 step 14, experimental)
- Synthetic wage panel: 2 firms × 50 markets
- Uses `UserSuppliedMarkups` for Monopsony and PerfectCompetition markdowns
- `Problem(..., market_side='labor')` — the v0.4 labor API

## What still needs to be verified

**The script has NOT been successfully run to completion yet.** It was interrupted before the
final test run was confirmed to pass. The last state before interruption:

- Everything up to and including the markup comparison table was printing correctly.
- The main testing problem (`model_core`, 13 models) was the one that produced the SVD
  non-convergence error originally (before the split into `model_core` vs full library).
  After the restructuring, it was NOT re-run to completion.

**To verify from the other computer, run:**
```bash
conda run -n pyRVenv python docs/notebooks/monte_carlo_example.py 2>&1 | tail -80
```

## Model split rationale

The script uses two model sets:
- `model_core` (13 models): main `testing_problem`; identical to the v0.3 original set
- `dearing_models` (3 models): separate `dearing_problem`; Keystone, RuleOfThumb(3.0), ConstantMarkup(0.3)
- `model_library = model_core + dearing_models`: markup comparison table only

This split avoids numerical collinearity in the MCS that arises when 16+ models are tested
simultaneously. The original 13-model set was numerically stable; adding Dearing models
directly triggered SVD non-convergence in `test_engine.py:248` via near-collinear moments.

## Files changed
- `docs/notebooks/monte_carlo_example.py` — fully rewritten (see current state)
