# Handover: pyRVtest CClean-fixes — Complete

**Date:** 2026-04-14
**Branch:** `CClean-fixes`, pushed to origin
**Tests:** 62 fast + 4 slow = 66 total, all passing
**Status:** Bug fixes, performance, demand_params feature all complete and validated.

---

## Session summary

In one session: reviewed the CClean branch, implemented 12 correctness fixes, a 900x clustering speedup, the demand_params feature (bypass PyBLP for logit/nested logit), analytical demand adjustment, and a comprehensive test suite. Validated against hand computation and PyBLP comparison.

---

## What shipped

### Bug fixes (12 correctness + 2 performance)
See `MEMO_pyRVtest_CClean_fixes_2026-04-14.md` in repo root for full details.

### demand_params feature
Users pass `demand_params={'alpha': -2.0, 'sigma': [0.4]}` — no PyBLP needed.

**New module `demand_jacobian.py`:**
- General L-level nested logit Jacobian (closed form, Berry 1994)
- Analytical Jacobian derivative d(D)/d(sigma_l)
- Analytical Hessian for vertical models (finite-diff on closed-form Jacobian)
- Nesting column order inference with hierarchy validation

**All model types supported:**

| Model | Markups | d/d(alpha) | d/d(sigma) |
|-------|---------|------------|------------|
| Bertrand | analytical | analytical | analytical |
| Cournot | analytical | analytical | analytical |
| Monopoly | analytical | analytical | analytical |
| Perfect comp | zero | zero | zero |
| Mixed | analytical | analytical | analytical |
| Vertical | analytical (Hessian) | finite-diff | finite-diff |
| Custom | user function | finite-diff | finite-diff |

**Demand adjustment:** fully analytical for standard models (d(xi)/d(theta) exact, no BLP contraction). Vertical and custom use finite differences on the closed-form Jacobian/Hessian.

### Tests (62 fast + 4 slow)

- 17 algebra tests (base, clustering, FEs, scale economies)
- 6 clustering equivalence
- 5 formulation (pickle, validation)
- 20 demand_params (15 Jacobian unit + 5 PyBLP comparison)
- 14 demand_params integration (all models, nested logit, vertical, mixed, multiple IVs, demand adj, clustering, custom)
- 4 size/power MC (500 replications)

**PyBLP comparison test:** Same data, both paths, markups/g/Q/TRV/F match to machine precision (< 5e-16).

---

## Remaining test gaps (edge cases, low risk)

These are untested combinations. The core paths that compose them are individually validated. Failure is unlikely but not impossible.

1. **Multi-level nested logit (L > 1) end-to-end:** Jacobian and derivative verified for L=2 via finite-difference comparison, but no end-to-end solve() test with a 2-level DGP.

2. **endogenous_cost_component + demand_params:** These are independent code paths (endogenous cost handles the IV correction for scale economies, demand_params handles the Jacobian). Never tested together. May work or may have an interaction bug.

3. **Multiple instrument sets + demand adjustment + demand_params:** Each tested separately but not in the same solve() call.

4. **Cost-side FEs + demand_params:** FE absorption is tested with user_supplied_markups. demand_params computes markups via the Jacobian. The FE absorption should apply identically downstream, but never verified.

5. **Nested logit + vertical + demand_adjustment:** Each pair tested but not the triple.

---

## Key file locations

| What | Where |
|------|-------|
| Branch | `origin/CClean-fixes` |
| Coauthor memo (bug fixes) | `pyrvtest/MEMO_pyRVtest_CClean_fixes_2026-04-14.md` |
| Jacobian module | `pyRVtest/demand_jacobian.py` |
| Tests | `pyrvtest/tests/` (6 test files) |
| Plan | `~/.claude/plans/snug-seeking-zephyr.md` |
| Project memory | `~/.claude/projects/.../memory/project_pyrvtest.md` |

---

## Next steps

### Immediate (scalable-testing-markups project)

See also `scalable-testing-markups/CLAUDE.md` for project context. The AFSSZ paper (Atalay, Frost, Sorensen, Sullivan, Zhu, JPE 2025) estimates nested logit demand in 70 product markets × 13 years = 910 market-years, recovers markups under Bertrand, and documents markup trends. The scalable testing project asks: is Bertrand actually the right model?

1. **Deploy to scalable-testing-markups:** Write the wrapper that reads AFSSZ demand estimates (alpha, sigma per market-year) and runs pyRVtest across 910 market-years. This goes in `scalable-testing-markups/code/conduct/`.

2. **Results aggregation:** Build a helper that extracts TRV, F, MCS from ProblemResults into a flat dict, enabling `pd.DataFrame([extract(r) for r in results])` across all market-years.

3. **Instrument construction:** The AFSSZ Stata pipeline builds BLP instruments for demand estimation. Need to construct testing instruments (rival characteristics, differentiation IVs) in the same pipeline and pass them to pyRVtest.

4. **Remaining edge case tests** (items 1-5 above) if time permits.

5. **Models to test.** Bertrand vs what? Candidates for supermarket CPG:
   - Perfect competition (zero markups — lower bound)
   - Monopoly / joint profit max (upper bound)
   - Cournot (if manufacturers set production quantities)
   - Partial collusion (Miller-Weinberg profit weight models, via kappa_specification)
   - Different vertical models (zero wholesale margin, linear pricing, zero retail margin)
   - Non-constant cost / scale economies (via endogenous_cost_component)

6. **Batch execution.** 910 market-years at ~0.1s each = ~90 seconds total. Trivially parallelizable with multiprocessing (8 cores → ~12 seconds). A simple loop is fine for a first pass.

7. **Input format bridge.** The AFSSZ Stata pipeline produces demand estimates (alpha, sigma per market-year) and product data (shares, prices, firm_ids, nesting_ids, cost shifters). Need a Python script that reads the Stata output, constructs the pyRVtest `product_data` DataFrame and `demand_params` dict per market-year, and calls `Problem.solve()`. This goes in `scalable-testing-markups/code/conduct/`.

8. **Instrument construction for testing.** The AFSSZ paper uses BLP instruments for demand estimation but doesn't build testing instruments. For the RV test, need instruments excluded from cost that are correlated with markup differences. From Dearing et al. (2026), the relevant instruments depend on which models are being tested:
   - Rival cost shifters → target ratios of off-diagonal to diagonal pass-through
   - Product characteristics → target the full pass-through matrix
   - BLP-style differentiation instruments → number of rival products, characteristic distances
   These can be constructed from the existing AFSSZ data (product characteristics and cost shifters are in the dataset).

9. **Results aggregation.** With 910 market-years, need a `results_to_row()` helper that extracts TRV, F, MCS p-values, gamma (if testing scale economies) into a flat dict. Then `pd.DataFrame([results_to_row(r) for r in all_results])` gives one row per market-year with columns for all test statistics. Key summary questions:
   - How many markets reject Bertrand?
   - How do F-statistics vary across markets?
   - Do results differ by product category (food vs household goods vs beverages)?
   - Do results change over time (2006 vs 2018)?
   - Where Bertrand is rejected, which model wins?

10. **Pass-through matrix display (Dearing et al.).** For markets where the test is inconclusive, computing the model-implied pass-through matrices would help diagnose why instruments lack power and guide instrument selection. This was discussed as a future pyRVtest feature.

### Architectural refactoring (discuss with coauthors)

**Ours (identified during implementation):**

1. **DemandBackend protocol.** The demand_params path is bolted onto Problem with `if self.demand_params is not None` branches in 5+ places. A `DemandBackend` abstraction (with `LogitBackend` and `PyBLPBackend` implementations) would encapsulate: compute Jacobian, compute Hessian, compute markup gradient, compute demand moments. Problem would call `self.backend.jacobian()` without caring which path. This cleans up issues 2-3 below as well.

2. **Duplicate demand adjustment methods.** `_compute_analytical_demand_adjustment` and `_compute_demand_adjustment_gradient` return the same 5-6 objects but are completely separate methods. The backend protocol would unify them behind a single interface.

3. **`_product_data_raw` on Problem.** Stored only because the Products recarray doesn't keep arbitrary columns that the analytical demand adjustment needs (x_columns, demand instruments). With a backend, demand-side data would live on the backend object.

4. **Analytical Hessian.** Currently finite-differences the Jacobian w.r.t. shares. The nested logit Hessian has a closed form. An analytical version would be faster for large markets and eliminate the ~1e-7 approximation.

5. **Test reorganization.** 6 test files with overlapping scope. Cleaner structure: `test_engine.py` (core RV algebra), `test_demand_params.py` (analytical path), `test_integration.py` (end-to-end combinations).

**Lorenzo's (from memo 3 — backend compatibility):**

6. **`DemandBackend` protocol** — same as our item 1. Lorenzo's design includes:
   - `PyBLPBackend`: wraps PyBLP results, isolates private attribute access (`_sigma`, `_pi`, `_beta`, `_rho`, `_delta`)
   - `LogitBackend` / `NestedLogitBackend`: what we built as demand_params, but as a proper class
   - `GrumpsBackend`, `FRACBackend`: Julia interop for alternative demand estimators
   - `UserSuppliedBackend`: for pre-computed Jacobians (what user_supplied_markups partially does now)

7. **Strip PyBLP private attribute mutation.** The demand adjustment gradient currently directly mutates `demand_results._sigma`, `._pi`, `._beta`, `._rho`, `._delta`. A PyBLPBackend wrapper would isolate this, protecting against PyBLP API changes.

8. **`Dict_K` class attribute.** `Dict_K` and `Dict_Z_formulation` are class-level dicts shared across all Problem instances. If two Problems are constructed in one session, they silently share state. Move to instance attributes.

**Lorenzo's (from memo 2 — outward polish):**

9. **`to_latex()`, `to_markdown()`, `summary_df()`** on ProblemResults — needed for the scalable project (910 results → one table).
10. **README rewrite** — the current README doesn't document demand_params, endogenous_cost_component, or any CClean features.
11. **CITATION.cff + Zenodo DOI** — for proper citation in papers.
12. **Error message fuzzy matching** — `difflib.get_close_matches` for column name typos.
13. **Tutorial notebook series** — replacing the old testing_firm_conduct.ipynb.
14. **Diagnostic plots** — F-statistic visualization, MCS p-value bars.
15. **`show_versions()`** — for bug reports and reproducibility.

**Lorenzo's (from memo 4 — expanded testing):**

16. **Property tests** — determinism, row-permutation invariance, FWL check.
17. **Golden-file tests** — pin published results (DMSS yogurt, DMQS autos, Miller-Weinberg beer).
18. **Performance benchmarks** — `pytest-benchmark` to catch regressions.
19. **GitHub Actions CI** — automated testing on push.
20. **Demand adjustment algebra test** — hand-compute markup gradient w.r.t. demand parameters (hard; may be better as property test).
