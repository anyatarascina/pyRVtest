# Handover: pyRVtest CClean Review and Fix Plan

**Date:** 2026-04-14
**Session:** ~3 hours, deep review + planning
**Branch:** `CClean-fixes` created off CClean tip `bc62371`

---

## What was done

### 1. Theory loading
Read three papers cover-to-cover via pymupdf:
- **DMSS 2024** "Testing Firm Conduct" (QE) — core RV test, F-diagnostic, MCS, variance estimator (eq. 6-8, 17-18)
- **DMQSS 2025** "Conduct and Scale Economies" — non-constant cost extension. **Appendix B page 48** has the modified psi formula and F-statistic with d_z-1. This is the key reference for fixes 1.1 and 1.3.
- **DMSQW 2026** "Learning Firm Conduct" — pass-through framework. Conclusion: primarily theoretical guidance, no immediate code changes needed. Pass-through matrix display is a future feature.

### 2. Code review
Full audit of all source files in `pyRVtest/`. Combined with Lorenzo's four memos (`~/Downloads/MEMO_pyRVtest_*.md`):
- `MEMO_pyRVtest_CClean_review_2026-04-14.md` — correctness review, found 12 issues
- `MEMO_pyRVtest_outward_polish_2026-04-14.md` — adoption/UX improvements
- `MEMO_pyRVtest_backend_compatibility_2026-04-14.md` — DemandBackend protocol design
- `MEMO_pyRVtest_test_strategy_2026-04-14.md` — test suite design

### 3. Key numerical verification
- **Clustering equivalence:** Verified that cluster-sum Gram matrix produces identical results to the roll loop (max diff < 2e-14). Benchmarked 909x speedup at N=50K.
- **K normalization cancellation:** Proved that `1/K` in sigma trace and `2*K` in F denominator cancel, so F value is invariant to K vs K-1. Only the critical value lookup row actually changes output. Plan changes all three to K_effective to match paper notation.

### 4. Agent false positives caught
Three findings from our explore agents were verified as incorrect:
- Symbol ordering logic (claimed backwards) — actually correct
- Uninitialized symbol arrays — `else` clause at line 1183 initializes all cells
- NaN column mismatch between ownership/response matrices — both trimmed consistently

---

## What was decided

### Approved plan
At `~/.claude/plans/snug-seeking-zephyr.md`. Four tiers, 16 items.

**Tier 1 — release blockers:**
1. K_effective for F-stat and critical values (change all 3 sites to match paper)
2. Gradient zeroed without cost FEs (de-gate the assignment from the FE conditional)
3. Variance psi first-stage correction (Appendix B page 48 formula)
4. Raise when demand_adjustment + endogenous_cost_component both True (short-term guard)

**Tier 2 — correctness:** IV correction FE absorption, demand_results state restoration, __reduce__ pickling, partial_xi_theta undefined, output indentation, per-instrument tau_list, 'other' model validation.

**Tier 3 — performance:** Batch Gram matrix for clustering (909x), inv to solve in markups.

**Tier 4 — docs:** Docstrings, dead code.

### Branching
`CClean-fixes` off CClean. Marco may be working on CClean concurrently. Merge back when reviewed.

### Scope
Correctness + performance + tests for each bug. Outward polish, expanded test suite, backend decoupling are future work (documented in plan).

---

## What's next

1. **Set up test scaffolding:** `tests/conftest.py`, pytest config, tiny synthetic fixture that can exercise the pipeline without real PyBLP data
2. **Fix 1.1 (K_effective):** Write a test that checks the critical value lookup uses K-1 when endogenous_cost_component is set. Test fails on current code. Apply the 3-line fix. Test passes.
3. **Fix 1.2 (gradient without FEs):** Write a test that checks gradient_markups is nonzero after demand adjustment without cost FEs. Apply Lorenzo's patch from §4.3.
4. **Continue through tiers 1-4.**

---

## Key file locations

| What | Where |
|------|-------|
| Repo | `~/Dropbox/Economics/claude/pyrvtest/` |
| Plan | `~/.claude/plans/snug-seeking-zephyr.md` |
| Lorenzo's memos | `~/Downloads/MEMO_pyRVtest_*.md` (4 files) |
| DMQSS paper (Appendix B) | `~/Downloads/Testing_with_Non_constant_Cost (16).pdf` |
| DMSS paper | `~/Downloads/Quantitative Economics - 2024 - Duarte - Testing firm conduct (1).pdf` |
| DMSQW paper | `~/Downloads/Falsifying_Models_and_Tax_Instruments (14).pdf` |
| Project memory | `~/.claude/projects/-Users-christophersullivan-Dropbox-Economics-claude/memory/project_pyrvtest.md` |

---

## Open questions

1. **Sigma trace `1/K` vs `1/K_effective`:** We proved these cancel in F, so both conventions give the same number. Decision: change to K_effective to match paper. But worth double-checking against the monte carlo example output after implementing.
2. **Fix 1.3 (psi correction):** The Appendix B formula involves `z^r` (instruments residualized on w only), `q^e` (first-stage residual), `Z_prec` (precision matrix), `lambda_q` (projection coefficient). The exact numpy translation needs care — verify against a hand-computed 2-product example.
3. **Test fixtures:** Need a tiny synthetic dataset that exercises the full pipeline. The monte carlo example in `docs/notebooks/` uses PyBLP simulation, which is heavy. Consider a mock/pre-computed approach for fast unit tests.
