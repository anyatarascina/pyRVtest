# Handover: pyRVtest CClean-fixes — Complete

**Date:** 2026-04-14
**Session:** ~5 hours, review + planning + implementation
**Branch:** `CClean-fixes` off CClean tip `bc62371`
**Commits:** `d1ab378` (batch 1), `16b655a` (batch 2)
**Status:** All 16 planned items complete. Branch ready for review.

---

## What was done

### Theory loading
Read three papers via pymupdf:
- DMSS 2024 "Testing Firm Conduct" (QE) — core RV test, F-diagnostic, MCS
- DMQSS 2025 "Conduct and Scale Economies" — non-constant cost. **Appendix B page 48** has modified psi and F formulas
- DMSQW 2026 "Learning Firm Conduct" — pass-through framework. Future feature only.

### Code review
Full audit of all source files combined with Lorenzo's four memos (`~/Downloads/MEMO_pyRVtest_*.md`). Found additional bugs beyond Lorenzo's list (partial_xi_theta undefined, psi vectorization). Caught false positives from our own agents (symbol ordering, uninitialized arrays). Verified K normalization cancellation numerically.

### Implementation (all 16 items)

**Commit 1 (`d1ab378`) — 8 correctness fixes + 2 perf + tests:**
- 1.1: K_effective for F-diagnostic (sigma, F denom, CV lookup all use K-1; cancels numerically but matches paper notation)
- 1.2: gradient zeroed without FEs (de-gated assignment from absorb_cost_ids conditional)
- 1.4: raise when demand_adjustment + endogenous_cost_component both True
- 2.2: demand_results state restoration via try/finally
- 2.3: ModelFormulation.__reduce__ (all 14 fields round-trip)
- 2.4: partial_xi_theta undefined (except Exception → except LinAlgError + re-raise)
- 2.5: Problem.__init__ output de-indented
- 2.7: validation for model_downstream='other' without custom_model_specification
- 3.1: batch Gram matrix replaces O(C*S*M^2) roll loop (~900x speedup, verified identical < 2e-14)
- 3.2: inv→solve for Bertrand/monopoly markups
- 11 tests (5 formulation pickle/validation, 6 clustering equivalence)

**Commit 2 (`16b655a`) — 3 correctness fixes + cleanup:**
- 1.3: variance psi first-stage correction per Appendix B. Precomputes z^r, q^e, Z_prec, lambda_q; adds vectorized per-model correction (verified against direct computation < 2e-13)
- 2.1: IV correction FE absorption (_compute_iv_correction absorbs before 2SLS; _compute_instrument_results absorbs w/endog_hat before joint residualization; raw endog_col used for mc_correction)
- 2.6: tau_list_per_instrument exposed on Progress/ProblemResults
- Removed dead self._max_J

---

## Key numerical insights

1. **K normalization cancels:** `1/K` in sigma trace and `2*K` in F denominator cancel. F value is invariant. Only the CV lookup row affects output. Changed all three to K_effective to match paper.

2. **Psi correction vectorization:** The Appendix B formula `W^{3/4}(W^+ Z u lambda' + lambda u' Z W^+)W^{3/4} g_m` contracts to two terms: `(M_corr @ u_i) * (lambda . W34 g_m)` + `(W34 lambda) * (u_i . v)` where `v = Z_prec W^+ W34 g_m`. Verified to < 2e-13 against direct per-observation computation.

3. **Clustering equivalence:** Cameron-Gelbach-Miller cluster-sum `(1/N) S'S` where `S[c,:] = sum within cluster c` is algebraically identical to the roll-based sum of cross-products. Benchmarked 909x speedup at N=50K.

---

## What's NOT done (future work documented in plan)

**Next release — outward polish:**
- `to_latex()`, `to_markdown()`, `summary_df()` on ProblemResults
- README rewrite, CITATION.cff, Zenodo DOI
- Error message fuzzy matching, tutorial series, diagnostic plots, `show_versions()`

**Next release — expanded tests:**
- Property tests, golden-file tests pinning published results, benchmarks, GitHub Actions CI

**Later — backend decoupling:**
- `DemandBackend` protocol, PyBLPBackend wrapper, Grumps/FRAC backends

**Later — new features:**
- Pass-through matrix computation (Dearing et al. 2026)
- `Dict_K` class→instance attribute fix
- `options.finite_differences_epsilon` dynamic update

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

## Next steps for this branch

1. **Push** `CClean-fixes` to origin for coauthor review
2. **Integration test** against the monte carlo example (`docs/notebooks/monte_carlo_example.py`) or the auto tariffs data if accessible
3. **Review with Lorenzo/Marco** — the psi correction (1.3) and FE absorption (2.1) are the items most likely to need discussion
4. **Merge** CClean-fixes → CClean → main once reviewed
