**MEMO**

**To:** Lorenzo Magnolfi, Marco Duarte
**From:** Christopher Sullivan
**Date:** April 14, 2026
**Re:** CClean-fixes branch: what changed, what we verified, what remains

---

## 1. Summary

I reviewed the CClean branch end-to-end alongside Lorenzo's four memos from the same date. I then implemented all correctness fixes from Lorenzo's memo 1 (plus additional items he identified as deferred), the ~900x clustering performance improvement Lorenzo flagged in his memo 1 section 4.12, and the variance estimator correction from Appendix B of our non-constant-cost paper. The work is on branch `CClean-fixes`, seven commits ahead of CClean tip `bc62371`, pushed to origin. 28 tests pass, covering:

- **Algebra validation:** hand-computed g, Q, tau, TRV, F match pyRVtest to < 1e-8 for three configurations (base, clustering, scale economies with endogenous cost component)
- **Size/power validation:** 500-replication Monte Carlo confirms correct size (~5%), high power for Bertrand vs perfect competition, correct rejection direction, and strong instrument detection
- **Integration test:** full 13-model monte carlo example runs end-to-end with demand_adjustment=True and endogenous_cost_component

Notably, the `demand_adjustment` x `endogenous_cost_component` interaction that Lorenzo flagged as deferred (memo 1 section 4.2) is now fully implemented. Also relaxed the overly restrictive validation that blocked clustering_adjustment with user-supplied markups (clustering is a variance computation with no demand dependency).

All items have been verified; no open discussion items remain. An additional FE absorption bug was found and fixed during verification (see section 4).

---

## 2. What changed

### 2.1 Tier 1: release blockers for endogenous_cost_component

**1.1 F-diagnostic K_effective (Lorenzo section 4.1).** Added `K_effective = K - 1` when `endog_hat is not None`. Changed sigma trace normalization (`1/K` to `1/K_effective`), F denominator (`2*K` to `2*K_effective`), and critical value lookup (`K` to `K_effective`). The sigma and F-denom changes cancel numerically (F value is invariant to the convention), but we changed both to match the Appendix B notation so that anyone reading the code against the paper sees `d_z - 1` consistently. The critical value lookup is the change that actually affects output.

**1.2 Gradient zeroed without cost FEs (Lorenzo section 4.3).** Pre-existing bug. In `_compute_first_difference_markups`, the residualization and assignment to `gradient_markups` were gated inside `if self._absorb_cost_ids is not None`. Without FEs the entire block was skipped, silently making `demand_adjustment=True` a no-op. Fixed by making FE absorption conditional while always performing the assignment.

**1.3 Variance psi first-stage correction (Appendix B of our paper).** When `endogenous_cost_component` is set, the influence function psi includes a correction for estimation of the linear predictor q-tilde. Implemented the formula from page 48 of the current draft:

```
psi += (1/2) W^{3/4} (W^+ Z_prec u_i lambda'_q + lambda_q u'_i Z_prec W^+) W^{3/4} g_m
```

Precomputes `z^r` (instruments residualized on w only), `q^e` (first-stage residual), `Z_prec`, and `lambda_q` once before the model loop. The vectorized implementation was verified against a direct per-observation computation to machine precision (max diff < 2e-13).

**This is the item most likely to need scrutiny.** The vectorization contracts the rank-2 outer product into two terms: `(M_corr @ u_i) * (lambda_q . W34 g_m)` and `(W34 lambda_q) * (u_i . v)` where `v = Z_prec W^+ W34 g_m`. I am confident in the algebra but a second pair of eyes on the contraction would be valuable.

**1.4 demand_adjustment x endogenous_cost_component interaction (Lorenzo section 4.2).** Fully implemented. Lorenzo identified three fix paths: (a) raise, (b) re-run IV inside perturbation loop, (c) analytical d gamma / d theta. We implemented (b). At each demand parameter perturbation, `_record_gamma_gradient` calls `_compute_iv_correction` at both perturbed markups to finite-difference `gamma_m` alongside the markups. In `_compute_instrument_results`, the G_k matrix (used in the demand-adjustment variance correction) is augmented with:

```
G_k -= (1/N) Z' @ (d_gamma/d_theta * endog_resid)
```

so that the psi influence function correctly captures both channels through which theta affects omega: the markup channel and the gamma channel. Cost: one QR decomposition per (model, instrument set, perturbation direction), which is microseconds compared to the BLP contraction at each perturbation. The monte carlo example now runs as originally written with `demand_adjustment=True` and `endogenous_cost_component='shares'`.

Integration tested: TRV=-0.817 with demand adjustment vs TRV=-0.831 without (wider standard errors when accounting for demand estimation uncertainty, as expected). F=30.8 vs 32.4 (same pattern). Both show `***` `^^^` (strong instruments).

### 2.2 Tier 2: correctness fixes

**2.1 IV correction FE absorption (Lorenzo section 4.4).** `_compute_iv_correction` now absorbs `exog_w`, `endog_col`, and `Z_inst` via `_absorb_cost_ids` before running 2SLS, so `gamma_m` is estimated within-group. The raw (un-absorbed) `endog_col` is preserved separately for `mc_correction`, since the correction is applied to raw marginal cost. Similarly, `_compute_instrument_results` now absorbs `w_for_ols` and `endog_hat` before joint residualization of `Z_orthogonal`, ensuring all inputs are in the same basis for valid Frisch-Waugh-Lovell.

**This is the second item that needs discussion.** Lorenzo's memo identifies the bug clearly but the fix has a subtlety: the mc_correction must use raw endog_col (not absorbed) because it's added to raw marginal_cost_base in `solve()`. The absorbed endog_col is only for estimating gamma_m. I want to confirm this is the right approach.

**2.2 demand_results state restoration (Lorenzo section 4.5).** Wrapped the entire perturbation block in `try/finally` that saves `sigma`, `pi`, `beta`, `rho`, `delta` before any perturbation and restores all of them on any exit path.

**2.3 ModelFormulation.__reduce__ (Lorenzo section 4.9).** Rewrote to match the `__init__` signature exactly. Was duplicating `_custom_model_specification` and missing `unit_tax`, `advalorem_tax`, `advalorem_payer`, `cost_scaling`, `mix_flag`.

**2.4 partial_xi_theta undefined after exception.** The bare `except Exception` at the matrix inversion in `_compute_demand_adjustment_gradient` was swallowing all errors and leaving `partial_xi_theta` undefined, causing a `NameError` at the next line. Changed to `except np.linalg.LinAlgError` with re-raise.

**2.5 Problem.__init__ output indentation (Lorenzo section 4.8).** De-indented the three `output()` calls so they run unconditionally, not only when collinearity checks are enabled.

**2.6 Per-instrument tau_list (Lorenzo section 4.6).** Added `tau_list_per_instrument` to `Progress` and `ProblemResults`. The existing `taus` attribute remains instrument set 0 for backward compatibility.

**2.7 Validation for model_downstream='other'.** Added check that `custom_model_specification` is provided when `model_downstream='other'` (unless `user_supplied_markups` is set).

### 2.3 Tier 3: performance

**3.1 Clustering variance: ~900x speedup.** Replaced the O(C * S * M^2) `np.roll` loop in `_compute_variance_covariance` with a batch Gram matrix using the Cameron-Gelbach-Miller cluster-sum formula:

```python
cluster_sums[m] = np.add.at(zeros, cluster_idx, psi[m])   # O(M*N)
gram = (1/N) * cs_flat.T @ cs_flat                         # one BLAS call
```

This is algebraically identical to the old computation (verified to < 2e-14 on synthetic data with N=50K, M=5, K=5, C=250). Benchmarked at 909x faster on the clustering path. Applied to both the psi Gram (RV denominator) and the phi Gram (F-statistic). The old `_compute_variance_covariance` method is removed; replaced by `_compute_block_gram` and `_extract_block`.

**3.2 inv to solve in markup computation (Lorenzo section 2.3).** Replaced `inv(A) @ b` with `np.linalg.solve(A, b)` for Bertrand and monopoly markups in `evaluate_first_order_conditions`. Cournot still uses `inv` because the Hadamard product `Omega * D^{-1}` genuinely requires the full inverse.

### 2.4 Tier 4: cleanup

Removed dead `self._max_J` (computed but never referenced).

---

## 3. What we verified

**Integration test on the monte carlo DGP.** Bertrand vs Cournot, `endogenous_cost_component='shares'`, `clustering_adjustment=True`. N=10,118.

| Quantity | Without demand adj | With demand adj |
|----------|-------------------|----------------|
| True gamma | 0.500 | 0.500 |
| Estimated gamma (Bertrand) | 0.365 | 0.365 |
| Estimated gamma (Cournot) | 0.652 | 0.652 |
| TRV | -0.831 | -0.817 |
| F | 32.4 (`***` `^^^`) | 30.8 (`***` `^^^`) |
| Solve time | 0.19s | 0.89s |

The wrong model (Cournot) compensates with a higher gamma, which is exactly the behavior predicted by the theory. The test correctly identifies Bertrand as the better-fitting model but does not reject at 5% (expected: 10K obs, one instrument set, mild endogeneity with correlation=0). With demand adjustment, TRV is smaller in magnitude and F is lower, reflecting wider standard errors from accounting for demand estimation uncertainty. Both configurations show strong instruments.

**Test suite (28 tests, all passing):**
- 6 base algebra tests: hand-computed g, Q, tau, TRV, F on 2-firm 20-market logit match pyRVtest to < 1e-8
- 4 clustering algebra tests: hand-computed clustered TRV and F match; GMM moments unchanged by clustering; clustered variance differs from unclustered
- 3 scale economies algebra tests: hand-computed gamma, TRV, F with endogenous cost component (2SLS, effective instruments, psi correction) match pyRVtest
- 6 clustering equivalence tests: batch Gram matrix matches old roll-based computation to < 1e-12
- 5 formulation tests: pickle round-trip, validation
- 4 size/power Monte Carlo tests (500 replications, T=500 markets each): size < 10%, power > 50%, correct rejection direction > 90%, strong instruments > 80%

**Numerical verification of psi correction.** The vectorized implementation matches a direct per-observation loop to < 2e-13 on random (K=5, N=100) data with non-identity weight/precision matrices.

**K normalization cancellation proof.** Showed algebraically and numerically that `1/K` in sigma and `2*K` in F cancel: the F value is invariant to the convention. Only the critical value lookup row matters.

---

## 4. Items verified (previously flagged for discussion)

All three items originally flagged for discussion have been independently verified. An additional bug was found and fixed during the verification process.

**4.1 Psi correction vectorization (fix 1.3) — VERIFIED.** Tested the contraction on four structured cases beyond random data: identity matrices, symmetric tridiagonal, lambda parallel to g (degenerate direction), and K=1 scalar. All match the direct per-observation computation to machine precision. The algebra is correct.

**4.2 FE absorption in mc_correction (fix 2.1) — VERIFIED.** The concern was whether applying mc_correction to raw data (after estimating gamma on absorbed data) is correct. Proved algebraically: since FE absorption is a linear projection (M_FE) and gamma is a scalar, `M_FE(mc - gamma * endog) = M_FE(mc) - gamma * M_FE(endog)`. The order of absorption vs correction does not matter. Verified numerically to machine precision.

**4.3 Basis consistency in G_k correction (fix 1.4) — VERIFIED, BUG FOUND AND FIXED.** Tracing the data flow revealed that `_compute_first_difference_markups` was absorbing diff_markups (correct) but residualizing on raw w_for_ols (incorrect). This is the same FWL inconsistency we fixed in `_compute_instrument_results` (fix 2.1), but we had missed it in this function. Fixed in commit `d401a3e`. Added an analytical test with cost-side firm FEs (`TestAlgebraWithFE`) that exercises this path and would have caught the bug. The test verifies TRV, F, and g match hand computation with FE absorption.

---

## 5. What remains (not in this branch)

### Near-term (next release)

**Outward polish (Lorenzo memo 2):**
- `to_latex()`, `to_markdown()`, `summary_df()` on ProblemResults
- README rewrite, CITATION.cff, Zenodo DOI
- Error message fuzzy matching
- Tutorial notebook series
- Diagnostic plots
- `show_versions()`
- Published-data replication notebook

**Expanded test suite (Lorenzo memo 4):**
- Property tests (determinism, row-permutation invariance, Frisch-Waugh-Lovell check)
- Golden-file tests pinning published results (DMSS auto, DMQS non-constant-cost, Miller-Weinberg beer)
- Performance benchmarks with `pytest-benchmark`
- GitHub Actions CI
- Demand adjustment algebra test: requires hand-computing markup gradient w.r.t. demand parameters; may be better as a property test (demand-adjusted variance wider than unadjusted)

### Later

**Backend decoupling (Lorenzo memo 3):**
- `DemandBackend` protocol abstracting PyBLP dependency
- `PyBLPBackend` wrapper to isolate private attribute mutation
- Grumps/FRAC Julia backends
- `UserSuppliedBackend` for pre-computed Jacobians

**New features:**
- Pass-through matrix computation and display (connects to our Dearing et al. paper)
- Pass-through matrix computation and display (connects to our Dearing et al. paper)

---

## 6. How to review

```bash
git fetch origin
git checkout CClean-fixes
git diff CClean..CClean-fixes --stat    # files changed
git log CClean..CClean-fixes --oneline  # 7 commits
pytest tests/ -v                         # 28 tests, all should pass (~56s including MC)
```

The ten commits are structured as:
1. `d1ab378` — mechanical fixes (1.1, 1.2, 2.2-2.5, 2.7, 3.1, 3.2) plus test scaffolding
2. `16b655a` — psi correction, IV FE absorption, tau_list, dead code
3. `9544fcd` — demand_adjustment x endogenous_cost_component interaction (1.4)
4. `51be3ba` — MC results update and this memo
5. `2d48ee8` — analytical base algebra test
6. `8838804` — clustering and scale economy algebra tests, relaxed clustering validation
7. `bb4acab` — Monte Carlo size/power test (500 replications)
8. `4b6d46d` — memo update
9. `d401a3e` — FE absorption fix in _compute_first_difference_markups (found during verification)
10. `78e8bfb` — cost-side FE algebra test (catches the bug fixed in commit 9)

All discussion items from the original memo have been verified (section 4). Commit 7 is the strongest evidence the package works correctly. Commit 10 is the test that exercises the FE path.
