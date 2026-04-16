**MEMO — ONGOING UPDATES**

**To:** Lorenzo Magnolfi, Marco Duarte
**From:** Christopher Sullivan
**Re:** pyRVtest development — cumulative changes since CClean-fixes
**Last updated:** 2026-04-16

This is a running memo of pyRVtest changes that affect methodology, results, or coauthor-visible API. I will keep adding to the top as things change. Read the "Status right now" block for the current state. Each dated section below documents a specific change and its blast radius.

---

## Status right now (2026-04-16 evening)

**Branch:** `CClean-fixes` at `d25f5de` on origin.
**Tests:** 71 pass (62 pre-existing + 9 new first-stage-correction tests).
**Headline:** three first-stage-correction bugs were found and fixed today. Two were in yesterday's `demand_params` feature (not released outside CClean-fixes). One was pre-existing in the PyBLP path and may have affected prior results.

**What coauthors need to know right now:**

1. **Prior pyRVtest results with `demand_adjustment=True` on the PyBLP path are affected.** The weight matrix used in DMSS Appendix C equation (77)'s Λ was wrong (`updated_W` instead of `W`). Default behavior now matches DMSS. To reproduce prior output: set `pyRVtest.options.demand_adjustment_weight = 'updated_W'` before `.solve()`.

2. **Yesterday's `demand_params` feature had two additional bugs** (sign error in d(markup)/dα; missing concentration adjustment). Both fixed. Anyone who ran demand_params with `demand_adjustment=True` on the pre-today CClean-fixes branch should rerun.

3. **A v0.4 refactor design is complete** (`.claude/plans/v0.4-refactor.md` on the branch). 892-line document covering backend protocol, class-based ConductModel API, labor-market hooks, custom demand integration, snapshot regression suite, and a 25-step migration plan. Not yet started; awaiting coauthor read.

---

## 2026-04-16 — First-stage correction bug fixes (commit b3b08a3)

### What changed

Three bugs in the DMSS (2024) Appendix C equation (77) first-stage correction, all fixed in a single commit on CClean-fixes.

**Bug 1 — sign error in analytical d(markup)/dα** (`pyRVtest/problem.py:1045`).
```python
# Before:  gradient_markups[m][idx, 0] = mu_t / alpha      # WRONG
# After:   gradient_markups[m][idx, 0] = -mu_t / alpha     # CORRECT
```
Bertrand markup Δ = −(O·D^T)^{-1}s. With D = α·N(σ,s), markup is homogeneous of degree −1 in D. By Euler (or chain rule): d(markup)/dα = −markup/α. Sign check: α<0 and markup>0 gives −markup/α > 0, matching the expectation that less-elastic demand (α closer to zero) raises markups. The code computed +markup/α, which is exactly negated. Caught by the new `test_gradient_markups_match` test.

**Bug 2 — missing concentration adjustment in analytic path** (`_compute_analytical_demand_adjustment`).
The PyBLP path has always applied a 2SLS residualization of ∂ξ/∂θ on X_D (the exogenous columns of X1) to profile β out of the asymptotic expansion, matching DMSS's convention that β is concentrated and θ is the non-concentrated parameter vector. The analytic path (shipped yesterday in CClean-fixes) skipped this step. Ported the formula from the PyBLP path:

```python
if X_D.shape[1] > 0:
    XtZW = X_D.T @ Z_D @ W_D
    M_xx = XtZW @ Z_D.T @ X_D
    projection_coeffs = np.linalg.inv(M_xx) @ (XtZW @ Z_D.T @ dxi_dtheta)
    partial_xi_theta = dxi_dtheta - X_D @ projection_coeffs
else:
    partial_xi_theta = dxi_dtheta
H = (1 / N) * Z_D.T @ partial_xi_theta
```

For pure logit, β is concentrated via 2SLS (no nonlinear θ to search over). For nested logit, β is concentrated at fixed ρ while ρ is profiled. In both cases the concentration residualization is what DMSS Appendix C requires.

**Bug 3 — wrong weight matrix in PyBLP path (pre-existing, not new in CClean-fixes)** (`_compute_demand_adjustment_gradient`).
```python
# Before:  WD = self.demand_results.updated_W    # WRONG
# After:   WD = self.demand_results.W            # CORRECT (when option='W')
```
`r.updated_W` is pyblp's "next-step efficient weight," computed from the current-step residuals for use in a hypothetical 2-step update. `r.W` is the weight *actually used* in the GMM step(s). DMSS eq (77) specifies Λ with the weight used to estimate θ̂, which is `r.W`. For `method='1s'` on pure logit, this is the 2SLS weight (Z'Z/N)^{−1}; for `method='2s'`, it's the efficient weight after step 2. Using `updated_W` was wrong in both cases.

### Backwards-compat flag

Added `pyRVtest.options.demand_adjustment_weight`:
- `'W'` (default, v0.3.3+): DMSS-correct.
- `'updated_W'`: reproduces pre-v0.3.3 PyBLP-path output for replication and validation.

Set to `'updated_W'` before `.solve()` to reproduce prior-version output.

Documented in `options.py` with rationale.

### Why these weren't caught earlier

The existing `TestDemandParamsVsPyBLP` in `tests/test_demand_params.py` validates markups/g/Q/TRV/F agreement between paths with `demand_adjustment=False` (lines 244-260). It never exercises the first-stage correction. The integration tests that DO use `demand_adjustment=True` check only that TRV is finite and differs from the no-adjustment case — not that it's numerically correct. So both paths could be producing wrong-but-plausible numbers indefinitely.

The new `tests/test_first_stage_correction.py` (9 tests) closes this gap. On the same DGP with matched specifications, the two paths now agree to machine precision on TRV, F, g, H, h, H'W_D, and the markup gradient. The option-toggle test confirms the `updated_W` choice produces meaningfully different output on over-identified DGPs.

### DMSS / DMQSS references

- DMSS (2024, QE) Supplementary Material **Appendix C, equation (77):** defines $\tilde\psi_{m,i} = \hat\psi_{m,i} - \hat W^{1/2} G_m \Lambda [h(\hat\theta^D) - h_i(\hat\theta^D)]$ with $G_m = -\frac{1}{n}\hat z' \nabla_\theta \hat\Delta_m$ and $\Lambda = (H'W^D H)^{-1} H'W^D$. The gradient $\nabla_\theta \hat\Delta_m$ is d(markup)/dθ; the weight $W^D$ is the GMM weight used in demand estimation.
- DMQSS (2026) **Appendix B, p. 52** (non-constant cost paper): gives the extended ψ with the q̃/γ correction for endogenous cost components. The existing pyRVtest code implements this correctly; the CClean-fixes 1.4 item fixed the interaction with demand_adjustment.

### Empirical verification

On a logit DGP with over-identified demand (4 moments, 3 parameters), matched specifications on both paths:

| Quantity | Agreement |
|---|---|
| markups | 1e-8 |
| g (GMM moments) | 1e-8 |
| H (demand moment Jacobian) | 9e-16 |
| h_i (per-obs contributions) | 9e-16 |
| H'W_D | 1e-14 |
| **TRV** with `demand_adjustment=True` | **r1 = −0.16875467, r2 = −0.16875468 (diff 1e-8)** |
| F with `demand_adjustment=True` | 1e-8 |

### Action items for coauthors

1. **Check whether any prior published result used `demand_adjustment=True` on the PyBLP path.** If so, those TRV/F values were slightly off. The magnitude depends on how different `W` and `updated_W` were in that DGP. For just-identified demand (K_z = K_β) the difference is zero (Λ is weight-invariant when H is invertible). For over-identified demand the difference is generically non-zero.

2. **Review the DMSS Appendix C mapping to code** in `MEMO_coauthor_updates.md` §"2026-04-16" above. If the math is right, no further action. If anyone disagrees with the mapping, the fixes can be reverted via `git revert b3b08a3` without loss.

3. **v0.4 design doc review.** See `.claude/plans/v0.4-refactor.md`. Design is complete and approved by Chris; coauthor input would be valuable before Step 0 begins. Especially the two open methodological questions: (a) Dearing notation for `constant_markup` / `rule_of_thumb` / `cost_plus` model classes, and (b) `market_side='labor'` validation strictness for the labor-market-conduct project.

---

## Earlier changes (see prior memos)

- **2026-04-14:** CClean-fixes branch with 12 correctness fixes + ~900x clustering speedup + analytical demand adjustment for logit/nested logit. See `MEMO_pyRVtest_CClean_fixes_2026-04-14.md` in repo root.

---

## How to use this memo

This file lives at the repo root and is committed. Each coauthor update adds a new dated section at the top with:

1. **What changed** (code-level detail)
2. **Why** (reference to paper, theory, or prior issue)
3. **Blast radius** (prior results affected, how to reproduce)
4. **Action items** (things coauthors should verify or decide)

To send to coauthors: copy this file into an email or attach the PDF. Or send the GitHub link. The content is self-contained.

Future updates (expected as v0.4 lands):
- Class-based `ConductModel` API
- Backend refactor (`DemandBackend` protocol + `PyBLPBackend` + `LogitBackend`)
- Results aggregation layer (`to_dataframe`, `summary_df`, `to_latex`, `PanelResults`)
- Stand-alone `build_passthrough` diagnostic
- Labor-side hooks (`market_side='labor'`)
- `UserSuppliedBackend` for Almagro-Sood-style custom demand
- σ → ρ notation alignment with PyBLP
- Analytical nested-logit Hessian
- `Dict_K` bug fix (class-level → instance-level)
