**MEMO — ONGOING UPDATES**

**To:** Lorenzo Magnolfi, Marco Duarte
**From:** Christopher Sullivan
**Re:** pyRVtest development — cumulative changes since CClean-fixes
**Last updated:** 2026-04-16

This is a running memo of pyRVtest changes that affect methodology, results, or coauthor-visible API. I will keep adding to the top as things change. Read the "Status right now" block for the current state. Each dated section below documents a specific change and its blast radius.

---

## Status right now (2026-04-16 late evening, post step 1)

**Branches:**
- `CClean-fixes` at `e921649` on origin — Step 0 protection (frozen for now)
- `v0.4-refactor` at `7e20ccb` on origin — branched from `CClean-fixes`, step 1 module skeleton landed

**Tag:** `v0.3.3-stable` annotated, pushed at `47b4457` on `CClean-fixes` (baseline anchor for nuclear revert).
**Tests (on v0.4-refactor):** 121 pass + 3 skipped (DMSS yogurt placeholders pending Lorenzo).
**Headline:** v0.4 Step 0 protection + step 1 module skeleton both landed. The refactor now has its safety net and its target directory layout. Step 2 (extract `Products` → products.py with type hints) is next. Lorenzo's 0d input is still the only Step 0 item outstanding.

**What coauthors need to know right now:**

1. **Prior pyRVtest results with `demand_adjustment=True` on the PyBLP path are affected.** The weight matrix used in DMSS Appendix C equation (77)'s Λ was wrong (`updated_W` instead of `W`). Default behavior now matches DMSS. To reproduce prior output: set `pyRVtest.options.demand_adjustment_weight = 'updated_W'` before `.solve()`.

2. **The `demand_params` feature had two additional bugs** (sign error in d(markup)/dα; missing concentration adjustment). Both fixed in `b3b08a3`. Anyone who ran demand_params with `demand_adjustment=True` before `b3b08a3` should rerun.

3. **v0.4 refactor design + adversarial review both on the branch.**
   - `.claude/plans/v0.4-refactor.md` — updated design doc (~980 lines). Backend protocol, class-based ConductModel API, labor hooks, custom demand integration, 25-step migration plan with Step 0 baseline protection, snapshot-update decision rule, rollback-trigger criteria, release-blocking status for the AFSSZ dogfood step.
   - `.claude/plans/review-2026-04-16.md` — adversarial review of the design. Nine sections: readiness, sequencing, gaps, scope honest reassessment. Each finding has a proposed fix. All approved fixes are in the updated design doc.

4. **Step 0 of the refactor has landed** (except the DMSS yogurt golden file, which needs Lorenzo's input):
   - 9 first-stage-correction equivalence tests (already in `b3b08a3`)
   - 6 snapshot regression tests at `atol=1e-10` on the main DGPs
   - 11 property-based tests covering all currently-implemented conduct models (bertrand, cournot, monopoly, perfect_comp, mix, vertical, kappa-weighted profit weights, custom 'other') with hand-computed ground truth
   - Rollback procedure + trigger criteria
   - `v0.3.3-stable` tag as the nuclear-revert anchor

5. **What we need from Lorenzo** (to complete Step 0d):
   - **Data.** Where the DMSS yogurt product_data lives (path / loader / script).
   - **Specification.** Which table/column from the DMSS paper to pin — demand side (logit / nested logit / BLP), instruments, cost side, FEs, conduct pairs, adjustment flags.
   - **Expected values.** Pinned TRV, F, MCS-p from the paper to 4-5 significant figures.
   - **Tolerance.** Default `rtol=1e-4, atol=1e-6`; tighten if the paper reports more digits.
   Scaffold is ready at `tests/replication/test_dmss_yogurt.py` with a `NEEDED FROM LORENZO` block listing these items. Populating the constants and un-skipping the three tests is sufficient to complete 0d.

6. **Next step on the refactor:** v0.4 migration step 1 — create the module skeleton (empty `backends/`, `models/`, `instruments/`, `solve/`, `results/` subpackages; `__init__.py` re-exports preserve the current public API). No behavior change. Will land on a new `v0.4-refactor` branch off `CClean-fixes`.

---

## 2026-04-16 (late evening, post step 1) — v0.4 migration step 1 landed

After the Step 0 protection was pushed, we branched `v0.4-refactor` off `CClean-fixes` at `e921649` and landed migration step 1 in a single commit (`7e20ccb`).

### What step 1 delivered

- 5 new subpackages matching the plan's §4.1 target layout: `backends/` (+ `backends/labor/`), `models/`, `instruments/`, `solve/`, `results/`.
- 22 empty skeleton `.py` modules, each with `__all__ = []` and a one-paragraph docstring naming the migration step that will populate it. Future populating-step commits just flip `__all__ = []` → `__all__ = [new_names]` within the existing files.
- `pyRVtest/results.py` verbatim-moved to `pyRVtest/results/__init__.py`. Resolves the file-vs-directory naming collision that the plan's `results/` subpackage needs. No code changed; imports from `pyRVtest.results` (internal and external) still work.
- `pyRVtest/__init__.py` updated to re-export the new subpackages as namespaces (`pyRVtest.backends`, `pyRVtest.models`, etc.) while preserving every v0.3 public-API symbol.
- `tests/test_import_roundtrip.py` — 29 tests (public API importability, paranoid v0.3 public-API preservation, parameterized step-1 skeleton coverage, ProblemResults/Progress accessibility after the subpackage move).

### Verification

Full test suite on `v0.4-refactor`: **121 passed + 3 skipped in 2:41**. All 6 Step 0 snapshots still match at `atol=1e-10` — confirming that the results.py → results/ move and the new subpackage re-exports changed nothing behaviorally.

### One small user-visible compat caveat

If any coauthor has an old `.pkl` file produced by v0.3.x containing a `ProblemResults`, unpickling under v0.4 **might** trip a `ModuleNotFoundError: pyRVtest.results` because the module is now a subpackage. Low risk (the class is still reachable at `pyRVtest.results.ProblemResults` and imports resolve correctly at runtime), but worth flagging here so it can be added to the v0.4 release notes.

### What's next

Step 2 — extract `Products` from `pyRVtest/problem.py` into `pyRVtest/products.py` with type hints and `mypy --strict` clean on that single file. Pure extraction; no behavior change.

---

## 2026-04-16 (late evening) — v0.4 refactor review + Step 0 protection landing

Second session on 2026-04-16. Adversarial review of the v0.4 design doc, then the Step 0 baseline protection committed and pushed to origin. Tag `v0.3.3-stable` created as the nuclear-revert anchor per the rollback procedure.

### What changed on CClean-fixes (5 commits, all on origin)

| Commit | Description |
|---|---|
| `f0840c3` | DOC: incorporate v0.4 review feedback into design doc |
| `1bcda1d` | TEST: v0.4 Step 0f property-based test scaffolding |
| `edcb2e1` | TEST: expand property tests to cover all current conduct models |
| `7268925` | TEST: v0.4 Step 0b snapshot regression suite |
| `47b4457` | TEST: v0.4 Step 0d scaffolding (DMSS yogurt golden file) |

Tag `v0.3.3-stable` pushed to origin at `47b4457`.

### Review findings — major design-doc updates

- **Tag name corrected.** The original plan tagged `v0.3.2-stable` at `45bcc4b` — the pre-bug-fix commit. Would have pointed the nuclear-revert anchor at known-broken math. Renamed to `v0.3.3-stable` at `47b4457` (post-fix).
- **Dearing verification reordered.** Original plan shipped `ConstantMarkup`/`RuleOfThumb`/`CostPlus` in step 5 before verifying the parameterization against Dearing's paper in step 12. Reordered: step 5 ships mechanical classes only (Bertrand, Cournot, Monopoly, PerfectCompetition, MixCournotBertrand, PartialCollusion); simple-markup classes defer to step 12 where their formulas are first pinned against the paper.
- **Step 16 (AFSSZ dogfood) now release-blocking for step 25.** Chris confirmed "one set of code, once it is ready, we will launch it on AFSSZ and scalable labor" — coauthors migrate once.
- **Minimal CI added as step 24.5.** One-config GitHub Actions (pytest + mypy + doctest). Full matrix deferred to v0.5.
- **mypy / property tests / `__all__` moved from late-stage batch to incremental rules** applied at every step.
- **Snapshot-update decision rule formalized.** `<= 1e-12` auto-update / `1e-12 to 1e-7` deliberate-source commit required / `> 1e-7` blocks merge.
- **Rollback triggers formalized.** Single-step-breaks = soft revert / two-step-regression = hard revert / >1% DMSS yogurt drift = nuclear.
- **Labor sign-convention validation** added to §4.5 so upstream sign errors don't produce silent garbage.
- **Scope tiers** (§2.5) listed as fire-sale order. Maximalist scope stays locked per Chris's decision.
- **Coauthor gates stay informal** per Chris.

### Step 0 protection now in place

Six pieces of protection, built out during this session:

1. **`tests/test_snapshots.py`** — 6 snapshot regression tests at `atol=1e-10`:
   - `analytical_base` — user_supplied_markups Bertrand-vs-perfect-competition
   - `analytical_clustering` — with `clustering_adjustment=True`
   - `analytical_base_fe` — with cost absorb=`C(firm_ids)`
   - `analytical_scale` — endogenous_cost_component path
   - `first_stage_pyblp_path` — the code that had Bug 3 (`updated_W` vs `W`)
   - `first_stage_demand_params_path` — the code that had Bugs 1 and 2 (sign + concentration)

   Mismatch detection verified. Snapshot-update decision rule documented in the helper module.

2. **`tests/test_properties.py`** — 11 property-based tests (Hypothesis library):
   - determinism / permutation invariance / seed reproducibility
   - perfect_competition markups exactly zero
   - Bertrand markups computed by pyRVtest match DGP ground truth across alpha
   - dispatch smoke test across all 5 standard conduct models + hand-computed match
   - Cournot / Monopoly markups match hand-computed (Hypothesis × 100)
   - profit-weight `kappa_specification` Bertrand matches hand-computed
   - custom `'other'` model extension point
   - vertical Bertrand-downstream + Monopoly-upstream

   TODO block at the end of the file names the conduct models not yet in pyRVtest (rule_of_thumb, constant_markup, cost_plus, nash_bargaining, monopsony, etc.) — these get property tests when their classes ship in v0.4 steps 5, 12, 14.

3. **`tests/test_first_stage_correction.py`** — 9 equivalence tests (already in `b3b08a3`).

4. **`tests/replication/test_dmss_yogurt.py`** — scaffolding only. Three `@pytest.mark.skip` tests (TRV / F / MCS). Awaiting Lorenzo's input on data + pinned specification + expected values.

5. **Design doc `§10` rollback procedure** — reverified, tag name corrected throughout.

6. **Tag `v0.3.3-stable`** at `47b4457` on origin. Nuclear-revert anchor.

### Action items from this session for coauthors

1. **Lorenzo** — populate the DMSS yogurt golden file. File `tests/replication/test_dmss_yogurt.py` has a `NEEDED FROM LORENZO` block with four items: data location, specification to pin, published expected values, tolerance. Once filled in, the three tests un-skip automatically (remove the `@pytest.mark.skip` decorator on each).

2. **Marco and Lorenzo** — read `.claude/plans/review-2026-04-16.md` for the full review and `.claude/plans/v0.4-refactor.md` for the updated plan. The Decisions Log section §8 lists each decision with rationale. Flag any disagreement before step 1 lands on the `v0.4-refactor` branch.

3. **Both** — check whether any prior PyBLP-path result used `demand_adjustment=True`. If so, rerun with default `options.demand_adjustment_weight='W'`. See next section for the bug-fix details.

### Uncertainty flags I want to name explicitly

1. **Monopoly ≥ Bertrand element-wise** is asserted in the dispatch smoke test; it passes on the single fixed-seed DGP we run but is not theoretically guaranteed for every DGP. If a future step breaks this assertion, weaken to `np.mean(mono) >= np.mean(bert)`.

2. **Labor sign-convention validation** defaults to strict (raises on negative wages). May need a `strict_signs=False` opt-out for users with deviation-from-mean wage-like data. Decided to keep strict default today; revisit at v0.4 step 14.

3. **`v0.3.3-stable` tag** is placed while 0d is still a scaffold. If Lorenzo's inputs change any pyRVtest behavior, we may create `v0.3.4-stable` rather than moving the existing tag.

4. **Hypothesis `max_examples=100`** gives strong coverage but adds ~13s to the test suite. If the suite creeps past 5 minutes during the refactor, split into dev (30 examples) and CI (100) Hypothesis profiles.

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
