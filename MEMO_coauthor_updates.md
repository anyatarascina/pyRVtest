**MEMO — ONGOING UPDATES**

**To:** Lorenzo Magnolfi, Marco Duarte
**From:** Christopher Sullivan
**Re:** pyRVtest development — cumulative changes since CClean-fixes
**Last updated:** 2026-04-20 (rc1 landed + post-rc1 polish + numpy 2 investigation + CI matrix now green)

This is a running memo of pyRVtest changes that affect methodology, results, or coauthor-visible API. I will keep adding to the top as things change. Read the "Status right now" block for the current state. Each dated section below documents a specific change and its blast radius.

---

## Status right now (2026-04-20 — v0.4.0rc1 tagged + post-rc1 polish + CI matrix green)

**Branch:** `v0.4-refactor` at `2aff863` on origin (19 commits past the pre-rc1 anchor).
**Tag:** `v0.4.0rc1` at `cdf4781` on origin (unchanged; post-rc1 commits will land in v0.4.0 final).
**CI:** ✅ **green on both matrix jobs.** First time CI has been green since the numpy 2 / pyblp 1.1+ wheels moved. Matrix runs Ubuntu Python 3.11 with (a) ``numpy<2`` + ``pyblp<1.2`` and (b) ``numpy>=2,<3`` + ``pyblp>=1.2``; each is verified independently end-to-end.
**Tests (numpy 1.26.4 / pyblp 1.1.2, macOS):** **633 passed + 3 skipped.** mypy `--strict` clean across 44 source files.
**Tests (numpy 2.4.4 / pyblp 1.2.0, Python 3.11):** 632 passed + 3 skipped + 1 xfailed (`test_snapshot_analytical_scale`, documented below). Same on Ubuntu numpy<2.

### rc1 release status

v0.4.0rc1 is tagged, pushed, and installable via `pip install git+https://github.com/anyatarascina/pyRVtest@v0.4.0rc1` (not yet on PyPI — publishing is a separate `tox -e release-test` / `release` step, held for v0.4.0 final). The tag captures the point where Lorenzo's P0 findings from 2026-04-18 were addressed. Subsequent commits (137f4e2 onward) are post-rc1 polish that will ride in v0.4.0 final.

### What landed in rc1 + post-rc1 (chronological)

Addressing Lorenzo's 2026-04-18 review items and Marco's follow-up email:

| Commit | Item addressed | Source |
|---|---|---|
| `679dbd9` FIX | `ModelFormulation(user_supplied_markups=...)` crash; added `pyRVtest.UserSuppliedMarkups` first-class class | Lorenzo P0 #1, Marco #2 |
| `44bb83b` REL | Version 0.4.0rc1 + `pyproject.toml` (fixes `pip install -e .` under modern pip) | Lorenzo P0 #3 |
| `cdf4781` FEAT | K>30 critical-value UserWarning; `options.digits` wired; `options.verbose` deprecated | Marco #1, Marco #3, Lorenzo P1 #6 |
| `137f4e2` DOC | CHANGELOG qualifier: analytical Hessian is plain logit + single-scalar-rho nested only; per-nest rho / BLP stay on pyblp finite-diff | Lorenzo Chris item #7 |
| `b2ab59a` DOC | Labor API marked experimental (README, AGENTS.md, agent_guide); AGENTS drift sweep (HHI, test count, v0.6 vs v0.7) | Audit G1 + B4, Lorenzo |
| `d13eb2d` FIX | `Problem(market_side='labor', demand_params=...)` now raises clean `NotImplementedError` (was silent wrong-sign trap); tax precedence documented | Lorenzo methodology, Lorenzo Chris item #5 |
| `22a13a1` FIX | `PanelResults` roster-signature validation; panels with `[Bertrand, PC]` / `[PC, Bertrand]` now raise at construction | Lorenzo P1 #4 / audit B1 |
| `f4b39fc` DOC | "No correctness changes" framing qualified in this memo; known-coefficient salience-opt-out note in migration guide | Lorenzo methodology |
| `e4ff40d` TEST | Share-weighted `Monopoly ≥ Bertrand` fallback pinned as theory-guaranteed alternative to element-wise | Lorenzo methodology |
| `176d691` FEAT | Per-model tax `DeprecationWarning` moved to `ConductModel.__init__` / `Vertical.__init__`; `Bertrand(unit_tax='col')` now warns on construction | Lorenzo P1 #7 |
| `880cc6b` FIX | numpy 2 compatibility — jinja2 in requirements; NaN-rho guard in CV lookup; `analytical_scale` xfailed under numpy ≥ 2.0 | Lorenzo P0 #2 (partial) |
| `de9104b` DOC | `analytical_scale` xfail reason upgraded to full root-cause (see investigation below) | — |
| `c04d555` DOC | Session handover + this memo's prior "Status right now" block | — |
| `72e9894` DOC | Coauthor email draft matching this memo state | — |
| `06a091f` CI | Two-version matrix: ``numpy<2 + pyblp<1.2`` AND ``numpy>=2,<3 + pyblp>=1.2`` on Ubuntu Py 3.11 | (new) |
| `9b78028` FIX | 16 scalar-conversion sites (`float(arr)` → `float(arr.item())`); broaden `analytical_scale` xfail to cover Linux (not just numpy ≥ 2) | (surfaced by matrix) |
| `6f500ca` FIX | 3 more scalar-conversion sites on the `cost_param[-1]` pattern missed by earlier grep | (surfaced by matrix) |
| `2aff863` FIX | numpy 2.4 + Python 3.11: ``_NDArray: TypeAlias = NDArray[Any]`` across 16 modules; mypy.ini disable `var-annotated` globally; one remaining scalar-conversion; doctest in `nested_logit_labor.py` (pre-existing bug); CI diagnostic reverted | (surfaced by matrix) |

Test suite grew from 592+3 to 633+3 on numpy 1 / 632+3+1 xfailed on numpy 2.4 (41 new regression tests across the items plus the numpy 2 tightening fixes).

### CI matrix + compatibility cleanup (the final-mile work)

The original CI was a single Ubuntu + Python 3.11 job with pip resolving whatever numpy/pyblp versions happened to be current. That job had been silently red since the numpy 2 ecosystem moved (circa late 2024); the rc1 commits only addressed what was visible to each of us on our local environments. The two-version matrix (`06a091f`) made the ecosystem axis explicit and surfaced four categories of residual issue:

1. **Dozens of `float(array_with_ndim>0)` sites in tests.** DeprecationWarning on numpy 1; hard `TypeError` on numpy 2. Fixed across 6 test files (19 sites total across three commits).
2. **`_NDArray = NDArray[Any]` no longer treated as a type alias by mypy 1.19 under numpy 2.4 stubs.** 217 errors from a pure type-alias issue. Fixed by changing all 16 module-level declarations to `_NDArray: TypeAlias = NDArray[Any]` with `typing_extensions.TypeAlias` (already a transitive dep).
3. **numpy 2.4 stubs under-infer `np.zeros(...)` return types**, tripping `[var-annotated]` on existing lax-typed modules. Silenced globally in `mypy.ini`.
4. **Pre-existing broken doctest** in `pyRVtest/backends/labor/nested_logit_labor.py` (the raise message had adjacent string literals with no newline, so `.splitlines()[0]` returned the full text). Had been silently failing the CI doctest step since the labor code was added; only visible once pytest passed far enough for doctest to run.

**Net effect:** pyRVtest now passes end-to-end on both the legacy-pinned environment (numpy 1.26 / pyblp 1.1) and the modern environment (numpy 2.4 / pyblp 1.2). The `analytical_scale` F-shift remains as a documented xfail under the broadened condition (non-macOS OR numpy ≥ 2), with root cause fully diagnosed in the investigation section below.

**Cross-LAPACK drift update.** The matrix run revealed that the F-shift described below isn't uniquely a "numpy 2" issue — Ubuntu numpy 1.26 also drifts from the macOS-generated snapshot at the same ~2.5% magnitude. So the real condition is "LAPACK build different from the one snapshots were generated under", which across our current setup means "numpy ≥ 2 OR not macOS." The xfail condition helper `_analytical_scale_expected_to_drift()` (`tests/test_snapshots.py`) encodes this.

### numpy 2 `analytical_scale` F-shift — the investigation

Lorenzo's P0 #2 was the one remaining item after rc1. After extensive instrumentation, the issue is now fully diagnosed. Honest summary for the record:

**What it is.** On the `test_snapshot_analytical_scale` snapshot, `F[0][0][1]` = 1.0316662590067232 on numpy 1.26.4 and 0.9980489511445025 on numpy 2.0.2 — a ~3% shift. All other snapshot fields (markups, TRV, Q, g, marginal_cost on paths that don't hit the same amplifier) remain bit-identical across numpy versions.

**Why.** Three-step chain:

1. **LAPACK build difference.** numpy 2.0 wheels ship a different bundled OpenBLAS / LAPACK build than numpy 1.26 wheels. `np.linalg.qr` dispatches to LAPACK's dgeqrf, and the new build uses different Householder reflection block sizes. Resulting `Q` matrices are equally orthonormal but don't span *exactly* the same column space — they span subspaces within `O(ε × κ)` of each other. Confirmed by bisect: numpy 1.26 + scipy 1.13 is bit-identical; numpy 2.0 + scipy 1.13 is not.
2. **Amplification through near-collinear controls.** The `controls = hstack([w, endog_hat])` matrix in the endogenous-cost path has condition number ~3582 (near-collinearity between `w` and `endog_hat`). That turns the ULP-level LAPACK difference into a ~1% shift in the three `sigma` values driving F.
3. **Catastrophic cancellation in F's denominator.** This specific fixture (Bertrand vs PerfectCompetition with a `log_quantity` IV correction that absorbs the markup difference) produces near-equivalent moment conditions for the two models. The three sigmas are nearly equal (~2.07e-3), so `F_den = σ₀σ₁ - σ₂² ≈ 4e-8` is in the catastrophic-cancellation regime. The ~1% σ shift is amplified to a 3% F shift by the near-degenerate geometry.

**TRV is unaffected** (shift ~9e-15, machine precision) because its structure `(Q[i] - Q[m]) / sqrt(var)` is self-normalizing — numerator and denominator scale together and the LAPACK noise partially cancels. F doesn't have that cancellation.

**What we tried (all failed).** Every algorithmic fix I experimented with left the 3% shift intact:

| Fix | Cross-version F shift | Outcome |
|---|---|---|
| SVD projection instead of QR | 3.2% | No help — SVD also goes through LAPACK |
| Algebraic `F = 2N/K · F_num / denom_sq²` (removes the algebraic F_den cancellation) | 3.3% | No help — the upstream 1% σ shift dominates |
| Sign-canonicalize Q via `diag(R) ≥ 0` | 3.3% | No-op — signs cancel in `Q @ Q.T` |
| Revise the DGP to avoid catastrophic cancellation | best ~0.013% | Reduces 250×, still not 1e-10 |

**Theoretical takeaway, relevant to the paper.** F's denominator is a 2×2 determinant of the moment-condition covariance. It vanishes precisely when the two candidate models produce nearly identical moment conditions given Z — which is the weak-instrument regime the F-stat is designed to diagnose. So **F is inherently numerically sensitive in the regime it's built to flag.** This fixture puts F in that regime on purpose: after the `log_quantity` IV correction absorbs the markup difference, the two models are near-equivalent in moments. The ~3% cross-version shift lands F in the same "weak" star bucket for this fixture (at K=1, ρ≈0.08, both F=1.03 and F=0.998 trivially clear all size thresholds and fall below all power thresholds), so no user-facing conclusion changes here. But for a fixture with F landing near a power CV threshold at high ρ (say `r_50 = 0.3` at K=1, ρ=0.99), the numerical instability could flip a significance star across numpy versions. This is a real user-facing concern about the F-stat *as a diagnostic*, not just about our test fixture.

**Where we're leaning.** Given nothing algorithmic closes the gap, the mature approach is to accept that `atol=1e-10` is a single-numpy-version guarantee and tier the snapshot tolerance by field:

- Linear / pass-through fields (markups from `user_supplied_markups`, prices, shares): keep `atol=1e-10`.
- Self-normalizing fields (TRV, Q, g): `atol=1e-8`.
- Nonlinear-amplification fields (F, σ, ρ): `atol=1e-3`.

With a two-version CI matrix (numpy 1.26 + numpy 2.x), tiered atol, and explicit documentation of the numerical-precision tier. This matches what scipy / pytorch / jax do. Not yet implemented — held for Chris's go-ahead after discussing with the team. Estimated 1-2 hours to land. If we go this route the xfail goes away and the snapshot becomes a real cross-version regression test with honest tolerance.

**Alternative:** keep the xfail. Accurate but fragile — every future numpy release that bumps OpenBLAS would re-trigger investigation. Also doesn't help users in production if their data puts them near the regime.

**Question for Lorenzo / Marco:** the ~3% numerical shift is always going to be there in the near-degenerate regime for any F computation on current numpy/LAPACK, regardless of fixture. The DMSS paper's F-stat is designed for the regime where it's numerically robust (well-separated models, strong instruments); the degenerate regime is exactly where the paper says F should be interpreted cautiously anyway. Is there anything in the paper's text that deserves a brief sentence acknowledging this numerical sensitivity? Or is it already implicit in the "weak-instrument" language?

### Outstanding items (by owner)

**Chris:**

- Resolve the `analytical_scale` F-shift per above. Lean: tiered atol + 2-version CI, 1-2 hours of work.
- Tag `v0.4.0` final after Step 16 (AFSSZ dogfood) lands clean and any API adjustments are absorbed.
- DMQSW provenance check: email Marco / Sal / Waldfogel to confirm whether Table 4 / Table 5 of the current submission were computed before or after `b3b08a3` (the W-vs-updated_W fix). If before, those TRV/F values for demand_adjustment=True + over-identified demand were slightly off.
- Audit post-tag queue: `Problem(demand_backend=...)` public kwarg, `CITATION.cff`, `CONTRIBUTING.md`, replication suite anchors, machine-readable exports. v0.4.0-final or v0.5.

**Marco (handoffs from Lorenzo, still open from 2026-04-18):**

- DMSS yogurt golden file (Step 0d). Scaffold at `tests/replication/test_dmss_yogurt.py`. Needs data path + pinned paper spec + expected TRV/F/MCS values + tolerance.
- Labor sign-convention review on `pyRVtest/models/labor.py`. Pairwise consistency across the three classes is verified; absolute convention against the labor-market-conduct manuscript is not. Flag discrepancies before v0.5 activates `LaborSupplyBackend`.
- `offdiag_frobenius` metric vs Dearing Remark 4 cross-check. Currently `‖P_i − P_j − diag(diag(P_i − P_j))‖_F`. Does it operationalize Remark 4 exactly?
- `Results.summary` investigation (Marco's own item). Only `summary_df` exists; if `.summary()` is the call, it's `AttributeError`. Awaiting Marco's follow-up.

**Lorenzo:**

- Re-run `.claude/breaks/probe_01..07.py` from `tests/memo1-regressions` branch against `v0.4.0rc1`. Confirm carRV crash is fixed.
- (Optional) Extend `probe_05_v03_vs_v04_parity.py` to exercise `endogenous_cost_component + demand_adjustment=True` once DMQSW replication data is accessible.

**Coauthor group decision:**

- DMQSW provenance (under Chris above).
- `v0.4.0` tag timing (post-Step 16 + post-xfail resolution).

---

## Status right now (2026-04-17 end of day — v0.4 effectively feature-complete)

**Branch:** `v0.4-refactor` at `5321a7b` on origin.
**Tests:** **592 passed + 3 skipped** (was 388+3 at the previous "Status right now" block). mypy `--strict` clean across 43 source files. Snapshot tests at `atol=1e-10` bit-identical throughout.
**Steps done:** 24 of 25. Only **step 16** (AFSSZ dogfood on real data) remains — data-blocked by ~1 week. **Step 25** (tag v0.4.0) is awaiting Chris's call.

### Bottom line for coauthors

Library-side v0.4 work is complete. The branch is ready to tag v0.4.0 whenever the call is made. All existing v0.3 scripts run unchanged modulo one-line deprecation warnings. No correctness changes vs. the snapshot suite (every refactor was verified bit-identical at `atol=1e-10`); everything landed is additive or renames with aliases.

### New this session (sessions 7 through end of day, 36 commits)

**Architecture / refactor:**

- **Step 8 — `Problem.solve` split into staged pipeline** (5 sub-commits). `solve()` is now a thin orchestrator that calls stages in `pyRVtest/solve/`: `markups.compute`, `orthogonalize.residualize`, `endogenous_cost.iv_correct`, `demand_adjustment.apply`, `test_engine.compute`. `problem.py` shrunk from 1733 → 1328 lines. Per-stage loggers (`pyRVtest.solve.markups`, etc.).
- **Step 14 — labor-side hooks** (4 sub-commits). `Problem(market_side='labor')` with `Monopsony`, `BertrandWages`, `CournotEmployment` (real sign-flipped math) and `NashBargaining` (v0.5 stub). `LaborSupplyBackend` skeleton wired through the `DemandBackend` protocol; real Jacobian/Hessian math deferred to v0.5.

**Dearing paper integration (step 12):**

The paper `Falsifying_Models_and_Tax_Instruments` (Dearing, Magnolfi, Quint, Sullivan, Waldfogel, 2026) is now folded into the library.

- `RuleOfThumb(phi)` — `p = ϕ·mc` (Example 1). Ergonomic wrapper over the existing `cost_scaling` machinery, which now accepts a numeric scalar in addition to a column name.
- `Keystone()` — `RuleOfThumb(phi=2.0)` shorthand (Escudero 2018).
- `ConstantMarkup(markup)` — `∆ⱼₜ = ζⱼ` (Example 7). Accepts a scalar or a column name; introduces a new additive-markup plumbing path.
- The plan's original `cost_plus` was dropped — mathematically identical to `rule_of_thumb` under a different parameterization.
- Legacy `PerfectCompetition(cost_scaling='lmbda_col')` still works unchanged.

**Tax specification: model-level → Problem-level (resolves OQ 14):**

Per-model `unit_tax` / `advalorem_tax` / `advalorem_payer` on `ConductModel` / `ModelFormulation` / `Vertical` are **deprecated**. New pattern: specify the tax once on `Problem`, opt individual models out via `unit_tax_salient=False` / `advalorem_tax_salient=False` for salience testing.

```python
# NEW recommended
Problem(
    ...,
    unit_tax='tax_col',
    advalorem_tax='vat_col',
    advalorem_payer='consumer',
    models=[
        Bertrand(ownership='firm_id'),                            # all taxes salient
        Bertrand(ownership='firm_id', unit_tax_salient=False),    # salience test
    ],
)
```

Legacy per-model spec still works with a once-per-session `DeprecationWarning`. **Removal scheduled for v0.7** (one release later than other v0.4 deprecations because of prevalence in user code).

**Known-coefficient cost shifters on `Formulation` (also OQ 14):**

```python
cost_formulation = Formulation('0 + w', known_coefficients={'input_price': 0.75, 'union_wage': 1.0})
```

Generalizes the unit-tax idea — shifters with known coefficients (per Dearing et al. 2026) enter `prices_effective` additively without per-model opt-out. Always DGP-level.

**Pass-through diagnostics on `ProblemResults` (resolves OQ 15):**

- `ProblemResults.passthrough_comparison(metric=...)` — DataFrame with pairwise pass-through distances. Three metrics: `frobenius` (default), `offdiag_frobenius` (Dearing Remark 4 distinguishability condition), `max_abs`. Metric recorded on `attrs['metric']`.
- `ProblemResults.passthrough_matrix(model_index, market_id=None)` — thin wrapper over `build_passthrough`.
- **Restriction:** v0.4 only supports Vertical candidate models. Non-Vertical candidates raise `NotImplementedError` with a v0.5 pointer. Per-model closed-form pass-through for Bertrand / Cournot / RuleOfThumb / ConstantMarkup / PerfectCompetition is a clean v0.5 follow-up.

**Results-export methods (steps 9 + 10):**

- `ProblemResults.to_dataframe()`, `.summary_df(alpha=0.05)`, `.to_latex()`, `.to_markdown()`. `summary_df` emits a stable `reject` column and records `alpha` on `DataFrame.attrs['alpha']`.
- `PanelResults` class for multi-problem aggregation (one `ProblemResults` per market-year, subsample, etc.). Mapping-like API plus `to_dataframe`, `rejection_rates`, `summary_df`, `to_latex`, `to_markdown`.

**Developer-experience (steps 18 / 19 / 21):**

- Per-module loggers (`pyRVtest.problem`, `pyRVtest.backends.logit`, etc.) replace the in-house `output()` / `print()` calls. Users silence subsystems via `logging.getLogger("pyRVtest.problem").setLevel(logging.WARNING)`.
- All 120 `raise` sites rewritten to expected/received/fix format. New `pyRVtest/exceptions.py` hierarchy (`PyRVTestError`, `ValidationError`, `BackendError`, and subclasses). Every class multi-inherits from the appropriate built-in so existing `pytest.raises(ValueError, ...)` callers keep working.
- 47 runnable doctests across 28 modules; `pytest --doctest-modules` wired into CI.

### Methodology-relevant notes

- **No changes to existing-model output; new math is additive and has dedicated test coverage.** All pre-v0.4 snapshot fixtures remain bit-identical at `atol=1e-10` — the `Problem.solve` split is pure code relocation, and Problem-level taxes / pass-through diagnostics enter on additive paths with their own tests. Two items DO introduce new math, just not on any pre-v0.4 snapshot path: (1) the labor-side `Monopsony` / `BertrandWages` / `CournotEmployment` classes implement sign-flipped FOCs, covered by `tests/test_labor_mode.py`; (2) `ConstantMarkup` adds an additive-markup branch to `evaluate_first_order_conditions` and `RuleOfThumb` routes numeric-scalar `cost_scaling` through a `perfect_competition` underlying-markup dispatch, both covered by `tests/test_dearing_models.py`. No existing fixture result shifts. *(Lorenzo note 2026-04-18 incorporated; the earlier "no correctness changes" framing was subtly stronger than the evidence supports.)*
- **Labor HHI instrument removed before release.** An agent initially shipped `pyRVtest.instruments.labor.concentration_hhi` by false analogy with product-side concentration. Chris caught in review: HHI in labor markets is endogenous in wages (labor concentration depends on the wage being tested). The helper was deleted before anyone could use it.
- **Labor sign-convention review needed.** `Monopsony`, `BertrandWages`, `CournotEmployment` are implemented with sign-flipped versions of their product-side counterparts (`np.linalg.solve(D, s)` instead of `-np.linalg.solve(D, s)`, etc.). Pairwise consistency is verified (`Bertrand(D)` magnitude ≡ `BertrandWages(-D)` magnitude) but the absolute sign convention against a specific labor-market paper is not — please flag if any of the formulas differ from your preferred convention before we activate the real `LaborSupplyBackend` in v0.5.
- **OQ 15 `offdiag_frobenius` metric** is implemented as `‖P_i − P_j − diag(diag(P_i − P_j))‖_F` (Frobenius norm of the off-diagonal part of the difference). This matches the literal reading of Dearing Remark 4; worth a cross-check against the paper if an external reviewer looks at the diagnostic.

### New public API surface summary

```python
# Dearing simple-markup conduct models
pyRVtest.RuleOfThumb(phi=2.0)
pyRVtest.Keystone()
pyRVtest.ConstantMarkup(markup=0.5)    # or column name

# Labor-side conduct
pyRVtest.Problem(market_side='labor', ...)
pyRVtest.Monopsony(ownership='firm_id')
pyRVtest.BertrandWages(ownership='firm_id')
pyRVtest.CournotEmployment(ownership='firm_id')
# NashBargaining is importable but raises NotImplementedError — v0.5 work

# Problem-level taxes (replaces per-model spec)
pyRVtest.Problem(..., unit_tax='tax_col', advalorem_tax='vat_col', advalorem_payer='consumer')
# Per-model salience opt-out:
pyRVtest.Bertrand(..., unit_tax_salient=False)

# Known-coefficient cost shifters on Formulation
pyRVtest.Formulation('0 + w', known_coefficients={'col': 0.75})

# CustomConductModel under labor mode must opt in
pyRVtest.CustomConductModel(markup_fn=..., side='labor')

# Results exports
results.to_dataframe()
results.summary_df(alpha=0.05)
results.to_latex()
results.to_markdown()
results.passthrough_comparison(metric='offdiag_frobenius')
results.passthrough_matrix(model_index=0, market_id='chicago_2015')

# Panel aggregation
pyRVtest.PanelResults(results={key: problem_results, ...})
panel.rejection_rates(alpha=0.05)
panel.summary_df()
panel.to_dataframe()
```

### Deprecation schedule (for reference)

| Deprecation | Still works through | Removed in |
|-------------|---------------------|------------|
| `ModelFormulation` / `model_formulations=` | v0.5 | v0.6 |
| `demand_params=dict(sigma=...)` | v0.5 | v0.6 |
| `pyRVtest.output.output()` | v0.5 | v0.6 |
| Per-model `unit_tax` / `advalorem_tax` / `advalorem_payer` | v0.6 | **v0.7** |

Each emits a once-per-session `DeprecationWarning` pointing at `docs/migrating_to_v0.4.rst`.

### What we still need from Lorenzo

- **DMSS yogurt data + pinned TRV/F/MCS values** for step 0d (unchanged since session 3). Scaffold at `tests/replication/test_dmss_yogurt.py`.

### What we still need from Marco / Lorenzo

- **Labor sign-convention sanity check** on `Monopsony`, `BertrandWages`, `CournotEmployment` before v0.5 activates the real `LaborSupplyBackend`. If any signs look wrong per your reading of the labor-market-conduct manuscript, flag now while the code is young.

### What's left before v0.4.0 tag

- **Step 16 — AFSSZ dogfood.** Deploy pyRVtest v0.4 on the real 910-market-year panel (in `scalable-testing-markups`), iterate on the API if anything is painful. Data arrives in ~1 week.
- **Step 25 — tag v0.4.0.** Trivial mechanics once step 16 is green. Chris has an open decision: tag now vs. wait for step 16. Current lean: tag now and let step-16 findings produce v0.4.1.

### Chris's open design decisions (defaults shipped, nothing breaking)

1. Labor `__str__` style: banner (shipped) vs per-cell labels.
2. `PerfectCompetition` side-neutrality (shipped: side-neutral).
3. Labor column naming: `employment_share` (shipped) vs raw `employment`.
4. Deprecation runway split (shipped: v0.6 / v0.7).
5. Shared `memory/feedback_python_env.md` rephrasing (coauthor-authored, untouched).
6. Nested `CustomConductModel` inside `Vertical` side-check (rare; not shipped).

None are blockers for v0.4 tag.

### Test-suite growth in this session

From 388 + 3 to 592 + 3 skipped (+204 tests). Notable additions:

- Step 8 (Problem.solve split) — snapshot suite preserved, no new tests by design.
- Step 9 (ProblemResults exports) — 28 tests in `tests/test_results_exports.py`.
- Step 10 (PanelResults) — 27 tests in `tests/test_panel_results.py`.
- Step 18 (logging) — 6 tests in `tests/test_logging.py`.
- Step 19 (error messages) — 10 tests in `tests/test_error_messages.py`.
- Step 21 (doctests) — 47 runnable docstring examples.
- Step 14 (labor) — 28 tests in `tests/test_labor_mode.py`.
- Step 12 (Dearing models) — 32 tests in `tests/test_dearing_models.py`.
- Problem-level taxes — 18 tests in `tests/test_problem_level_taxes.py`.
- Known_coefficients — 16 tests in `tests/test_known_coefficients.py`.
- OQ 15 passthrough_comparison — 17 tests in `tests/test_passthrough_comparison.py`.
- `CustomConductModel` side opt-in — 11 tests added to `tests/test_labor_mode.py`.

Subtracted: 3 tests removed when `concentration_hhi` was dropped.

### Where to read more

- `CHANGELOG.md` — full Unreleased v0.4.0 section with Migration guide and Deprecated section.
- `docs/migrating_to_v0.4.rst` — user-facing migration guide + explicit deprecation timeline.
- `docs/agent_guide.rst` + `AGENTS.md` — architecture tour and class-based-API + labor + Dearing pointers.
- `.claude/plans/v0.4-refactor.md` — design doc with Decisions Log.
- `.claude/handovers/2026-04-17-session7-v0.4-essentially-complete.md` — full session-by-session narrative including the honest retrospective on what went right and wrong.

---

## Status at 2026-04-17 late, post steps 6 / 7 / 11 / 13 / 15 / 17 / 20 / 22 / 23 / 24.5 (superseded by above)

**Branch:** `v0.4-refactor` at `15e1005` on origin (plus session handover commit).
**Tests:** **388 passed + 3 skipped** (was 259 + 3 at end of prior session).
**Steps done:** 16 of 25. Remaining unblocked: 8, 9, 10, 14, 16, 18, 19, 21, 24, 25. Blocked: 0d (Lorenzo's yogurt data), 12 (Dearing paper).

### What coauthors need to know (new this session)

**No new correctness changes** in steps 6 / 7 / 11 / 13 / 15 / 17 / 20 / 22 / 23 / 24.5 beyond what step 4 already covered. All math is consistent with what landed previously; these steps extend the public API, harden tests, and add docs.

**One long-standing bug fixed (step 6a):** `Problem.Dict_K` and `Container.Dict_Z_formulation` were class-level mutable dicts. Two concurrent `Problem` instances in the same Python session would share state and accumulate each other's instrument-set counts. Now instance-level. If you ever constructed multiple Problems in one script and the second one's K0/K1 counts looked wrong, that's why.

**One user-facing deprecation (step 6b):** `demand_params['rho']` is now the canonical nested-logit parameter name (aligns with pyblp). `demand_params['sigma']` still works as a deprecated alias with a once-per-session `DeprecationWarning`. Removal: v0.6. Internal backend classes (`NestedLogitBackend`) keep the `sigma=[...]` constructor kwarg unchanged (AFSSZ L-level math convention).

### New public API you can use now

**Conduct instruments** (step 13): `pyRVtest.instruments.product.rival_sums`, `differentiation_ivs`, `blp_instruments`, and `pyRVtest.instruments.labor.hausman`, `bartik`. Standard BLP / Hausman / Bartik helpers for building Z matrices. All accept DataFrame / structured recarray / dict-like `product_data`. (A labor-side `concentration_hhi` was prototyped and removed before release: labor-market HHI is endogenous in wages -- shares respond to the variable being tested -- so it is not a valid wage instrument.)

**Passthrough inspection** (step 11): `pyRVtest.build_passthrough(problem, model_index, market_id=None)`. Returns the Villas-Boas passthrough matrix for a vertical model, either per-market or as a dict across all markets. Diagnostic for the upstream-markup computation.

**Analytical nested-logit Hessian** (step 7): `NestedLogitBackend.compute_hessian` now uses a closed-form for plain logit and 1-level nested logit (matching the AFSSZ L=1 case pyblp supports). Multi-level falls back to finite-diff. No user-facing code change, but your vertical-integration results with nested logit are now O(eps²) more accurate than before. Validated three ways: Clairaut symmetry (pure math property), cross-check against `pyblp.compute_demand_hessians` at atol=1e-6, and a new snapshot fixture in `tests/snapshots/nested_logit_vertical.json` for future regression protection.

### Documentation shipped

- `docs/migrating_to_v0.4.rst` — full before/after migration guide (step 5d, previous session; now also covers the rho↔sigma rename from step 6b).
- `docs/custom_demand.rst` (step 15) — UserSuppliedBackend worked example for custom demand systems.
- `AGENTS.md` at project root + `docs/agent_guide.rst` (step 23) — architecture tour + deprecation policy + where-to-start-here for new contributors or AI assistants.
- `pyRVtest.show_agent_guide()` — prints the guide to stdout, for quick reference in a REPL.

### Test-suite growth

From 259 + 3 skipped to 388 + 3 skipped. The notable additions:

- 11 Hessian-validation tests (Clairaut symmetry + pyblp cross-check + vertical snapshot).
- 18 instrument-helper unit tests.
- 65 parametrized public-API-pin assertions across 32 modules (zero gaps found — confirms the incremental `__all__` discipline held throughout v0.4).
- 3 property tests (market-partition moment linearity, FWL identity, Bertrand α-homogeneity).
- 6 `build_passthrough` tests.
- 10 `show_agent_guide` / `AGENTS.md`-shape tests.
- Plus mypy-strict, deprecation, state-isolation, rho-alias, custom-demand example tests (~15 more).

### Minimal CI

`.github/workflows/ci.yml` (step 24.5) now runs pytest on every push / PR against `main`, `CClean-fixes`, and `v0.4-refactor`. Single job (Ubuntu + Python 3.11); multi-version matrix deferred to v0.5. Mypy and doctest steps stubbed out pending steps 17 completion and step 21 landing.

### What we still need from Lorenzo

Unchanged since session 3: DMSS yogurt data + pinned TRV/F/MCS values for step 0d. Scaffold at `tests/replication/test_dmss_yogurt.py`.

### Next session work

Per the session-6 handover in `.claude/handovers/2026-04-17-session6-nine-steps.md`: 6 more sessions to complete v0.4. Big remaining: step 8 (split Problem.solve into solve/*.py stages), step 14 (labor-side support — directly unlocks AFSSZ / scalable-labor research), step 16 (AFSSZ dogfood, release-blocking). End-game: step 24 (CHANGELOG) + step 25 (tag v0.4).

---

## Status right now (2026-04-17, post steps 4 + 5)

**Branches:**
- `CClean-fixes` at `e921649` on origin — Step 0 protection (frozen)
- `v0.4-refactor` at `ebb8779` on origin — steps 0 + 1 + 2 + 3 + 4 + 5 all landed

**Tag:** `v0.3.3-stable` still the nuclear-revert anchor at `47b4457`.
**Tests (on v0.4-refactor):** **248 pass + 3 skipped** (DMSS yogurt placeholders pending Lorenzo's data). Full step-0 snapshot suite + first-stage-correction + property tests + 5e parity all green.

**Headline — step 4:** The demand-adjustment unification is complete. `Problem.solve(demand_adjustment=True)` now routes through a single `compute_demand_adjustment` function in `pyRVtest/solve/demand_adjustment.py` generic over a `DemandBackend`. The two inline methods (`_compute_analytical_demand_adjustment`, `_compute_demand_adjustment_gradient`) that had three bugs in b3b08a3 are **deleted**.

**Headline — step 5:** pyRVtest has a class-based conduct API:

```python
pyRVtest.Problem(
    ...,
    models=[
        pyRVtest.Bertrand(ownership='firm_ids'),
        pyRVtest.PerfectCompetition(),
    ],
)
```

Old `ModelFormulation` + `model_formulations=` keeps working with a once-per-session `DeprecationWarning`. Migration guide at `docs/migrating_to_v0.4.rst`. Removal scheduled for v0.6.

### What coauthors need to know

**Correctness changes in step 4 that may affect published / in-flight results:**

1. **Endogenous-cost gamma gradient now computed for `demand_params` users.** Pre-v0.4 `Problem._compute_analytical_demand_adjustment` silently returned `gradient_gamma_per_instrument=None`, disabling the endogenous-cost correction whenever `demand_params` was used (instead of `demand_results`). The PyBLP path computed it correctly. Post-v0.4 both paths compute it. **If you ran `demand_params` + `endogenous_cost_component` + `demand_adjustment=True` under v0.3.x, your TRV/F/MCS were missing the gamma-gradient correction; rerun under v0.4.**

2. **Tax / cost-scaling factor now applied to markup gradient.** The inline analytical path did NOT apply the `advalorem_tax_adj / (1 + cost_scaling)` factor to `gradient_markups`. PyBLP path did. Step 4i applies the factor uniformly in the unified function. **Fixtures without taxes or cost_scaling are unaffected (factor = 1). Users with nontrivial `advalorem_tax` or `cost_scaling` columns: your prior `demand_params` + `demand_adjustment=True` output was missing this factor; rerun under v0.4.**

3. **Plain logit / single-scalar-rho nested logit estimated in pyblp now auto-routes to the analytical backend.** `Problem` detects K2=0 pyblp results and uses `LogitBackend` (plain logit) or `NestedLogitBackend(sigma=[rho])` (single-rho nested) for exact `d(D)/d(theta)` rather than pyblp's finite-diff. Per-nest rho (Cardell-Nevo) and BLP stay on `PyBLPBackend` with finite-diff. **Nested-logit users with a scalar rho: your prior TRV/F/MCS shifted by O(1e-10) to O(1e-8) from the finite-diff → analytical switch. Documented in the updated `first_stage_pyblp_path` snapshot.**

**User-facing API change in step 5:**

4. **Class-based conduct models.** New:

   ```python
   pyRVtest.Bertrand(ownership='firm_ids')
   pyRVtest.Cournot(ownership='firm_ids')
   pyRVtest.Monopoly(ownership='firm_ids')
   pyRVtest.PerfectCompetition()
   pyRVtest.MixCournotBertrand(ownership='firm_ids', mix_flag='mix_col')
   pyRVtest.PartialCollusion(ownership='firm_ids', kappa_specification='collusion_row')
   pyRVtest.CustomConductModel(markup_fn=my_fn, ownership='firm_ids', name='my_model')
   pyRVtest.Vertical(
       downstream=pyRVtest.Bertrand(ownership='firm_ids'),
       upstream=pyRVtest.Monopoly(ownership='manufacturer_ids'),
       vertical_integration='vi_col',
   )
   ```

   Pass to `Problem` via `models=[...]`. Everything else (tax kwargs, `user_supplied_markups`, etc.) keeps the same names.

   Old code still works with a `DeprecationWarning` on first `ModelFormulation()` call. Migration guide covers every case: `docs/migrating_to_v0.4.rst`.

5. **What's not affected:** `Formulation`, `build_markups`, `build_ownership`, `construct_passthrough_matrix`, `evaluate_first_order_conditions`, `read_pickle`, `ProblemResults`, `Problem`, `demand_results`/`demand_params` kwargs, every field on `ProblemResults` (markups, TRV, F, MCS_pvalues, cost_param, taus, ...). Only `ModelFormulation` is deprecated; nothing else.

### What we still need from Lorenzo (Step 0d, unchanged since session 3)

- **Data:** where the DMSS yogurt product_data lives (path / loader / script).
- **Specification:** which table/column from the DMSS paper to pin — demand side (logit / nested logit / BLP), instruments, cost side, FEs, conduct pairs, adjustment flags.
- **Expected values:** pinned TRV, F, MCS-p from the paper to 4-5 significant figures.
- **Tolerance:** default `rtol=1e-4, atol=1e-6`; tighten if the paper reports more digits.

Scaffold at `tests/replication/test_dmss_yogurt.py` unchanged; un-skip the three tests after populating constants.

### Next step

**Step 6:** fix `Dict_K` (class-attribute → instance-attribute bug that makes two concurrent `Problem` instances share state) + add `rho`-canonical naming in `demand_params` with a `sigma` deprecation alias (aligns pyRVtest's nested-logit parameter name with pyblp's). Small step, 2 sub-commits, no expected snapshot changes.

---

## 2026-04-17 — v0.4 migration steps 4 and 5 landed

(Detailed session handover: `.claude/handovers/2026-04-17-session5-step5-complete.md`. Step-4 specifics in `.claude/handovers/2026-04-16-session4-step4-complete.md`.)

### Step 4 — unify demand adjustment behind the backend protocol (11 commits, 4a through 4i with correctness follow-ups)

Delivered:

- Single `compute_demand_adjustment(backend, problem, M, N, markups, advalorem_tax_adj, cost_scaling, marginal_cost_base)` in `pyRVtest/solve/demand_adjustment.py`. Generic over any `SupportsDemandAdjustment` backend.
- `_residualize_on_xd` helper (the DMSS eq. 77 2SLS profile-out) extracted and shared by all backends.
- `LogitBackend` and `NestedLogitBackend` now implement `SupportsDemandAdjustment` (analytical `d(xi)/d(theta)`, `d(D)/d(theta)`).
- `_nested_logit_jacobian_derivative` finally validated against finite-diff — 8 parametrized tests covering 1-level and 2-level nested logit.
- `Problem.__init__` constructs `self._demand_backend` at init. Auto-routing: plain logit and single-scalar-rho nested logit from `demand_results` go to the analytical backend for exact `d(D)/d(theta)`; per-nest rho and BLP stay on `PyBLPBackend` (finite-diff is the right tool there).
- Two silent bugs closed: (a) `gradient_gamma_per_instrument=None` for `demand_params` users (capability fix), (b) tax/cost-scaling factor missing from markup gradient for `demand_params` users.
- Permutation-invariance property test with `demand_adjustment=True`.
- 519 lines of now-dead inline code removed from `problem.py`.

### Step 5 — class-based ConductModel API (6 commits, 5a through 5e with a mid-course 5b')

Delivered:

- `pyRVtest/models/` package populated with: `ConductModel` base, `Bertrand`/`Cournot`/`Monopoly`/`PerfectCompetition` (standard.py), `MixCournotBertrand` (mixed.py), `PartialCollusion` (collusion.py, inherits Bertrand), `CustomConductModel` (custom.py), `Vertical` composer (vertical.py).
- `Problem(models=[...])` kwarg. Mutually exclusive with `model_formulations=`.
- `ConductModel`/`Vertical` is the canonical internal representation; `ModelFormulation` is translated to the class form at `Problem.__init__` via `from_model_formulation`.
- `ModelFormulation.__init__` emits `DeprecationWarning` on first construction per session (class-level flag, no global warnings state mutation).
- Migration guide at `docs/migrating_to_v0.4.rst` with before/after examples for every case.
- 62 new tests: 41 unit (class math vs legacy string dispatch at atol=1e-14), 10 integration (class API vs legacy API on shared DGP), 4 deprecation-behavior, 6 step-0 parity (class API byte-identical on every snapshot scenario), plus 1 module-import tracking.

---



**Branches:**
- `CClean-fixes` at `e921649` on origin — Step 0 protection (frozen)
- `v0.4-refactor` at HEAD on origin — branched from `CClean-fixes`, steps 1 + 2 + 3 landed plus `NestedLogitBackend` split into its own file

**Tag:** `v0.3.3-stable` annotated, pushed at `47b4457` on `CClean-fixes` (nuclear-revert anchor).
**Tests (on v0.4-refactor):** 155 pass + 3 skipped (DMSS yogurt placeholders pending Lorenzo).
**Headline:** v0.4 migration now has the `DemandBackend` protocol and all four backend classes (`PyBLPBackend`, `LogitBackend`, `NestedLogitBackend`, `UserSuppliedBackend`). `_compute_markups` accepts a `demand_backend` parameter and a new equivalence test asserts backend output matches the legacy `pyblp_results` path to `atol=1e-14`. The existing `demand_params` and `pyblp_results` paths remain fully functional — step 3 is **additive-only**. Step 4 (unifying the two demand-adjustment paths, deleting the `demand_jacobian.py` shim, adding `SupportsDemandAdjustment` to the logit backends, threading vertical Hessian through the backend) is the next work item.

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

## 2026-04-16 (very late, post step 3 + split) — v0.4 migration step 3 landed (five sub-commits) + file split

### Step 3: DemandBackend protocol and all four backend classes

Shipped as five sub-commits on `v0.4-refactor`:

| Commit | Sub-step | Contents |
|---|---|---|
| `063cfd5` | 3a | `DemandBackend` + `SupportsDemandAdjustment` Protocols (`backends/base.py`) |
| `f0b61a8` | 3b | `PyBLPBackend` class + 10 unit tests. Wraps a `pyblp.ProblemResults` behind the protocol; encapsulates private-attribute access to `_sigma`, `_pi`, `_beta`, `_rho`, `_delta`. Default weight is `r.W` per DMSS eq. 77 (guards against regression of b3b08a3's Bug 3). |
| `e1ee8cb` | 3c | `LogitBackend` + `NestedLogitBackend` classes + 11 unit tests. Content of `pyRVtest/demand_jacobian.py` moved verbatim to `pyRVtest/backends/logit.py`; the old file is now a thin shim preserving backward compatibility for existing tests and internal callers. |
| `ee5e31b` | 3d | `UserSuppliedBackend` escape hatch + 9 unit tests. Wraps a precomputed Jacobian + optional Hessian callable + optional perturb callback. Implements only the core `DemandBackend` protocol, not `SupportsDemandAdjustment` — users supplying a custom demand system provide adjustment inputs separately. |
| `c87b199` | 3e | Wire `demand_backend` parameter into `_compute_markups`. Additive: legacy `pyblp_results` / `demand_jacobian` paths unchanged. New `test_pyblp_backend_matches_pyblp_results_path` asserts `_compute_markups(pyblp_results=X)` == `_compute_markups(demand_backend=PyBLPBackend(X))` to `atol=1e-14`. |

Post-step-3 polish (pending commit): `NestedLogitBackend` split from `backends/logit.py` into its own `backends/nested_logit.py`. Rationale is user-facing: tracebacks point at the right file, auto-generated API docs get one page per backend class, future additions (RC-logit analytical, BLP analytical) fit naturally without another rename.

### What step 3 deliberately did NOT do (deferred to step 4)

The plan's §5 step 3 and step 4 descriptions were updated to reflect these deferrals accurately. The gap between original plan and actual step 3 ships:

1. **Vertical-integration Hessian** in `_compute_markups` still uses the legacy code — `construct_passthrough_matrix` for the `pyblp_results` path, `compute_analytical_hessian` for the `demand_jacobian` path. Threading `backend.compute_hessian(market_id=t)` through the vertical loop is step 4.

2. **`pyRVtest/demand_jacobian.py` is kept as a backward-compat shim**, not deleted. Its canonical content lives in `backends/logit.py`; the shim re-exports the module-level functions. Deletion + import updates across `problem.py`, `markups.py`, and the two `test_demand_params*.py` files is step 4.

3. **`Problem.__init__` does not yet construct a backend internally.** The `demand_backend` parameter lives on `_compute_markups` only. Users calling `Problem(demand_results=...)` or `Problem(demand_params={...})` still go through the legacy construction paths. Problem-level backend integration is step 4 — doing it requires the vertical path (deferral 1) to also be backend-aware.

4. **`LogitBackend` and `NestedLogitBackend` do NOT implement `SupportsDemandAdjustment`** yet. They have only the core `DemandBackend` methods. Adding `demand_moments`, `xi_gradient`, `jacobian_gradient` to them requires re-deriving the analytical demand-adjustment logic from `_compute_analytical_demand_adjustment` (the code that had Bugs 1 and 2 in b3b08a3). That work happens in step 4 under the first-stage-correction tests' watchful eye.

All four deferrals are aligned with the plan's §5 step 4 description.

### Why additive-only wiring was the right call

Step 3e could have REPLACED the legacy demand-Jacobian fetching logic in `_compute_markups` instead of adding a parallel path. I chose additive because:

- Step 0 snapshots hash the legacy behavior. Any change to the default code path risks snapshot drift.
- The first-stage-correction equivalence tests (9 tests) currently verify the b3b08a3 bug fixes. Those tests go through the legacy path; any refactor of that path needs to preserve their passing state.
- Additive wiring lets the next Claude commit to step 4 (which DOES need to change the default path) with the confidence that step 3's six sub-commits can each be reverted independently without disturbing the rest.

### Verification

Full test suite on `v0.4-refactor` after step 3 + split: **155 passed + 3 skipped in 2:47**. All 6 Step 0 snapshots still match at `atol=1e-10`. All 9 first-stage-correction equivalence tests still pass at machine precision. All 31 new backend unit tests pass.

### What coauthors should notice

- **No behavior change yet.** If you pull `v0.4-refactor` and run your existing code, outputs are byte-identical to `CClean-fixes`.
- **New import paths are available.** `from pyRVtest.backends import PyBLPBackend, LogitBackend, NestedLogitBackend, UserSuppliedBackend` works. Classes are importable and unit-tested but not yet used by `Problem.solve()`.
- **Step 4 is where the real rewire happens.** That's the risky one — it deletes the two parallel demand-adjustment paths (`_compute_analytical_demand_adjustment` and `_compute_demand_adjustment_gradient` in `problem.py`) and replaces them with a single backend-generic `solve/demand_adjustment.py`. If anything breaks there, Step 0 snapshots + first-stage-correction tests catch it at the commit that introduces the regression.

---

## 2026-04-16 (late evening, post step 2) — v0.4 migration step 2 landed

Commit `f7da57b` on `v0.4-refactor`. Extracts the `Products` class from `pyRVtest/problem.py` into a standalone `pyRVtest/products.py` with mypy-strict type hints. No behavior change; all 6 Step 0 snapshots still match at `atol=1e-10` and the full test suite (121 + 3 skipped) passes in 2:43.

Small scope widening: added `__all__` to `pyRVtest/formulation.py` (needed so mypy can resolve the `Formulation` import in products.py without a broad `# type: ignore`). Consistent with the incremental-`__all__` rule in §5 of the plan.

New infrastructure: `mypy.ini` with lax defaults and a `[mypy-pyRVtest.products]` strict section. Future steps append a strict section as each new module lands. `requirements-dev.txt` now pulls `mypy>=1.0.0` alongside `hypothesis`.

`_qr_residualize` deliberately stays in `problem.py`; step 8 relocates it.

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
