# Plan — DMQSW pass-through diagnostic suite (v0.4)

> **Note on scope**: an earlier draft of this plan staged the diagnostic
> for v0.5. The decision to pull it into v0.4 (between rc1 and final tag)
> simplifies the backward-compat layer substantially: rc1 was internal /
> coauthor-only, so renames and removals can land directly without
> deprecation aliases. The earlier v0.5-framed version is in git history
> at commit `d4b6e31` if needed for reference.

## Goal

Ship the full Dearing, Magnolfi, Quint, Sullivan, and Waldfogel (2026)
("Learning Firm Conduct: Pass-Through as a Foundation for Instrument
Relevance") diagnostic suite **as part of v0.4 final**, alongside the
DMSS (Duarte et al. 2024) reliability machinery and the DMQSS
(non-constant cost) extension already in rc1.

The release tells one coherent paper-package story: the test (DMSS),
the framework that motivates instrument selection (DMQSW), and the
non-constant-cost extension (DMQSS) all ship together with first-class
diagnostics for each.

## Design principles

1. **Pre-solve = structural; post-solve = empirical with framework
   labels.** No data-driven instrument selection methods (no
   ``moment_relevance``). Pre-solve methods report γ-free framework
   features; post-solve methods report γ-fitted magnitudes alongside
   the inference-honest reliability table.

2. **No verdict labels.** Direct CV reporting in ``reliability_summary``;
   the user reads off the reliability claim that matches their (α, β)
   application. No package-imposed reliable / weak / degenerate
   classification.

3. **Numerical-first implementation.** One numerical perturbation routine
   handles every conduct class plus composites (where applicable),
   replacing per-conduct closed-form derivations. Analytical formulas
   stay in math.rst as reference; v0.5+ can add closed-form fast paths
   if performance becomes a bottleneck.

4. **Composite-aware honesty.** Composite IVs (BLP-style sums,
   differentiation IVs) get partial framework coverage: the indirect
   channel via empirical ``dp_0/dz`` regression, the direct channel via
   conditional regression of Δ_m on z given p. The package does not
   pretend to apply primitive-instrument formulas to arbitrary
   composites.

5. **Self-documenting output.** Each method's printed view ends with a
   short Notes block (4–10 lines) flagging what's empirical vs.
   structural and pointing at the docstring + DMQSW for methodology.
   No verbose "How to read" + "Methodology" footers.

## Out of scope for v0.4 final

- Closed-form analytical pass-through per conduct (numerical-first;
  v0.5+ can add closed-form fast paths).
- ``moment_relevance`` (empirical pre-screening; suppressed to avoid
  post-selection inference issues).
- ``counterfactual_relevance`` (Section 3.4 / Remark 6 of DMQSW; deferred
  to v0.5 unless user demand emerges).
- ``F_se`` / ``F_ci_low`` / ``F_ci_high`` (removed entirely; DMSS verdict
  + CVs is the inference channel — see Phase 4).
- Instrument-type metadata for package-generated composites
  (``pyRVtest.instruments.blp_instruments`` etc. could thread through
  primitive-column metadata for full direct-channel decomposition; v0.5+).
- Demand-curvature flexibility checks (DMQSW Section 4.2; doc-only
  addition can ship independently).

## Final API surface

### Pre-solve

```
print(problem)                                    brief summary + hint
problem.passthrough_summary([with_models=True])   pair × feature distances + per-feature notes
problem.passthrough_matrix(model_idx, market_id)  raw P_m for one model in one market
```

### Post-solve

```
print(results)                                    existing TRV/F/MCS table (unchanged)
results.reliability_summary([as_dataframe=True])  CVs + F-stat + rho² (renamed from F_reliability_summary)
results.passthrough_summary(...)                  same as pre-solve; callable post-solve too
results.causal_effects(column=...)                γ-fitted dp/dz with channel decomposition
```

### Sample outputs

#### `problem.passthrough_summary()`

```
Per-pair pass-through-feature distances (median across 3000 markets):

                          offdiag_ratio  full_pass   row_sum    level_adj
(Bertrand, Cournot)              0.245      0.412      0.532       2.41
(Bertrand, Monopoly)             0.398      0.722      1.241       3.18
(Bertrand, PC)                   0.245      0.412      0.014       0.39
(Cournot,  Monopoly)             0.341      0.467      0.529       2.83
(Cournot,  PC)                   0.000      0.350      0.532       1.62
(Monopoly, PC)                   0.341      0.587      0.227       0.74

Per-feature notes:
  offdiag_ratio (rival cost shifters): γ-free; column ratios. Zero ⇒ structural
    degeneracy. Magnitude doesn't predict power.
  full_pass (own+rival cost; product chars under linear-index demand): γ_known
    scaled or γ=0 by rival exclusion; full pass-through difference.
  row_sum (unit tax): row sums of pass-through. ν observed; fully computable.
  level_adj (ad valorem tax): ‖P_m·(p−Δ_m) − P_m'·(p−Δ_m')‖. ν, p observed.

Distances are computed from candidate models + demand fit at observed states.
See passthrough_summary() docstring and Dearing et al. (2026) for derivations.
```

With `with_models=True`, prepends a per-model structural block:

```
Per-model pass-through structure (median across 3000 markets):

  model              diag_avg   max_offdiag   row_sum_avg
  Bertrand              0.91         0.16         0.99
  Cournot               0.46         0.00         0.47
  Monopoly              0.65        -0.35         0.45
  PerfectComp.          1.00         0.00         1.00

[per-pair table as above]
```

#### `results.reliability_summary()`

```
DMSS critical values at worst-case ρ:
  IV set z0:
                   α=.075   α=.10   α=.125          β=.50   β=.75   β=.95
    Size CV:         5.1     4.7     4.4              3.1    16.2    28.4

Per-pair F-stat and rho²:
  pair                  IV set    F-stat    rho²
  (Bertrand, Cournot)   z0          92.8   0.873
  (Bertrand, Monopoly)  z0         170.2   0.937
  (Bertrand, PC)        z0           2.6   0.087
  (Cournot,  Monopoly)  z0         178.4   0.940
  (Cournot,  PC)        z0           0.0   0.000
  (Monopoly, PC)        z0           1.4   0.058

Notes:
  F-stat and rho² are empirical estimates with finite-sample noise. Compare
  F-stat to size CVs (α-thresholds) and power CVs (β-thresholds) to read off
  reliability claims. CVs at empirical rho² (sharper but ρ-dependent) are
  available as additional columns in the returned DataFrame.
  See reliability_summary() docstring for methodology and Duarte, Magnolfi,
  Sølvsten, Sullivan (2024) for derivations.
```

Full DataFrame (when called with `as_dataframe=True` or via attribute access)
columns:

```
pair, model_i, model_j, iv_set,
F, rho_squared,
size_cv_075, size_cv_100, size_cv_125,
power_cv_050, power_cv_075, power_cv_095,
size_cv_075_emp, size_cv_100_emp, size_cv_125_emp,
power_cv_050_emp, power_cv_075_emp, power_cv_095_emp,
F_high_precision, rho_squared_high_precision,
lambda_dmss
```

#### `results.causal_effects(column='rival_z2')`

```
Post-solve causal-effect decomposition: rival cost shifter rival_z2.
γ_m fitted from solve.

Indirect channel = P_m · (P_m^{-1} − P_m'^{-1}) · (dp_0/dz)
                          └─── structural ────┘   └── data ──┘

Data-side: empirical effect of z on prices
  ‖dp_0/dz‖_obs = 0.487     (sample regression slope of p on rival_z2)
  SD(rival_z2)  = 0.412

Structural-side: per-pair inverse pass-through difference
  pair                       ‖P_m^{-1} − P_m'^{-1}‖
  (Bertrand, Cournot)                          1.04
  (Bertrand, Monopoly)                         1.43
  (Bertrand, PC)                               0.40
  (Cournot,  Monopoly)                         0.39
  (Cournot,  PC)                               0.00
  (Monopoly, PC)                               1.51

Direct channel: per-candidate β_m (OLS slope of Δ_m on z | p)
  Bertrand          β = 0.000   (cost shifter — markup independent of z)
  Cournot           β = 0.000
  Monopoly          β = 0.000
  PerfectComp.      β = 0.000

Per-pair channel decomposition:

  pair                       indirect    direct    total
  (Bertrand, Cournot)            0.534     0.000    0.534
  (Bertrand, Monopoly)           0.892     0.000    0.892
  (Bertrand, PC)                 0.534     0.000    0.534
  (Cournot,  Monopoly)           0.741     0.000    0.741
  (Cournot,  PC)                 0.000     0.000    0.000
  (Monopoly, PC)                 0.741     0.000    0.741

Notes:
  Numbers are empirical estimates with finite-sample noise. Direct channel
  β_m is the OLS slope of model-implied Δ_m on z controlling for p (FWL).
  Indirect = P_m · (P_m^{-1} − P_m'^{-1}) · dp_0/dz at observed states.
  Structural-side ‖P_m^{-1} − P_m'^{-1}‖ has no IV sampling noise.
  See causal_effects() docstring for methodology and Dearing et al. (2026)
  for the framework.
```

For composite IVs (BLP, differentiation), the direct channel is identified
via the same conditional regression — no special handling needed since
β_m comes from regressing Δ_m on z | p regardless of z's primitive
structure. The structural-side and data-side blocks are computed
identically.

## Phases

### Phase 1 — Numerical pass-through for all conducts (1–2 sessions)

Generalize the existing Vertical pass-through machinery to dispatch
across all conduct classes via a single numerical perturbation routine.

**Steps:**

1.1 Add `compute_passthrough_numerical(conduct_model, p, ownership,
    response, hessian, shares, markups, δ=1e-7)` in
    `pyRVtest/solve/passthrough.py`. Computes `∂Δ_m/∂p` by central-
    difference perturbation of prices in the candidate's
    `evaluate_first_order_conditions` machinery, then
    `P_m = (I − ∂Δ_m/∂p)^{-1}`. Same numerical pattern handles every
    conduct that exposes a markup function (all of them).

1.2 Generalize `build_passthrough(problem, model_index, market_id)` to
    dispatch through `compute_passthrough_numerical` rather than the
    Vertical-only Villas-Boas closed form. Drop the `is_vertical`
    validation gate.

1.3 Update `math.rst` to document the closed-form analytical formulas
    per conduct as reference (Bertrand, Cournot, Monopoly, JP,
    PartialCollusion, MixCournotBertrand, RuleOfThumb, ConstantMarkup,
    PerfectCompetition, Vertical, UserSuppliedMarkups). The package
    computes numerically; math.rst documents what the numerics
    approximate.

1.4 Tests in `tests/test_passthrough_numerical.py`:
    - Numerical vs. trivial closed forms: PC, ConstantMarkup,
      UserSuppliedMarkups all match `I` exactly. RuleOfThumb(φ) matches
      `φ·I` exactly.
    - Numerical vs. existing Vertical analytical implementation:
      bit-identical or within 1e-12 on a 100-market vertical fixture.
    - Numerical vs. paper Example 2 hand-computed 2×2 logit matrices
      for Bertrand, Cournot, JP, Keystone — match within 1e-9.
    - Smoke tests: every conduct class returns a finite, well-conditioned
      `P_m` on the synthetic example data.

**Deliverables:** generalized `build_passthrough`; numerical core;
tests; math.rst updates.

### Phase 2 — `passthrough_summary` (1–2 sessions)

Pre-solve (and post-solve) multi-feature per-pair distance table with
per-feature footnotes.

**Steps:**

2.1 Implement metric registry: `full_pass`, `offdiag_ratio`, `row_sum`,
    `level_adj`. Each is a callable
    `(P_m, P_m', p, Δ_m, Δ_m') → scalar` aggregated across markets.

2.2 `Problem.passthrough_summary(with_models=False, detail='median')` →
    pandas DataFrame + printable summary view. One row per candidate
    pair, one column per feature.

2.3 `with_models=True` adds a per-model structural block (`diag_avg`,
    `max_offdiag`, `row_sum_avg` — median across markets).

2.4 `detail='full'` returns the per-(market, pair) frame instead of
    aggregating; `detail='median'` is the default.

2.5 Per-feature footnotes (one short line each) hardcoded in the printed
    view; longer per-feature text in the docstring.

2.6 **Methodology line** in the printed output indicating how the
    underlying pass-through matrices were computed, conditional on the
    candidate-set composition:
    - All non-Vertical candidates: "Pass-through computed numerically
      via central-difference perturbation through the markup function
      (delta=1e-7). See docs/math.rst."
    - All Vertical candidates: "Pass-through computed analytically via
      Villas-Boas (2007) closed form. See docs/math.rst."
    - Mixed: "Pass-through: Villas-Boas analytical for Vertical
      candidates; central-difference numerical (delta=1e-7) for
      others. See docs/math.rst."
    Trivial-closed-form short-circuit conducts (PC, ConstantMarkup,
    UserSuppliedMarkups, RuleOfThumb) are noted as "exact" rather than
    numerical when present.

2.7 `ProblemResults.passthrough_summary` is a thin wrapper delegating
    to the underlying Problem (results.problem.passthrough_summary(...)).

2.8 Tests on the synthetic example confirming `(Cournot, PC)` shows
    `offdiag_ratio = 0.000` and other features nonzero; cross-validate
    median aggregation against a hand-computed sample.

**Deliverables:** `passthrough_summary` on Problem and ProblemResults;
per-model block; metric registry; methodology line; tests.

### Phase 3 — `causal_effects` with channel decomposition (1–2 sessions)

Post-solve magnitude decomposition of per-pair `‖dp_m/dz − dp_m'/dz‖`
into indirect (pass-through-mediated) and direct (markup-derivative)
channels.

**Steps:**

3.1 `ProblemResults.causal_effects(column=None, instrument=None)`. If
    `column=None`, iterates over all instrument columns in declared
    `instrument_formulation`s. If `column=...`, drills into one column.
    `instrument=` reads from `Problem(instrument_types=...)` declaration
    by default; explicit override accepted.

3.2 Three computation blocks:
    - **Data-side**: `‖dp_0/dz‖_obs` from OLS regression of observed
      prices on z with cost-formulation controls. Sample `SD(z)`,
      `range(z)`.
    - **Structural-side**: per-pair `‖P_m^{-1} − P_m'^{-1}‖_F` aggregated
      across markets.
    - **Direct channel**: per-candidate `β_m` from OLS regression of
      model-implied `Δ_m,jt` on `z_jt` with `p_jt` as a control (FWL
      partialling). For cost shifters and taxes (not in markup
      function), `β_m ≈ 0`.

3.3 Combined per-pair table: `indirect`, `direct`, `total`. Indirect
    computed as `P_m · (P_m^{-1} − P_m'^{-1}) · dp_0/dz` evaluated per
    market and aggregated. Direct = `β_m − β_m'`. Total = indirect +
    direct.

3.4 Per-instrument-type footnotes hardcoded for primitive types; for
    composite or unlabeled types, footnote acknowledges the composite
    treatment (direct channel via conditional regression works for any
    z; structural and data sides too).

3.5 **Methodology line** in the printed output (same scheme as Phase 2's
    step 2.6): document how the underlying pass-through matrices were
    computed, conditional on the candidate-set composition (numerical
    central-difference / analytical Villas-Boas / mixed). Note also
    that `dp_0/dz` comes from a sample regression (not analytical) and
    `β_m` from a conditional regression of `Δ_m` on `z` given `p`. The
    user reads the methodology line and knows which numbers in the
    table are empirical vs structural without consulting the docstring.

3.6 Tests:
    - On synthetic example with rival cost shifter: direct channel = 0
      across all candidates and pairs (cost shifters don't enter markup).
      Indirect for `(Cournot, PC)` is exactly zero.
    - On a synthetic product-char fixture: `(Cournot, RT(2))` shows
      indirect ≈ 0 (both diagonal P) but direct ≠ 0 (Cournot has
      markup-on-x dependence; RuleOfThumb doesn't).
    - Cross-validate indirect formula against per-market
      `dp_m/dz − dp_m'/dz` differences (within numerical tolerance).

**Deliverables:** `causal_effects` method; conditional regression for
direct channel; per-pair table assembly; tests.

### Phase 4 — rc1 → final cleanup (≤0.25 session)

Direct renames and removals — no deprecation layer, since rc1 was internal.

4.1 **Rename** `F_reliability_summary` → `reliability_summary` on
    `ProblemResults`. No alias.

4.2 **Remove `F_se`, `F_ci_low`, `F_ci_high` from the codebase entirely.**
    - Remove computation in `pyRVtest/solve/test_engine.py` (or
      wherever the SE is computed).
    - Remove the per-cell array attributes from `ProblemResults`.
    - Update api.rst's "Per-cell array attributes" list.
    - Remove from any returned DataFrames (reliability_summary,
      summary_df, etc.).

4.3 **Remove `verdict` column** from `reliability_summary` returned
    DataFrame. Underlying classification logic stays if needed by
    `print(results)` symbol annotations (††/^^^), but not exposed in
    the diagnostic DataFrame.

4.4 **Rename CV columns** in `reliability_summary` to the explicit
    (axis, level) grid:
    - `worst_case_cv_size` → split into `size_cv_075`, `size_cv_100`,
      `size_cv_125`
    - `worst_case_cv_power` → split into `power_cv_050`, `power_cv_075`,
      `power_cv_095`
    - `strongest_claim_size` / `strongest_claim_power` → covered by
      `size_cv_075` / `power_cv_095`
    - Add empirical-ρ CV columns: `size_cv_*_emp`, `power_cv_*_emp`

4.5 `passthrough_comparison` (existing v0.4 rc1 method) is superseded by
    `passthrough_summary`. Removed outright; users redirected via the
    rc1 → final notes in CHANGELOG and migrating_to_v0.4.rst.

4.6 The "offdiag_frobenius is Remark 4" docstring claim in the existing
    rc1 code is wrong (or paper-renumbered). The metric is removed
    along with `passthrough_comparison`; the corresponding feature
    `offdiag_ratio` (Remark 1) lives in `passthrough_summary`.

**Deliverables:** clean v0.4-final API; updated docstrings; no F_se /
F_ci anywhere in the codebase; no v0.4 rc1 cruft.

### Phase 5 — Documentation (1–2 sessions)

5.1 Rewrite `docs/advanced_features.rst` Pass-through section: replace
    the "Vertical models only, v0.4" caveat with the three-method
    walkthrough on the synthetic example. Sections:
    - The framework's role in instrument selection
    - `passthrough_summary` walkthrough
    - `causal_effects` walkthrough
    - Reading `reliability_summary` alongside the framework view

5.2 Expand `math.rst`:
    - Per-conduct closed-form pass-through formulas (reference for
      what the numerics approximate)
    - Decomposition formula `dp/dz = P·(∂Δ/∂z + ∂c̄/∂z)` and the
      indirect/direct split
    - Conditional-regression derivation for direct channel
    - Worst-rho CV bound notation

5.3 New tutorial section/notebook: "Pre-test framework reasoning per
    DMQSW (2026)" demonstrating:
    - Pass-through inspection on the synthetic example
    - Identifying the (Cournot, PC) degenerate pair via offdiag_ratio = 0
    - Switching to unit-tax instruments to break the degeneracy
    - Post-solve `causal_effects` channel decomposition

5.4 FAQ entries:
    - "How do I check whether my IVs will distinguish my candidates
      before running the test?" (use `passthrough_summary`)
    - "Why doesn't `passthrough_summary` give me a verdict?" (γ-free
      structural feature distance; the framework can't predict
      finite-sample power)
    - "How do I tell if a weak F-stat means structural degeneracy or
      limited identifying variation in the data?" (cross-read
      `reliability_summary` against `passthrough_summary`)
    - "Why isn't there a `moment_relevance` or empirical pre-screening
      method?" (post-selection inference issues)

5.5 Update `docs/migrating_to_v0.4.rst` with the rc1 → final changes:
    - `F_reliability_summary` → `reliability_summary`
    - `F_se` / `F_ci_*` removed
    - `verdict` column removed
    - `passthrough_comparison` removed; use `passthrough_summary`
    - CV column renames
    - New `passthrough_summary` and `causal_effects` methods on
      `Problem` / `ProblemResults`
    - `Problem(instrument_types=...)` kwarg

5.6 README update: the v0.4 headline-features bullet list expands to
    include the DMQSW pass-through diagnostic suite.

5.7 Docstring methodology text for each new/renamed method
    (`reliability_summary`, `passthrough_summary`, `causal_effects`).
    Single source of truth for the methodology; printed-output footers
    point to the docstring.

5.8 CHANGELOG entry under `[0.4.0]` (final tag, not rc1) describing the
    diagnostic suite as a v0.4 feature.

**Deliverables:** rewritten advanced_features section; expanded
math.rst; tutorial section; FAQ entries; updated migration guide;
README and introduction.rst updates; docstrings; CHANGELOG.

### Phase 6 (optional) — Washington marijuana replication (1 session)

Replication script demonstrating the three-method workflow on the paper's
headline application.

**Deliverables:** `docs/notebooks/replication_DMQSW_marijuana.py` that
reproduces the Bertrand-vs-Keystone test using ad valorem tax instruments,
with ex-ante `passthrough_summary` framing and post-solve `causal_effects`
+ `reliability_summary` interpretation.

## Open design decisions

These can be resolved during implementation; flagging now so they don't
block later.

1. **DEGENERATE asterisk threshold in `passthrough_summary`.** No
   asterisks; users read the numbers. **Resolved.**

2. **Aggregation default for `passthrough_summary`.** Median across
   markets (`detail='median'`). `detail='full'` for per-market.
   **Resolved.**

3. **Per-feature footnote text length.** One line per feature in printed
   view; longer text in docstring. **Resolved.**

4. **Composite IV: structural-side computability.**
   `‖P_m^{-1} − P_m'^{-1}‖` is candidate-specific, not z-specific →
   computable for any z. ✓

5. **Composite IV: direct channel via conditional regression.** Works
   uniformly for primitive and composite z. No special-casing needed
   in v0.4. ✓

6. **Naming `causal_effects`.** **Resolved.**

7. **Naming `reliability_summary`.** **Resolved.**

8. **Worst-case-ρ CV layout: header block per IV set.** **Resolved.**

9. **Empirical-ρ CV columns in printed view.** Header block uses worst-ρ;
   empirical-ρ CVs only in the returned DataFrame. **Resolved.**

10. **`Problem(instrument_types=...)` declaration: optional.** Methods
    gracefully degrade if undeclared. **Resolved.**

11. **Composite IV labels: single `composite` fallback.** Parent-class
    labels (`composite_rival_product_char`) deferred to v0.5+.
    **Resolved (deferred).**

12. **`Problem.declare_composite_iv(...)` API.** Deferred to v0.5+.

## Risks

1. **rc1 → final scope creep.** Adding 3.75–7.5 sessions of new feature
   work to a release at rc1 risks pushing v0.4 final further out and
   introduces additional rc cycles (likely rc2 and rc3). Mitigation:
   ship in three coherent rc tags (see Incremental shipping below);
   keep coauthors informed of expected timeline; trim Phase 6
   (replication) if scope tightens.

2. **rc1 user breakage.** Anyone running v0.4.0rc1 in production sees
   `AttributeError` on `F_se`, `F_reliability_summary`, etc. Confirmed
   acceptable since rc1 is internal / coauthor-only. CHANGELOG entry
   documents the rc1 → final changes prominently.

3. **Numerical pass-through performance.** A 3000-market × 4-candidate
   diagnostic computes 12,000 numerical pass-through matrices, each
   requiring `J + 1` markup recomputations. For `J = 2` products this
   is ~36,000 markup evaluations; for larger `J` it scales linearly.
   Mitigation: cache the demand Hessian once per market and reuse
   across candidates; benchmark on a `J = 20` BLP fixture during
   Phase 1; if too slow, add a closed-form Bertrand fast path before
   final or defer to v0.5.

4. **Existing Vertical implementation regression.** Generalizing
   `build_passthrough` could subtly change Vertical numerics.
   Mitigation: cross-validate against the existing Villas-Boas closed
   form on the existing test fixtures; require bit-identical or
   within-1e-12 agreement before merging Phase 1.

5. **Conditional regression noise for direct channel.** OLS slope of
   `Δ_m` on `z` conditional on `p` has finite-sample noise. For
   primitive cost shifters where `∂Δ_m/∂z = 0` analytically, β_m might
   come out small-but-nonzero. Acceptable for diagnostic purposes;
   document as "near-zero β_m for cost shifters reflects sample
   correlation, not structural direct effect."

6. **Step 16 (AFSSZ dogfood) interaction.** Step 16 is already
   outstanding before v0.4 final per CHANGELOG. The new diagnostic
   work is independent of Step 16's data path, so they can proceed in
   parallel — but each could surface issues affecting the other.
   Mitigation: run the synthetic-example tests after each phase merge;
   keep Step 16 and the diagnostic suite in separate commits / merge
   units.

7. **Paper renumbering of Remarks.** The existing rc1
   `passthrough_comparison` docstring claims `offdiag_frobenius`
   "implements Remark 4." In the current DMQSW draft, the off-diagonal-
   to-diagonal ratio is **Remark 1**. Mitigation: drop inline Remark
   numbers from output; keep them only in math.rst where they're
   co-located with derivations.

## Test strategy

- **Numerical pass-through (Phase 1):** validate against trivial closed
  forms (`P = I` for PC, `φ·I` for RuleOfThumb), against existing
  Vertical implementation, and against paper Example 2 hand-computed
  2×2 logit matrices for Bertrand / Cournot / JP / Keystone.

- **`passthrough_summary` (Phase 2):** the synthetic example shows
  `(Cournot, PC)` with `offdiag_ratio = 0.000` and other pairs nonzero.
  Per-feature footnotes render correctly. `with_models=True` adds the
  per-model block with expected diagonal/off-diagonal patterns.

- **`causal_effects` (Phase 3):** synthetic example shows zero direct
  channel for cost shifters. Indirect channel reproduces the
  `(Cournot, PC)` degeneracy. Synthetic product-char fixture shows
  `(Cournot, RT(2))` as direct-only.

- **`reliability_summary` (Phase 4):** existing F-stat values unchanged;
  CV columns rename correctly; `F_se` / `F_ci_low` / `F_ci_high` raise
  `AttributeError`; `F_reliability_summary` raises `AttributeError`
  (no alias in v0.4 final since rc1 was internal).

- **End-to-end:** the README quick-start narrative now runs
  `passthrough_summary()` between Problem construction and `solve()`;
  the diagnostic output flags `(Cournot, PC)` ex-ante; post-solve
  `reliability_summary` confirms with F-stat = 0; `causal_effects`
  shows the zero indirect.

- **Snapshot regression suite:** existing snapshots (analytical_base,
  analytical_base_fe, first_stage_*_path, nested_logit_vertical, etc.)
  remain bit-identical or within tolerance. Any drift gets investigated
  before merge.

## Estimated effort

| Phase | Sessions |
|---|---|
| 1 — Numerical pass-through | 1–2 |
| 2 — `passthrough_summary` | 1–2 |
| 3 — `causal_effects` | 1–2 |
| 4 — rc1 → final cleanup (renames, drops; no aliases needed) | ≤0.25 |
| 5 — Documentation | 1–2 |
| 6 (optional) — Replication script | 1 |

**Total: 4.25–8.25 sessions, plus optional Phase 6.** Saves ~0.25
session vs. v0.5 framing because no deprecation aliases; otherwise
same scope.

## Incremental shipping plan — three rc tags to v0.4 final

Each merge can ship as an rc tag, letting coauthors test along the way:

**rc2: Phases 1 + 4** — pass-through generalization + rc1 cleanup. Drops
F_se / F_ci / verdict; renames `F_reliability_summary` → `reliability_summary`
and CV columns; generalizes `build_passthrough` beyond Vertical. Already
a meaningful release: removes the Vertical-only caveat and tightens the
inference API without yet introducing new methods.

**rc3: + Phase 2** — `passthrough_summary` lands. Researchers get the
structural pre-solve view with multi-feature per-pair distances.

**v0.4.0 final: + Phase 3 + Phase 5** — `causal_effects` + tutorial /
FAQ / docs. The headline DMQSW diagnostic is complete and documented.

Phase 6 (replication) ships as a follow-on commit on `main` or in v0.4.1.

This sequencing also means rc2 and rc3 catch any regression issues in
the pass-through machinery before the new methods are layered on top.

## Acceptance criteria for v0.4 final tag

- All conduct classes have working numerical pass-through (no
  Vertical-only restrictions).
- `passthrough_summary` returns the four pass-through-feature distances
  per pair on the synthetic example with the expected pattern
  (`(Cournot, PC)` shows `offdiag_ratio = 0.000`).
- `causal_effects(column='rival_z2')` returns the structural-side,
  data-side, direct-channel blocks plus the per-pair indirect/direct/
  total table. Direct channel is ~0 for cost shifters across all
  candidates.
- `reliability_summary` is the canonical name (no `F_reliability_summary`
  alias in final). `F_se` / `F_ci` / `verdict` columns are gone.
- `passthrough_comparison` removed; users redirected via migration
  guide.
- Documentation: advanced_features.rst Pass-through section rewritten;
  math.rst expanded; tutorial section + FAQ entries written;
  migrating_to_v0.4.rst updated; CHANGELOG entry under `[0.4.0]`.
- README highlights the DMQSW diagnostic suite as a v0.4 feature.
- All numerical results on the existing test suite are bit-identical
  or within numerical tolerance (1e-12 for closed-form-equivalent
  numerics; paper-derived analytical comparisons within 1e-9).
- Step 16 (AFSSZ dogfood) — independent of this plan; gated on AFSSZ
  panel data availability per the rc1 CHANGELOG note.

## Coauthor-coordination notes

- Coauthors who tested v0.4.0rc1 will see API renames in v0.4 final.
  Worth a heads-up email between rc1 and rc2 so they don't pin rc1.
- The DMQSW paper is authored with Lorenzo / Daniel / Adam / Sarah; if
  they're using pyRVtest in the Washington marijuana replication, the
  v0.4 final API is what they should be pinning. Phase 6 (replication
  script) gives them a working in-package example.
- Lorenzo's audits historically catch labor-side and exception-hierarchy
  issues. The new methods (`passthrough_summary`, `causal_effects`)
  should get a dedicated audit pass before final tag, similar to the
  rc1 break-it pass on 2026-04-18.
