# MEMO — F-stat reliability diagnostic: design, calibration, and implementation

**To:** Lorenzo Magnolfi, Marco Duarte
**From:** Chris Sullivan
**Date:** 2026-04-28
**Subject:** `F_reliability` diagnostic for pyRVtest — implemented on `feat/f-reliability`, ready for review

## Executive summary

A structured reliability diagnostic alongside `results.F` that flags when (a) the F-statistic's *value* is numerically unreliable due to denominator cancellation and (b) the user's *conclusion* about instrument strength is sensitive to sampling. The design is informed by Lorenzo's "indeterminate zone" framing and Codex's "false precision is the bug" reframe; threshold values are calibrated against simulation rather than guessed.

Two verdicts (plus the existing trivially-degenerate NaN guard):

- **near-degenerate**: λ < 0.05, where λ = ((σ₀+σ₁)² − 4σ₂²) / (σ₀+σ₁)². Calibrated against numerical fragility (drift > 1% under realistic σ-noise).
- **borderline**: the asymptotic 95% confidence interval for the population F overlaps the relevant CV for the strongest size or power claim. Equivalently: F − 1.96·SE(F) < CV.

**Status: implemented on branch `feat/f-reliability`** (4 commits, 26 new tests, full existing test suite still green). PR URL: https://github.com/anyatarascina/pyRVtest/pull/new/feat/f-reliability. The five open threshold questions in §"Open questions" remain open — your input determines whether to re-tune any of the calibrated values before this lands.

## Problem statement

Lorenzo flagged that pyRVtest currently reports F as a scalar with stars and lets users interpret the verdict, even though F is **inherently numerically sensitive** in the regime it is designed to flag. Codex sharpened this: "false precision is the bug, not the F-shift." When F is near a critical-value boundary, or when the σ-determinant is near zero, the user-facing output should surface that fragility rather than report a clean scalar with stars.

Two distinct failure modes:

1. **Numerical fragility.** When the moment-condition Gram-matrix structure approaches degeneracy (σ₀ ≈ σ₁ ≈ σ₂), F's denominator is a near-cancellation of nearly-equal terms. Tiny perturbations to σ values — from LAPACK build differences, BLAS variants, floating-point ordering, or ill-conditioned controls matrices — get amplified into observable F shifts. The `analytical_scale` test fixture demonstrated ~3% F drift across LAPACK builds at σ₀ ≈ σ₁ ≈ σ₂ ≈ 2 × 10⁻³.

2. **Statistical fragility.** F's sampling distribution at given (K, ρ̂, μ²) has natural spread. If F is close to a critical-value boundary, the asymptotic CI for the population F overlaps the CV — meaning the *strength claim* the user reports is decision-uncertain at conventional levels (the same reason a regression coefficient with t = 1.5 is not reported as "significant"). This is what Codex called "distance to nearest CV," made precise as a CI overlap.

These are independent: a cell can be one, the other, both, or neither.

## Conceptual framework

The user's actual question when looking at F is "**Can I claim my instruments are strong, and how confident is that claim?**" The DMSS framework gives them two parallel claims:

- **Strong for size at level X%**: F > r_X gives "worst-case size ≤ X%" (X ∈ {7.5, 10, 12.5}).
- **Strong for power at level Y%**: F > r_Y gives "best-case power ≥ Y%" (Y ∈ {50, 75, 95}).

Researchers report the **strongest claim they can make** — the strictest level F clears.

The reliability diagnostic answers two distinct questions about that claim:

- **near-degenerate**: "Can I trust the *value* of F=4.3 at all?" → no, if numerical cancellation has eaten most of the denominator's magnitude.
- **borderline**: "Is my point-estimate F precise enough that the strength claim is decision-stable at conventional inference levels?" → no, if the asymptotic 95% CI for the population F overlaps the relevant CV.

**Important conceptual note:** The borderline diagnostic operates at a different level than the CV's calibration. The CV is constructed (DMSS Proposition 4) so that under the worst-case null on the noncentrality, P(F > CV) ≤ 5% — that's the test's Type I error guarantee, and it remains intact. The CI-overlap diagnostic is a separate inferential question about the *precision of the F point estimate* — analogous to checking whether a regression coefficient's CI excludes zero. The two probabilities (the CV's 5% and any CI calculation) condition on different distributions and do not contradict each other.

## Verdict design

```
trivially-degenerate  if  NaN guard fires (existing at solve/test_engine.py:302)
near-degenerate       if  λ = ((σ_0+σ_1)² - 4σ_2²) / (σ_0+σ_1)² < 0.05
borderline            if  asymptotic 95% CI for population F overlaps the relevant
                          CV for the strongest size or power claim
                          (equivalently:  F - 1.96·SE(F) < CV)
robust                otherwise
```

The two non-trivial verdicts are calibrated, not heuristic.

### Near-degenerate calibration

λ measures how much of F's natural denominator magnitude has been lost to cancellation. λ ∈ [0, 1] with:
- λ = 1: no cancellation, F's value is fully precise
- λ = 0: total cancellation, F is undefined

We calibrated λ < 0.05 by perturbing (σ₀, σ₁, σ₂) by a relative noise level matching the empirical observation from `analytical_scale` (~10⁻⁴ relative). Below λ = 0.05, F drifts > 1% under this noise level — operationally meaningful given star spacing.

The 1e-4 noise level is the conservative target. It comes from the inferred σ-noise that produced the analytical_scale fixture's observed 3% F drift across LAPACK builds. For applications with well-conditioned controls matrices, actual σ-noise is much smaller (closer to machine epsilon), and the threshold remains conservative.

The diagnostic is **directly computable** from σ values that pyRVtest already has in `solve/test_engine.py:281–285`. Reduces to ρ²_pearson > 0.95 in the symmetric case σ₀ = σ₁ but generalizes correctly to asymmetric cases (where Pearson and the relevant denominator measure diverge).

### Borderline calibration

For each cell, we compute the **strongest claim** the user can make (e.g., "worst-case size ≤ 10%" if F > r_10) and ask whether the asymptotic 95% CI for the population F overlaps the relevant CV. If so, the claim is decision-uncertain at conventional inference levels and the cell is flagged borderline.

The asymptotic standard error of F under the paper's distribution is
```
SE(F) = (1 - ρ̂²) / (2K) · sqrt(2 · (2K + 2·nc_implied))
nc_implied = max(0, 2K · (F / (1 - ρ̂²) - 1))
```
and the borderline condition is `F - 1.96·SE(F) < relevant_CV`.

This is the same structural test as a confidence-interval-vs-zero check on a regression coefficient: the test about the parameter (here, "is F > CV") is decision-uncertain when the CI for F overlaps the CV.

The 1.96 cutoff is the conventional 95% CI half-width. Tightening (2.58 → 99% CI) flags fewer cells; loosening (1.65 → 90% CI) flags more. We default to 95%.

**This diagnostic is independent of the CV's calibration.** The CV's 5% Type I error guarantee under the worst-case null (DMSS Proposition 4) is intact and unchanged by the borderline diagnostic. The CI-overlap question is about the *precision* of the F point estimate at conventional inference levels — analogous to reporting CIs alongside coefficient estimates in a regression table.

Calibration was done in two stages: a Monte Carlo sweep over (K, ρ̂, μ_p) using F samples drawn from the paper's asymptotic distribution, and a check that the simulator's distribution matches the published CV-table parameterization (eq. 17 in the paper).

## Output design

Three knobs, jointly producing minimal-but-honest output:

1. **Glyphs in the F-stat table** — fragile cells get a `⚠` marker next to their F value.
2. **Footer** — for each fragility type that fires, two lines: the worst-case sensitivity number plus methodological caveat.
3. **`results.F_reliability_summary()`** — opt-in detailed table with one row per cell, columns: F, SE(F), 95% CI for F, ρ̂², λ, strongest size/power claims, verdict.

The all-clear case shows a single one-line "all robust" footer with worst-case ρ̂² and minimum λ for paper-appendix reporting.

For symbol notation, we propose moving F-stat size markers from `*` to `†` (repeated dagger: `†`, `††`, `†††`) so that `*` becomes available for two-sided TRV significance markers (`*` = |TRV| > 1.64, `**` = > 1.96, `***` = > 2.58). This closes a real interpretation gap (TRV had no symbols previously) and avoids the same glyph meaning two different things.

### Sample output

```
Testing Results - Instruments z0:
==========================================================================================
   TRV:                  |     F-stats:              |    MCS:
       0        -2.1 -0.3|         12.3      8.7     |    0.234
                  **     |         ††† ^             |
       1   2.1        0.4|  12.3             4.3⚠   |    0.456
                **       |  ††† ^^^          †       |
       2   0.3   -0.4    |   8.7    4.3⚠            |    0.789
                         |   † ^     †               |
==========================================================================================
TRV significance (two-sided, asymptotic):
  *, **, *** indicate |TRV| > 1.64, 1.96, 2.58
F-stat significance (DMSS):
  †, ††, ††† indicate F > cv for worst-case size of 0.125, 0.10, 0.075 (given d_z, ρ̂)
  ^, ^^, ^^^ indicate F > cv for best-case power of 0.50, 0.75, 0.95 (given d_z, ρ̂)
F-stat reliability:
  ⚠ borderline (1 cell): F = 4.3 (95% CI [4.13, 4.47]), strongest size claim "worst-case size ≤ 10%"
     CI overlaps 5%-CV = 4.20: claim is decision-uncertain at conventional levels
     fallback claim still robust: "worst-case size ≤ 12.5%" (CI lower bound 4.13 > 12.5%-CV = 1.10)
  See: results.F_reliability_summary()
==========================================================================================
```

## Implementation status (all three phases complete)

Branch: `feat/f-reliability`. Commits, in order:

| Commit | Phase | Scope |
|---|---|---|
| `ab0dc9a` | Design | This memo + the session handover |
| `83069f0` | **Phase 1** | Core diagnostic math + `F_reliability_summary()` method, no UI changes |
| `db708dd` | **Phase 2** | Glyphs in F-stat cells + reliability footer in `__str__` |
| `92b9800` | **Phase 3** | TRV stars at 1.64/1.96/2.58 + F-stat size symbols moved `*` → `†` to free `*` |

Files touched:
- `pyRVtest/solve/test_engine.py` — λ, SE(F), CI bounds, strongest-claim labels, verdict, TRV symbols, dagger swap
- `pyRVtest/problem.py` — extracts new fields per instrument and threads into Progress
- `pyRVtest/results/results.py` — new fields on Progress, new attributes on ProblemResults, new `F_reliability_summary()` method, glyph + footer rendering in `_format_results_tables`
- `pyRVtest/output.py` — `format_table` accepts an `extra_notes` parameter; caption restructured into TRV / F-stat / reliability sections
- `tests/test_f_reliability.py` — 26 new tests covering math, attribute presence, output format, and Phase 3 symbol scheme

New attributes on `ProblemResults`: `lambda_dmss`, `F_se`, `F_ci_low`, `F_ci_high`, `verdict`, `strongest_claim_size`, `strongest_claim_power`, `_symbols_rv_list`. New method: `F_reliability_summary()` returning a long-form DataFrame.

Threshold values (calibrated, exposed as module constants in `solve/test_engine.py`):
- `RELIABILITY_LAMBDA_THRESHOLD = 0.05`
- `RELIABILITY_CI_LEVEL = 1.96`

Can be promoted to a `Problem.solve(reliability_thresholds=...)` kwarg in a follow-up if you want them user-tunable.

Test results: 484 existing tests pass + 26 new pass (full suite excluding pre-existing environment-dependent failures unrelated to this work, like `to_latex` requiring jinja2). Snapshot suite still green; the existing `analytical_scale` xfail is unchanged.

Estimated time: 2–3 days of careful work. Can ship as v0.4.1 (post the v0.4.0 release).

## Open questions for coauthors

1. **σ-noise calibration target (currently 1e-4).** This level matches the empirical analytical_scale observation. If you'd rather calibrate against typical (well-conditioned) applications, a stricter noise level (e.g., 1e-6) would yield a much smaller λ threshold (~5×10⁻⁴), flagging only truly-degenerate cells. The 1e-4 choice is conservative — flags cells where a numerically pathological setting *could* matter. Want this changed?

2. **CI level (currently 95%, threshold 1.96).** Maps to conventional CI conventions. Could be relaxed (90% CI, threshold 1.65) to flag fewer cells, or tightened (99% CI, threshold 2.58) to flag more. The 95% level matches what researchers use elsewhere when reporting CIs.

3. **TRV stars (proposed addition).** Adding two-sided significance markers on TRV is a parallel feature, not strictly part of the F-reliability work. If you'd rather keep TRV un-marked, the F-stat symbols can stay as `*`/`**`/`***` and skip the dagger move. Mostly a question of whether closing the TRV-symbol gap is worth the change.

4. **Retrospective replication.** Codex suggested running this diagnostic on DMSS, DMQSW, DMQS, Backus-Conlon-Sinkinson published tables before locking in thresholds. The thresholds we propose feel defensible from simulation, but a published-tables retrospective would let us state empirically how often real applications would have flagged. Worth doing?

5. **`F_reliability_summary()` as opt-in or auto-shown.** Default proposal: keep it opt-in (call the method explicitly). If even one cell is flagged, the footer carries the worst-case info, and the user can drill in via the method. Alternative: auto-show when any cell flags. Slightly less back-compat-friendly.

## Files for review

**Implementation (in `pyrvtest`, branch `feat/f-reliability`):**
- `pyRVtest/solve/test_engine.py` — diagnostic math + TRV symbols + dagger swap
- `pyRVtest/problem.py` — wiring per-instrument
- `pyRVtest/results/results.py` — Progress fields, ProblemResults attributes, `F_reliability_summary()`, glyph + footer rendering
- `pyRVtest/output.py` — caption sections + `extra_notes` plumbing
- `tests/test_f_reliability.py` — 26 new tests

**Calibration (in `degeneracy-conduct-testing/code/calibration/`):**
- `cv_simulator.py`, `validate_against_published.py` — Python port of the MATLAB CV generator
- `fragility.py`, `analyze_fragility.py` — statistical fragility (borderline calibration)
- `numerical_fragility_v2.py`, asymmetric sweeps — numerical fragility (near-degenerate calibration)
- `KNOWN_ISSUES.md` — a small persistent CV-table offset at high-ρ, high-K (does not affect the calibration here)

Review priorities, in order:
1. **The math.** Inspect the verdict logic in `pyRVtest/solve/test_engine.py` (lines around the F-stat loop) and the F SE / CI / strongest-claim derivation. Specifically: does the implied-noncentrality formula and the asymptotic SE match what Mikkel's framework gives?
2. **The threshold values** (open questions §). Especially the σ-noise calibration target and the dagger swap.
3. **The printed output.** Run `print(results)` on a real fixture and let me know if the layout, glyphs, and footer text read well in your applications.
4. **The `F_reliability_summary()` schema.** Check the column set is what you'd want for paper-appendix tables.

Pointing your AI agents at the PR is fine — happy to iterate on any of the structural choices.
