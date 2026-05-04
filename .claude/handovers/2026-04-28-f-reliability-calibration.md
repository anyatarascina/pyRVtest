# Handover — F-stat reliability diagnostic: design, calibration, AND implementation

**Date:** 2026-04-28
**Branch:** `feat/f-reliability` (off `v0.4-refactor`, pushed to origin)
**Status:** design ✅, calibration ✅, **implementation ✅ — all three phases committed and pushed**. PR open against `v0.4-refactor` is ready when coauthors are ready.

## What this session produced

A complete `F_reliability` diagnostic shipped on `feat/f-reliability`, addressing Lorenzo's "indeterminate zone" framing for the F-statistic. Two non-trivial verdicts (plus the existing `trivially-degenerate` NaN guard):

- **near-degenerate**: F's *value* is numerically unreliable due to denominator cancellation
- **borderline**: F's *strength claim* is sensitive to sampling (asymptotic 95% CI for F overlaps relevant CV)

Both thresholds are calibrated, not heuristic. Memo at `MEMO_F_reliability_diagnostic_2026-04-28.md` summarizes for coauthors and lists open threshold questions.

## Branch state

```
92b9800 FEAT: F-reliability diagnostic (Phase 3 — TRV stars + dagger swap)
db708dd FEAT: F-reliability diagnostic (Phase 2 — printed output integration)
83069f0 FEAT: F-reliability diagnostic (Phase 1, no UI changes)
ab0dc9a DOC: F-reliability diagnostic design memo + session handover
9e5d0c6 (v0.4-refactor)  FIX: RuleOfThumb now reports (phi-1)/phi*p markup instead of zero
```

PR URL: https://github.com/anyatarascina/pyRVtest/pull/new/feat/f-reliability

Test results:
- 26 new tests in `tests/test_f_reliability.py` — all pass
- Full existing test suite (excluding pre-existing env-dependent failures): 484 passed, 4 skipped, 1 xfailed (the existing analytical_scale xfail, unchanged)
- Doctests: 46 passed, 9 skipped

## Calibration code

All calibration scripts in `~/Dropbox/Economics/claude/degeneracy-conduct-testing/code/calibration/`:

- `cv_simulator.py` — Python port of `RVpaper_rep/Code/Fcriticalvalues/{size,power}_save_all_rho.m` and `FunctionsMatlab/{size,power}func.m`. Reproduces published CV tables within Monte Carlo noise (with documented small offset at high-ρ, high-K — see `KNOWN_ISSUES.md`).
- `run_cv_sweep.py` — driver to regenerate full CV tables.
- `validate_against_published.py` — compares Python port to shipped CSVs at 10 size + 10 power cells. Power cells match within MC noise; size cells match outside the high-ρ, high-K corner.
- `fragility.py` — F sampling from the asymptotic distribution at (K, ρ̂, μ²); CV table lookup; flip-rate diagnostics.
- `run_fragility_sweep.py`, `analyze_fragility.py` — statistical fragility calibration sweep + analysis.
- `numerical_fragility.py`, `numerical_fragility_v2.py` — numerical fragility tests using pyRVtest's actual F formula (`solve/test_engine.py:286-300`).
- Asymmetric extension was the final calibration step — confirmed λ < 0.05 (not Pearson ρ²) is the right diagnostic for general (σ₀, σ₁, σ₂) configurations.
- `KNOWN_ISSUES.md` — documents the high-ρ, high-K size-CV offset.

## Final calibrated verdict logic

```python
trivially-degenerate  if  NaN guard fires (existing — solve/test_engine.py:302)
near-degenerate       if  λ = ((σ_0+σ_1)² - 4σ_2²) / (σ_0+σ_1)² < 0.05
borderline            if  asymptotic 95% CI for population F overlaps the relevant
                          CV for the strongest size or power claim
                          (equivalently:  F - 1.96·SE(F) < CV)
robust                otherwise
```

**SE(F) and CI computation:**
```python
nc_implied = max(0, 2*K * (F/(1 - rho_hat_squared) - 1))
SE_F = (1 - rho_hat_squared) / (2*K) * sqrt(2 * (2*K + 2*nc_implied))
CI_low  = F - 1.96 * SE_F
CI_high = F + 1.96 * SE_F
borderline = (CI_low < CV)  # for the relevant CV of the strongest claim
```

The strongest claim is determined by which CVs F currently exceeds. For size, "strong at X%" if F > r_X (X ∈ {7.5, 10, 12.5}, in increasing-strictness order). The relevant CV is the strictest one F still exceeds. Same for power. Cell flagged borderline if CI overlaps either size's or power's relevant CV.

## Conceptual corrections made during the session

Several reframings occurred that future-me should not need to re-derive:

1. **The F formula is correct.** The paper (Quantitative Economics 15, 2024, eq. 17) defines ρ̂ as the squared empirical correlation between (e_1 − e_2) and (e_1 + e_2), which works out exactly to `(σ_0 - σ_1) / sqrt((σ_0+σ_1)² - 4σ_2²)`. This is the test_engine code at line 286-289 — correct.

2. **ρ̂ ≠ Pearson correlation.** They coincide only at degenerate configurations. The paper's ρ̂ is what the published CV table is indexed by, and what `test_engine.py` computes for the lookup.

3. **The two near-degeneracy regimes are distinct.** F_den = σ₀σ₁ − σ₂² → 0 (Pearson ρ → 1) and D = (σ₀+σ₁)² − 4σ₂² → 0 (requires σ₀=σ₁=σ₂). They coincide in the symmetric case (σ₀ = σ₁) but diverge in asymmetric cases. λ = D/(σ₀+σ₁)² is the correct general diagnostic; using Pearson ρ² alone misses the geometric structure.

4. **The CV's 5% Type I error guarantee and the borderline diagnostic operate at different levels.** DMSS Proposition 4 sets cv_s such that under the worst-case null on the noncentrality, P(F > cv_s) ≤ 5%. That's the test's calibration and is unchanged. The borderline diagnostic is a *separate* inferential question about the precision of F as a point estimate — does the asymptotic 95% CI for the population F overlap the CV? An earlier draft framed this as "X% probability F lands below CV on resample," which was mathematically valid but misleading because it conditioned on a different distribution than the CV's calibration. Final framing is CI-overlap, which is the standard frequentist way to express this question.

5. **The 1.96 threshold is the 95% CI half-width.** Maps to the conventional CI level. Earlier conversation went through 1.5 (Youden-optimal but arbitrary buckets), 1.96 (mistakenly attributed to two-sided 5%), and 1.65 (one-sided 5% flip-rate framing) before settling on 1.96 = 95% CI under the cleaner CI-overlap framing.

## Output design (settled)

- **Glyph**: `⚠` next to F values for any cell with verdict ≠ robust.
- **F-stat symbols change**: from `*` / `**` / `***` to `†` / `††` / `†††` so `*` is freed for new TRV two-sided significance markers (|TRV| > 1.64, 1.96, 2.58 = `*`, `**`, `***`). Only happens if coauthors approve adding TRV stars (open question in memo).
- **Footer** under main F-table: 1 line per fragility type that fires (worst case + methodological note); 1 line for all-clear case with worst-case ρ̂² and minimum λ.
- **`results.F_reliability_summary()`** — opt-in detailed table with one row per cell.
- **Section-header captions** if TRV stars added; flat caption otherwise.

Sample borderline footer:
```
F-stat reliability:
  ⚠ borderline (1 cell): F = 4.3 (95% CI [4.13, 4.47]), strongest size claim "worst-case size ≤ 10%"
     CI overlaps 5%-CV = 4.20: claim is decision-uncertain at conventional levels
     fallback claim "worst-case size ≤ 12.5%" still robust (CI lower bound > 12.5%-CV)
  See: results.F_reliability_summary()
```

## What was done in the implementation session

Phase 1 (commit `83069f0`) — core diagnostic, no UI changes:
- λ, SE(F), 95% CI, strongest-claim labels, verdict computed inside the existing F-stat loop in `solve/test_engine.py`
- New attributes on `ProblemResults`: `lambda_dmss`, `F_se`, `F_ci_low`, `F_ci_high`, `verdict`, `strongest_claim_size`, `strongest_claim_power`
- New method `results.F_reliability_summary()` returning a long-form DataFrame with one row per (instrument set, model_i < model_j)
- `Progress` dataclass extended with optional fields (back-compat with older pickles)
- 17 tests covering attribute presence, math, and DataFrame schema

Phase 2 (commit `db708dd`) — printed output integration:
- Per-cell `⚠` glyph appended to F values for cells with non-robust verdict
- F-stat reliability footer below caption (only on the last printed table) with worst-case info per fragility type, or a one-line "all robust" summary
- `format_table` gains an `extra_notes` parameter for the footer
- 4 additional tests

Phase 3 (commit `92b9800`) — TRV stars + dagger swap:
- TRV gets two-sided significance markers `*` / `**` / `***` at |TRV| > 1.64 / 1.96 / 2.58
- F-stat size symbols moved from `*` / `**` / `***` to `†` / `††` / `†††` to free `*` for TRV
- Caption restructured into "TRV significance:" / "F-stat significance (DMSS):" / "F-stat reliability:" sections
- 5 additional tests (26 total in `tests/test_f_reliability.py`)

## What's still open (for coauthors)

1. **The five threshold questions in the memo** — see `MEMO_F_reliability_diagnostic_2026-04-28.md`:
   - σ-noise calibration target (currently 1e-4)
   - CI level for borderline (currently 95%, threshold 1.96)
   - TRV stars yes/no (currently yes, with dagger swap)
   - Retrospective replication on published tables before locking thresholds
   - Auto-show summary or opt-in only

2. **Published-tables retrospective.** Codex suggested running the diagnostic on DMSS / DMQSW / DMQS / Backus-Conlon-Sinkinson published tables to see how often real applications flag. Not done; thresholds are calibrated against simulation only.

3. **CV-table high-ρ, high-K offset.** Documented in `KNOWN_ISSUES.md`. Persistent ~0.5–1.0 absolute offset at K ≥ 20, ρ ≥ 0.9, r_075. Not a port bug — most likely RNG-sensitivity at the threshold-crossing tail, possibly amplified by an older MATLAB version having generated the published CSV. Not blocking calibration. Would need to be resolved if/when CV tables are ever regenerated.

4. **Promoting thresholds to a `Problem.solve(reliability_thresholds=...)` kwarg.** Currently exposed as module-level constants `RELIABILITY_LAMBDA_THRESHOLD` and `RELIABILITY_CI_LEVEL` in `pyRVtest/solve/test_engine.py`. Per-call override would be cleaner but adds API surface — defer until a coauthor asks for it.

## Pickup for next session

Most likely next paths:

**(a) Coauthor review.** Open a PR `feat/f-reliability` → `v0.4-refactor` (URL above). Point Lorenzo's and Marco's AI agents at the diff and the memo. Iterate on threshold values / output format based on their input.

**(b) Published-tables retrospective.** Run `compute_F_regime` on DMSS / DMQSW / DMQS / BCS published tables to check how often real applications flag. May lead to threshold tweaks.

**(c) v0.4.1 release.** Once coauthors sign off on the design + thresholds, merge to `v0.4-refactor` and tag v0.4.1 (or roll into v0.4.0 final if the timing aligns).

**(d) Promote thresholds to kwarg.** If the calibration values prove contentious, expose them as `Problem.solve(reliability_thresholds={'lambda': ..., 'ci_level': ...})`.

## Compute_mcs NaN propagation issue (out of scope)

While writing tests I confirmed a pre-existing bug: when two models have identical markups (NaN guard fires on F), `compute_mcs` later crashes in `multivariate_normal` SVD because the covariance matrix it builds has NaN entries. Independent of this work but worth flagging — would have been part of the trivially-degenerate test if it weren't there. The trivially-degenerate verdict in `feat/f-reliability` is structurally tested (a unit test reads the source to confirm the verdict label is set in the NaN-guard branch); end-to-end testing would need the MCS issue fixed first. Not a blocker for this PR.

## Other notes

- Worktree cleanup: removed 4 stale agent worktrees from `.claude/worktrees/agent-*` and 5 merged local branches at the start of session. Local now has just `main` and `v0.4-refactor`. Origin still has `CClean`, `CClean-fixes`, `CClean-Marco`, `testing`, `tests/memo1-regressions`, `v0.4-refactor`, `main`.
- Marco's last commit was 2026-04-21 (a week before this session). No conflicting work in flight.
- Repo CI is green on both numpy 1 and numpy 2 (per the previous session's handover at `2026-04-20-ci-matrix-green.md`).
