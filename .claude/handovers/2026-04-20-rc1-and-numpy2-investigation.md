# Handover — v0.4.0rc1 shipped + post-rc1 polish + numpy 2 F-shift investigation

**Date:** 2026-04-20
**Branch:** `v0.4-refactor` at `de9104b` (12 commits ahead of pre-session anchor `2370e76`).
**Tag:** `v0.4.0rc1` at `cdf4781` (pushed).
**Test counts:**
- numpy 1.26.4 / pyblp 1.1.2 (local): **633 passed + 3 skipped**
- numpy 2.0.2 / pyblp 1.2.0 (fresh venv): **631 passed + 4 skipped + 1 xfailed**
- mypy `--strict` clean across 44 source files.

## What this session produced

Twelve commits addressing Lorenzo's 2026-04-18 break-it findings, Marco's follow-up email items, plus a deep investigation of the residual numpy 2 F-shift. Ordered chronologically:

1. **`679dbd9` FIX:** `ModelFormulation(user_supplied_markups=..., ownership_downstream=...)` crash. Added `UserSuppliedMarkups` first-class class (Marco's path), adapter routing, 14 regression tests at `tests/test_adapter_user_supplied_markups.py`. Unblocks carRV's production `conduct_test.py`.
2. **`44bb83b` REL:** version bumped to `0.4.0rc1`, `pyproject.toml` added. `pip install -e .` verified under pip 26.
3. **`cdf4781` FEAT:** K>30 critical-value UserWarning in `test_engine.py`; `options.digits` wired through `to_markdown` / `to_latex` / `summary_df`; `options.verbose` deprecated via PEP 562 `__getattr__`.
4. **`137f4e2` DOC:** CHANGELOG qualified — analytical nested-logit Hessian applies to plain logit and single-scalar-rho nested logit only; per-nest rho, multi-level nesting, and BLP continue through pyblp finite-diff.
5. **`b2ab59a` DOC:** Labor API labeled experimental in README, AGENTS.md, and `docs/agent_guide.rst`. Fixed AGENTS.md drift: stale HHI reference, stale 357→619 test count, v0.6→v0.7 on per-model tax deprecation.
6. **`d13eb2d` FIX:** `Problem(market_side='labor', demand_params=...)` now raises clean `NotImplementedError` pointing at v0.5; was a silent wrong-sign trap. Tax-spec tiebreaking documented in `docs/migrating_to_v0.4.rst` with pinning test.
7. **`22a13a1` FIX:** `PanelResults` roster-signature validation (audit B1 / Lorenzo P1 #4). Panels mixing `[Bertrand, PerfectCompetition]` and `[PerfectCompetition, Bertrand]` used to mislabel rejection-rate tables silently; now raise at construction with a named first-diverging position.
8. **`f4b39fc` DOC:** MEMO "no correctness changes this session" framing qualified (Lorenzo methodology note); migration guide gets a `known_coefficients` salience-opt-out note.
9. **`e4ff40d` TEST:** Share-weighted `Monopoly ≥ Bertrand` fallback pinned as theory-guaranteed alternative to the element-wise assertion (Lorenzo methodology note).
10. **`176d691` FEAT:** Per-model tax `DeprecationWarning` moved from solve time to `ConductModel.__init__` / `Vertical.__init__`. `pyRVtest.Bertrand(unit_tax='col')` now warns on construction; previously silent until `Problem()`.
11. **`880cc6b` FIX:** numpy 2 compatibility — jinja2 added to requirements (pandas 2.3 needs it for `to_latex`); NaN-rho guard in the critical-value lookup (salience test crash); `analytical_scale` snapshot xfailed under numpy >= 2.0 with bisect evidence.
12. **`de9104b` DOC:** `analytical_scale` xfail reason upgraded to full root-cause diagnostic after instrumentation traced the shift to QR basis differences × near-degenerate sigma geometry.

All pushed to `origin/v0.4-refactor`. CI remains red for the deferred numpy 2 item (pre-existing, not caused by this session — see below).

## numpy 2 F-shift investigation (in depth)

Spent a focused session diagnosing the last Lorenzo P0 item — a 3% shift in `F[0][0][1]` on `tests/test_snapshots.py::test_snapshot_analytical_scale` between numpy 1.26 and numpy 2.0. All other snapshots remain bit-identical cross-version.

### Bisect

- numpy 1.26.4 + scipy 1.12.0: snapshot bit-identical to stored value (1.0316662590067232).
- numpy 1.26.4 + scipy 1.13.1: still bit-identical. scipy 1.13 alone not the cause.
- numpy 2.0.2 + scipy 1.13.1: F = 0.9980489511445025. ~3% relative shift.

**Conclusion: numpy 2 is the sole trigger.** Most likely mechanism is that the numpy 2 wheel ships a different bundled OpenBLAS / LAPACK build than the numpy 1.26 wheel (numpy 2.0 released April 2024, numpy 1.26 October 2023). `np.linalg.qr` dispatches to LAPACK's dgeqrf, and different LAPACK builds use different Householder reflection sequences / block sizes for BLAS-3 QR. The resulting Q matrices are equally orthonormal but differ at sub-ULP level, which propagates through downstream nonlinear operations.

### What the instrumentation revealed

Traced the numerical divergence through the pipeline. Key observations:

- DGP output (`_build_scale_dgp` result) is bit-identical across numpy versions (`default_rng` is stable by numpy policy).
- First-stage QR in `pyRVtest/solve/endogenous_cost.py::iv_correct` produces a different `Q_fs` on numpy 2 even though the input `first_stage_X` is byte-identical. The projector `Q_fs @ Q_fs.T` applied to `endog_col` shifts `endog_hat` at the 1e-13 level.
- Downstream: `controls = hstack([w_absorbed, endog_hat_absorbed])` has condition number ~3582 (near-collinearity between `w` and `endog_hat`). The QR residualization on this matrix amplifies the 1e-13 input shift into a ~1% shift in `omega` residuals.
- Sigmas (variance decompositions of `phi` blocks): shift by ~0.1% to 0.8% relative.
- F formula: `F = (1 - ρ²) · unscaled_F = (1-ρ²) · N/(2K) · F_num / (σ₀σ₁-σ₂²)`. In this fixture all three sigmas are nearly equal (~2.07e-3), so `F_den = σ₀σ₁-σ₂² ≈ 4e-8` is in the catastrophic-cancellation regime. The ~1% σ shift is amplified to a 3% F shift by the near-degenerate geometry.
- TRV is self-normalizing (`(Q[i] - Q[m]) / sqrt(var)`) so its cross-version shift is ~9e-15 (machine precision).

### Why F, specifically

F's denominator is a 2×2 determinant of the moment-condition covariance matrix. It vanishes precisely when the two candidate models produce nearly identical moment conditions given Z — the weak-instrument regime the F-stat is designed to diagnose. **F is inherently numerically sensitive in the regime it's built to flag.** This fixture puts F in that regime by pairing Bertrand with perfect competition and relying on the `log_quantity` IV correction to absorb the markup difference; after the correction the two models are near-equivalent in moments.

Sanity check against CV tables: at K=1, ρ≈0.08 (this fixture), size critical values are all zero (low-ρ regime); power critical values are 17.4 / 21.7 / 29.0. Both F=1.03 (numpy 1) and F=0.998 (numpy 2) fall below all power thresholds (`" "` = weak power) and trivially beat all size thresholds (`"***"` = strong size). The 3% shift does not change the reported star classification **for this fixture**. Could flip stars for a fixture with F near a threshold — not hypothetical but not currently tripped.

### Fixes tried (all empirical, all failed or near-failed)

| Attempted fix | Cross-version F shift | Outcome |
|---|---|---|
| Candidate A — revise DGP to avoid catastrophic cancellation (best variant) | 0.013% | Still not 1e-10 |
| Candidate B — SVD projection instead of QR in `qr_residualize` | 3.2% | Doesn't help |
| Algebraic rewrite `F = 2N/K · F_num / denom_sq²` (removes F_den cancellation) | 3.3% | Doesn't help (F_num not the source) |
| Sign-canonicalize Q via `diag(R) ≥ 0` | 3.3% | No-op (signs cancel in `Q @ Q.T`) |

The empirical finding: **no Python-level algorithmic change closes the cross-version gap.** The numpy 2 σ shift is ~1% on our near-collinear controls matrix, which any nonlinear downstream math inherits. SVD projection doesn't help because SVD also goes through LAPACK with the same version sensitivity. Sign canonicalization is mathematically a no-op because the projector `Q @ Q.T` is sign-invariant.

### Leaning on resolution

The investigation concluded that `atol=1e-10` was always implicitly a single-pinned-LAPACK-version guarantee. This matches the scientific-Python convention: scipy's snapshot suite, pytorch's tests, etc. typically use `rtol≈1e-6` or `atol≈1e-5`, not `atol=1e-10`, precisely because LAPACK backends change across wheel builds. The status quo xfail is accurate but fragile — every future numpy release ships new LAPACK, and we'd be chasing xfail updates indefinitely.

**Recommended path (not yet executed, pending Chris's decision):**

1. **Tier the snapshot atol per field.**
   - Linear / pass-through fields (`markups` from `user_supplied_markups`, prices, shares): keep `atol=1e-10`.
   - Self-normalizing fields (`TRV`, `Q`, `g`): `atol=1e-8`.
   - Nonlinear-amplification fields (`F`, `sigma`, `rho`, symbols): `atol=1e-3` or per-field looser.
2. **Teach `tests/_snapshot_helpers.py::assert_snapshot`** to accept `field_atols: dict` override.
3. **Drop the xfail** on `analytical_scale` — it becomes a normal passing test under the tiered atol.
4. **Two-version CI matrix** — numpy 1.26 + numpy 2.x jobs on every push. Same tiered atol applied to both; both must go green.
5. **Document** the numerical-precision tier system in `docs/agent_guide.rst` (or new `docs/numerical_precision.rst`) so users understand what reproducibility pyRVtest promises (1e-10 intra-version on linear fields; 1e-3 to 1e-8 cross-version).

Estimated cost: 1-2 hours. Turns the numpy 2 sore into a formalized cross-version guarantee. Paused here for Chris's decision.

**Alternative (status quo):** keep xfail. Accurate but would need revisiting every time numpy ships new LAPACK. Also doesn't help users in production who hit the near-degenerate regime.

### How to reproduce the investigation

```bash
python3 -m venv /tmp/rvt-np2
/tmp/rvt-np2/bin/pip install --upgrade pip --quiet
/tmp/rvt-np2/bin/pip install -e . pytest hypothesis jinja2 --quiet
/tmp/rvt-np2/bin/python -m pytest tests/test_snapshots.py::test_snapshot_analytical_scale -v
# expected: 1 xfailed

# full sweep
/tmp/rvt-np2/bin/python -m pytest -q
# expected: 631 passed + 4 skipped + 1 xfailed
```

Scratch scripts used during investigation (not committed): `/tmp/dgp_diff.py`, `/tmp/iv_diff.py`, `/tmp/sweep_scale_dgp.py`, `/tmp/cross_test.py`, `/tmp/f_stable.py`, `/tmp/sign_canon_test.py`. All demonstrate the same bisect: numpy 2 alone shifts F, and nothing algorithmic on our side fixes it.

## Outstanding items by owner

### Chris (maintainer decisions)

1. **Resolve the `analytical_scale` xfail.** Three options on the table (tiered atol / keep xfail / fixture revision). Lean: tiered atol (1-2 hour implementation). See "Leaning on resolution" above.
2. **Tag `v0.4.0` final after Step 16 lands clean.** Step 16 (AFSSZ dogfood on the 910-market-year panel) is data-blocked ~1 week. After that, if no API adjustments are needed, tag `v0.4.0` and drop rc1 aliasing.
3. **DMQSW blast-radius provenance email.** Lorenzo raised: were Table 4 / Table 5 numbers in the current DMQSW submission computed before or after `b3b08a3` (W-vs-updated_W fix)? Outreach to Marco / Sal / Waldfogel to check commit-at-time-of-estimation. Not a code task; coauthor coordination.
4. **`Problem(demand_backend=...)` public kwarg** (audit B3 / Lorenzo P1 #5). Still deferred. Users are currently routed through the private `_compute_markups` for non-pyblp demand. Design + implement + tests. Low priority if no user has asked.
5. **Audit post-tag queue:** `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, replication suite anchors, machine-readable exports (`to_dict`, `to_json`). v0.4.0-final or v0.5.

### Marco (handoffs from Lorenzo, still open)

1. **DMSS yogurt golden file** (Step 0d). Scaffold at `tests/replication/test_dmss_yogurt.py`. Needs data path + pinned paper spec + expected TRV/F/MCS values + tolerance.
2. **Labor sign-convention review** on `pyRVtest/models/labor.py`. `Monopsony` / `BertrandWages` / `CournotEmployment` implement sign-flipped FOCs; pairwise consistency verified, absolute convention against labor-market-conduct manuscript not. Flag any discrepancies before v0.5 activates `LaborSupplyBackend`.
3. **`offdiag_frobenius` metric vs Dearing Remark 4** cross-check. Currently implemented as `‖P_i − P_j − diag(diag(P_i − P_j))‖_F`. Verify this operationalizes Remark 4's distinguishability condition (or propose renaming if stricter).
4. **`Results.summary` investigation** (Marco's own item). Awaiting his findings. Only `summary_df` exists today; if Marco is calling `.summary()` it's `AttributeError`. He said he'd check.

### Lorenzo (break-it follow-up)

1. Re-run `.claude/breaks/probe_01..07.py` from his `tests/memo1-regressions` branch against `v0.4.0rc1` to confirm the carRV crash is fixed on his machine.
2. Sign off on carRV migration; possibly extend `probe_05_v03_vs_v04_parity.py` to exercise `endogenous_cost_component` + `demand_adjustment=True` once DMQSW data becomes available.

### Coauthor-group discussion

1. **DMQSW provenance** (under Chris above).
2. **v0.4.0 tag timing** — rc1 is shipped; final tag depends on Step 16 + above decisions.

## Session-specific notes for the next session

- The v0.4.0rc1 tag is at `cdf4781`. Do NOT move it — post-rc1 commits (`137f4e2`, `b2ab59a`, `d13eb2d`, `22a13a1`, `f4b39fc`, `e4ff40d`, `176d691`, `880cc6b`, `de9104b`) will be in v0.4.0 final.
- CI was already red for the numpy 2.x P0 item before this session started. Our commits didn't introduce new failures; the pre-existing numpy 2.x test failures are now addressed (8 of 9 fixed, 1 documented xfail with root cause).
- Every commit has a structured message with file-level detail — see `git log v0.4.0rc1^..HEAD` for the full provenance.
- The investigation demonstrated convincingly that the 3% cross-version shift is a property of LAPACK builds, not of pyRVtest's algorithm. Future diagnostic questions about numpy-2 differences should bisect first (scipy version vs numpy version) before spending time on library-side fixes.

## File-level landmarks

- New: `pyRVtest/models/user_supplied.py`, `pyproject.toml`, `tests/test_adapter_user_supplied_markups.py`, `tests/test_k_gt_30_warning.py`, `tests/test_options.py`, `tests/test_k_gt_30_warning.py`.
- Modified significantly: `pyRVtest/models/base.py` (per-model tax warn), `pyRVtest/models/vertical.py` (same), `pyRVtest/models/_adapter.py` (routing + ValidationError), `pyRVtest/results/panel.py` (roster hash), `pyRVtest/results/_format.py` (options.digits wiring), `pyRVtest/solve/test_engine.py` (NaN-rho guard + K>30 warning), `pyRVtest/options.py` (PEP 562 verbose deprecation), `pyRVtest/problem.py` (labor+demand_params guard, conflict warning still lives here), `pyRVtest/__init__.py` (UserSuppliedMarkups export).
- Docs: `CHANGELOG.md` (rc1 section + deferred list), `README.rst` (labor experimental), `AGENTS.md` (drift sweep), `docs/agent_guide.rst` (labor experimental + deprecation timeline), `docs/migrating_to_v0.4.rst` (tax tiebreaker + known_coefficients note), `MEMO_coauthor_updates.md` (methodology framing qualification).
