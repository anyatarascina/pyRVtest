# Email draft — v0.4.0rc1 tagged + numpy 2 investigation

**To:** Lorenzo Magnolfi, Marco Duarte
**From:** Christopher Sullivan
**Subject:** pyRVtest v0.4.0rc1 tagged — your findings addressed + one open question

Draft below. Copy-paste into your client; tweak tone / cut as needed.

---

Hi Lorenzo, Marco,

Quick update on pyRVtest. The items both of you flagged in the 2026-04-18 break-it pass and follow-up email are now addressed, and `v0.4.0rc1` is tagged.

Branch: `v0.4-refactor` at `c04d555` on https://github.com/anyatarascina/pyRVtest. Tag: `v0.4.0rc1` at `cdf4781`. Install target (not yet on PyPI):

```
pip install git+https://github.com/anyatarascina/pyRVtest@v0.4.0rc1
```

**Test suite (numpy 1.26.4 / pyblp 1.1.2):** 633 passed + 3 skipped, up from 592+3 pre-session. 41 new regression tests pinning the fixes. mypy `--strict` clean.

**Test suite (numpy 2.0.2 / pyblp 1.2.0):** 631 passed + 4 skipped + 1 xfailed. The xfail is `test_snapshot_analytical_scale` — see below.

## What landed

The full ledger with per-commit detail is in `MEMO_coauthor_updates.md` at the top of the branch. Highlights:

**From Lorenzo's break-it P0s:**

- carRV crash fixed. `ModelFormulation(user_supplied_markups='col', ownership_downstream='firm_ids')` now translates to a new first-class `UserSuppliedMarkups` class (picked Marco's "its own class" recommendation over my earlier adapter-only patch). TRV / F bit-identical to `Bertrand(user_supplied_markups='col')`. Commit `679dbd9`. **Please re-run your `probe_01..07.py` scripts against rc1 to confirm on your end.**
- Version metadata + `pyproject.toml`. `pip install -e .` works under modern pip now. Commit `44bb83b`.
- Cross-platform numpy 2 test failures: 8 of the 9 resolved — jinja2 added to requirements (pandas 2.3 needs it for `to_latex`), NaN-rho guard in the critical-value lookup for degenerate salience-test pairs. The ninth (the 3% F shift on `analytical_scale`) is the xfail I'll discuss below. Commit `880cc6b`.

**From Lorenzo's P1 + methodology:**

- `PanelResults` roster-signature check: panels mixing `[Bertrand, PC]` and `[PC, Bertrand]` children used to mislabel the rejection-rate table silently; now raise at construction naming the diverging position. Commit `22a13a1`.
- K > 30 critical-value warning. `UserWarning` fires once per instrument set when K exceeds the tabulated range. Commit `cdf4781`.
- Per-model tax `DeprecationWarning` moved from solve time to `ConductModel.__init__` — `pyRVtest.Bertrand(unit_tax='col')` now warns on construction (was silent until `Problem()`). Commit `176d691`.
- Per-nest rho CHANGELOG qualifier: analytical Hessian applies to plain logit + single-scalar-rho nested only; per-nest rho / BLP stay on pyblp finite-diff. Commit `137f4e2`.
- Tax precedence documented with pinning test — per-model wins with two DeprecationWarnings (generic + conflict-specific). Commit `d13eb2d`.
- Labor API labeled experimental in README + AGENTS.md + agent guide. AGENTS.md drift swept (stale HHI reference, 357→619 test count, v0.6→v0.7 on per-model tax deprecation). Commit `b2ab59a`.
- `Problem(market_side='labor', demand_params=...)` now raises clean `NotImplementedError` pointing at v0.5 — was a silent wrong-sign trap. Commit `d13eb2d`.
- Share-weighted `Monopoly ≥ Bertrand` fallback pinned as theory-guaranteed alternative to the element-wise assertion. Commit `e4ff40d`.

**From Marco's follow-up:**

- `UserSuppliedMarkups` first-class class (your suggestion over my adapter hack). Commit `679dbd9`.
- `options.digits` now actually works — wired through `to_markdown`, `to_latex`, `summary_df`. `options.verbose` formally deprecated via PEP 562 `__getattr__`. Commit `cdf4781`.
- K > 30 warning added (Commit `cdf4781`).
- Your `Results.summary` item: still waiting on your investigation. Only `summary_df` exists today; if `.summary()` is the call, it's an `AttributeError`. Let me know what you find.

## The open question: numpy 2 F-shift on `analytical_scale`

The one Lorenzo P0 that isn't fully closed. Spent a focused session diagnosing it. Short version:

`F[0][0][1]` on `test_snapshot_analytical_scale` is 1.0317 on numpy 1.26 and 0.9980 on numpy 2.0 — ~3% shift. All other snapshots are bit-identical cross-version. The chain is:

1. numpy 2.0 ships a newer bundled OpenBLAS / LAPACK than numpy 1.26, so `np.linalg.qr` uses different Householder reflections. The resulting Q matrices are equally orthonormal but span subtly different approximate column spaces (at the `ε × κ` level).
2. The `controls = hstack([w, endog_hat])` matrix in the endogenous-cost path has condition ~3580 (near-collinearity), which amplifies the LAPACK difference into a ~1% shift in the three `sigma` values driving F.
3. This specific fixture (Bertrand vs PerfectCompetition with `log_quantity` IV correction absorbing the markup difference) produces nearly-equivalent moment conditions for the two models, so `F_den = σ₀σ₁ - σ₂²` is in the catastrophic-cancellation regime (~4e-8). The ~1% σ shift gets amplified to a 3% F shift.

Four algorithmic fixes tried on my side, all empirically failed (SVD projection, stable-form F rewrite, sign-canonicalization of Q, DGP revision). Nothing closes the gap because the 1% upstream σ shift is already orders of magnitude beyond `atol=1e-10`. **This is the LAPACK-build-dependence issue that scipy and pytorch pin explicitly.** TRV is unaffected (shift ~9e-15, machine precision) because its self-normalizing structure cancels the LAPACK noise — F doesn't have that cancellation.

The theoretical framing I keep coming back to: **F's denominator is a 2×2 determinant of the moment-condition covariance, which vanishes when the two candidate models produce nearly identical moments given Z — i.e., the weak-instrument regime the F-stat is designed to diagnose.** So F is inherently numerically sensitive in exactly the regime it flags. On our fixture this is academic (both F=1.03 and F=0.998 fall in the same star bucket — `***` for size, blank for power at K=1, ρ≈0.08, so the user conclusion doesn't change). But for a fixture with F near a power CV threshold at high ρ, the numerical instability could flip a star classification across numpy versions.

**My leaning:** adopt the standard scientific-Python approach — tier the snapshot `atol` by field (linear / pass-through at 1e-10, self-normalizing like TRV/Q/g at 1e-8, nonlinear-amplification like F/σ/ρ at 1e-3), run CI on both numpy 1 and numpy 2, document the tier system. About 1-2 hours of work. Drops the xfail, makes cross-version reproducibility explicit. Holding on this pending your thoughts.

**Question for the paper:** does the theoretical sensitivity point deserve a brief sentence in the DMSS text? Something like "numerical precision of the F-stat degrades near the degenerate boundary; the critical-value machinery is designed for this, but users should interpret F classifications as indicative rather than exact near the boundary." Or is this already implicit in the weak-instrument language and we don't need to call it out explicitly?

## Still on deck

**Marco — open handoffs from Lorenzo (still):**

- DMSS yogurt golden file (Step 0d). Need data path + pinned spec + expected values.
- Labor sign-convention cross-check on `pyRVtest/models/labor.py` before v0.5 activates `LaborSupplyBackend`.
- `offdiag_frobenius` metric — does it operationalize Dearing Remark 4 exactly as written?

**Coauthor-group question:**

- **DMQSW provenance.** Lorenzo flagged that DMQSW runs `demand_adjustment=True` on the pyblp path, which is where Bug 3 (W vs updated_W) lived pre-`b3b08a3` (2026-04-16). If the Table 4 / Table 5 numbers in the current submission were computed on a pre-`b3b08a3` pyRVtest with over-identified demand, they were affected. Can one of us check the commit-at-time-of-estimation and see whether a rerun against v0.3.3-stable or later is warranted before the next revision? Loop in Sal and Joel as relevant.

**Me:**

- Resolve the `analytical_scale` xfail per above (pending your reactions).
- Tag `v0.4.0` final after Step 16 (AFSSZ dogfood on the 910-market-year panel) lands. Data-blocked ~1 week.
- Audit post-tag queue: `Problem(demand_backend=...)` public kwarg, `CITATION.cff`, `CONTRIBUTING.md`, replication-suite anchors, machine-readable exports. All v0.4.0-final or v0.5 scope.

Take your time with this one. rc1 is installable as-is; v0.4.0 final tag waits for Step 16 and coauthor sign-off.

Best,
Chris
