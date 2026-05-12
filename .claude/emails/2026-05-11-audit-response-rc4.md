# Email draft — audit response, rc4 tagged

**To:** [Auditor]
**From:** Christopher Sullivan
**Subject:** pyRVtest audit — actionable items addressed, rc4 tagged

Draft below. Copy-paste into your client; tweak tone / cut as needed.

---

Hi [Auditor],

Thanks again for the thorough audit. The actionable items are now addressed and `v0.4.0rc4` is tagged on `v0.4-refactor`. Below is the per-item ledger and a couple of items we've deliberately deferred with reasons.

Branch: `v0.4-refactor` at `32de263` on https://github.com/anyatarascina/pyRVtest. Tag: `v0.4.0rc4` at `32de263`. Install target (not yet on PyPI):

```
pip install git+https://github.com/anyatarascina/pyRVtest@v0.4.0rc4
```

## What we did

**1. Dearing, Magnolfi, Quint, Sullivan, and Waldfogel — citation year (2026 → 2024).** You were right; NBER WP 32863 has issue date August 2024. Swept every citation in source, docs, tests, and CHANGELOG from "(2026)" to "(2024)" — 17 real files touched. `references.rst` and `README.rst` gained the "August 2024" suffix on the NBER citation. (We left `.claude/handovers/`, `.claude/plans/`, `.claude/emails/` untouched — those are timestamped session records and are factually correct as of the date they were written.) Note this only affects DMQSW; the unrelated DMQSS paper (Duarte, Magnolfi, Quint, Sølvsten, Sullivan — "Conduct and Scale Economies") is genuinely 2026 and was left as-is. Commit `cc22d89`.

**2. Python support metadata vs CI matrix.** `setup.py` claimed `python_requires='>=3.7'`. CI exercises Python 3.11 only — 3.7 and 3.8 are EOL and we don't test them. Tightened the floor to `>=3.9` with a comment cross-referencing `.github/workflows/ci.yml`. Commit `cc22d89`.

**3. K > 30 instruments fallback — regression coverage.** We already had `UserWarning` tests proving the alert fires (`test_k_gt_30_warning.py`, three tests). Added a new `TestKGreaterThan30Regression` class with value-level checks:

- F-stat values are finite and non-negative on the off-diagonal pair entries at `K_effective ∈ {31, 33, 40}` (parametrized).
- Critical-value rows from `F_cv_size_list` exactly match an entry in the K=30 row of the published table at the same rho — i.e., the fallback actually consults the K=30 row, as designed.
- `ProblemResults.summary_df()` renders without error at K=33.

Eight K>30 tests now, all green. Commit `cc22d89`.

## Test suite

Full suite green on numpy 1.26.4 / pyblp 1.1.2 (the matrix variant CI runs by default). The Dearing-touched modules — `test_dearing_models`, `test_import_roundtrip`, `test_passthrough_numerical`, `test_passthrough_summary` — re-run clean at 105/105.

## Deferred (called out for transparency)

A few items from the audit we're carrying to v0.5 rather than rolling into rc4, with reasoning:

- **PyPI alignment.** PyPI still has v0.3.x. We're deliberately holding the v0.4 PyPI release until `v0.4-refactor` settles after one more coauthor pass; rc-tags pin everything in the meantime. The install-from-tag command above is the current canonical path.
- **ReadTheDocs alignment.** The public RTD build still tracks `main` (v0.3.x). Once v0.4 final lands on PyPI we'll point RTD at v0.4. For now the canonical docs are the ones at HEAD on `v0.4-refactor`.
- **NumPy 2 `analytical_scale` xfail.** This is the ~3% F-stat shift on the `test_snapshot_analytical_scale` snapshot under numpy 2 / pyblp 1.2 that we tracked from rc1. The shift is from upstream LAPACK changes in numpy 2 and is below DMSS's plug-in CV noise floor; we'd rather understand the root cause than re-baseline the snapshot. Tracked as the second matrix-variant xfail in CI.

If anything in the ledger looks short, send back a follow-up and we'll iterate before cutting v0.4 final.

Chris
