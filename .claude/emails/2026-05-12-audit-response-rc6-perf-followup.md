# Email draft — rc6 supplement, PT performance

**To:** [Auditor 1]
**From:** Christopher Sullivan
**Subject:** pyRVtest rc6 — Audit 1 Finding 4 (PT diagnostic performance) addressed

Draft below. Copy-paste into your client; tweak tone / cut as needed.

---

Hi [Auditor],

Quick supplement to my rc5 ledger: I had filed your Finding 4 (PT diagnostics slow + uncached) as "v0.5 work," but on closer reading the fix is small and well-defined, so we did it in rc6 instead of deferring.

Tag: `v0.4.0rc6` at `077654d`. Install:

```
pip install git+https://github.com/anyatarascina/pyRVtest@v0.4.0rc6
```

## Finding 4 — what we did

Your call-graph analysis was correct: `compute_passthrough_summary` calls `_perturb_and_build_markups` once, then calls `build_passthrough(model_index=m)` for each model — and each inner call re-assembles markups for *all* models. For `n_models` candidates the markups stage ran `1 + n_models` times. `compute_instrument_channels` had the same pattern.

The patch adds a private `_precomputed_markups=(markups_full, markups_downstream)` parameter to `build_passthrough` that suppresses its own assembly call when supplied. Both diagnostics now compute markups once and thread the tuple through every per-model `build_passthrough` call.

**Measured on the shipped 3000-market synthetic, 4 candidate models:**

| Call | rc5 baseline | rc6 | Reduction |
|---|---|---|---|
| `passthrough_summary` | 12.5s, 5 calls | 6.0s, 1 call | ~½ wall time |
| `instrument_channels` | 11.6s, 5 calls | 5.1s, 1 call | ~½ wall time |

(My Python 3.9 / numpy 1.26 / pyblp 1.1.2 stack; your Python 3.13 / numpy 2.4 stress stack saw ~50s because pyblp's older LAPACK paths under numpy 2 are themselves slower. The ratio should be comparable.)

**Numerical equivalence:** values are bit-identical. Spot-checked all four DMQSW feature distance metrics (`offdiag_ratio`, `full_pass`, `row_sum`, `level_adj`) on the (Bertrand, Cournot) pair against an explicit no-hook reconstruction; difference is 0 on every column.

**Regression coverage:** two new tests in `tests/test_passthrough_summary.py::TestMarkupsAssemblyCallCount` pin the call count at exactly 1 across both diagnostics. If a future change inadvertently re-introduces a redundant rebuild, CI catches it.

**Public API:** unchanged. The new parameter is underscore-prefixed (`_precomputed_markups`) and documented as an internal optimization hook with explicit "public callers should leave this at None" language. We can promote it to a stable name in v0.5 if it turns out to be useful externally; for now the only consumers are the two high-level diagnostics.

## What I'm not claiming this fixed

- The PT diagnostics are still substantially heavier than `Problem.solve()`. The cache hook removes the dominant *quadratic-in-n_models* term, but the per-model `compute_passthrough_numerical` calls (central-difference Jacobians) still dominate for non-Vertical conduct. v0.5 will look at whether those can be batched or short-circuited further.
- The slow-test problem in the test suite organization (Audit 1 Finding 3) is not addressed by this patch. PT tests are still ~5 minutes; we should still mark them `@pytest.mark.slow` and split the CI runs. That's the next item in the queue.

Chris
