# Email draft — rc15, rc14 re-audit follow-through

**To:** [Auditor]
**From:** Christopher Sullivan
**Subject:** pyRVtest rc15 — all four rc14 re-audit findings addressed

Draft below.

---

Hi [Auditor],

Thanks for the rc14 re-audit. All four findings landed in `v0.4.0rc15`. The two release blockers (typing_extensions, doctest) are fixed; the methodology finding on `offdiag_ratio` is fixed with the NaN-when-denominator-degenerate behavior you suggested; and `passthrough_reliability` now reuses the rc6-rc11 caches so it runs in roughly the same wall time as the other PT methods.

Tag: `v0.4.0rc15` at `71740d8`. Install:

```
pip install git+https://github.com/anyatarascina/pyRVtest@v0.4.0rc15
```

## Per-finding ledger

### Finding 1 (P1 release blocker): typing_extensions missing

Confirmed and fixed. 10+ runtime modules import `from typing_extensions import TypeAlias` (the package's Python floor is 3.9, where `typing.TypeAlias` doesn't exist yet). `requirements.txt` now pins `typing_extensions>=4.0`.

```
# requirements.txt (excerpt)
jinja2>=3.0
# typing_extensions provides TypeAlias for Python 3.9 (stdlib typing.TypeAlias
# is 3.10+). Imported unconditionally across pyRVtest/backends/*, pyRVtest/solve/*,
# pyRVtest/models/* — required at runtime, not just for type-checking.
# Caught by rc14 audit (Finding 1).
typing_extensions>=4.0
```

### Finding 2 (P1 release blocker): doctest collection fails

Confirmed your reproducer and fixed. The rc14 docstring had `>>> # df = problem.passthrough_reliability()  # doctest: +SKIP` — the `+SKIP` directive was attached to a comment-only line, which doctest rejects at collection. Removed the leading `#` so it's a real `SKIP`-decorated statement:

```python
>>> df = problem.passthrough_reliability()  # doctest: +SKIP
>>> df[df.pt_status != 'robust']            # doctest: +SKIP
```

`pytest --doctest-modules pyRVtest -q` now passes. Added `tests/test_doctest_gate.py` that runs the doctest gate as a subprocess in the regular tests/ suite, so the next time someone introduces a malformed docstring it surfaces locally instead of in CI.

### Finding 3 (P1 methodology): offdiag_ratio masks denominator degeneracy

This was the most important one — it closed the gap in my rc14 reliability work. You're right that the rc14 `pt_status` vocabulary only covered ill-conditioned inverses (Audit 2 C3 cases 3-4) and missed denominator degeneracy in the feature metric itself (case 2). Your repro made this crisp:

```python
P1 = np.array([[0.0, 1.0], [1.0, 0.0]])
P2 = np.array([[0.0, 2.0], [2.0, 0.0]])
# pre-rc15: _metric_offdiag_ratio returns 0.0 (wrong: matrices differ)
# rc15:     returns np.nan
```

Both `_metric_offdiag_ratio` (scalar) and `_metric_offdiag_ratio_batched` now return NaN per market when any column has `|diag(P)| <= eps_diag` in either matrix (default `eps_diag=1e-12`, configurable kwarg).

The aggregation in `compute_passthrough_summary` was also updated:
- Uses `np.nanmedian` instead of `np.median`, so well-defined markets aggregate cleanly when only some markets are degenerate.
- Adds a new column `offdiag_ratio_n_degenerate` per pair, counting markets where the metric was undefined. Users now have an explicit signal to interpret a small/zero median.

Three new regression tests pin this:
- `TestOffdiagRatioDenominatorDegeneracy.test_auditor_repro_returns_nan` — your exact (P1, P2)
- ditto `test_batched_auditor_repro_returns_nan`
- `test_well_defined_case_unchanged` — non-degenerate inputs unchanged
- `test_eps_diag_kwarg_threshold` — tolerance configurable

The other three feature metrics (`full_pass`, `row_sum`, `level_adj`) don't have this issue — they're norms of differences, no denominator. Only `offdiag_ratio` was affected.

### Finding 4 (P2 performance): reliability bypasses caches

Confirmed your timing. The pre-rc15 path called `build_passthrough(problem, model_index=m, market_id=...)` without the `_precomputed_markups` / `_precomputed_demand_derivatives` / `_precomputed_market_indices` kwargs, so every model re-ran the entire upstream caching chain.

rc15 threads all three caches through. New timing on the shipped synthetic:

| Workflow | rc14 | rc15 |
|---|---|---|
| `passthrough_reliability()` (all markets, 4 models) | 5.81s | **0.95s** |

In the same ballpark as `passthrough_summary` (0.50s) and `instrument_channels` (0.85s) now. Pinned with `TestPassthroughReliabilityPerformance.test_reliability_calls_markups_assembly_once` asserting the call count is exactly 1.

## Correctness invariant

The bit-identical-vs-rc5 parity invariant we've maintained across rc6 → rc14 still holds at rc15. On the shipped synthetic:

| Quantity | max |Δ| (rc5 vs rc15) |
|---|---|
| TRV (constant-cost) | 0 |
| F (constant-cost) | 0 |
| markups | 0 |
| MCS p-values (constant-cost) | 0 |
| TRV (demand-adjusted) | 0 |
| F (demand-adjusted) | 0 |
| MCS p-values (demand-adjusted) | 0 |

The offdiag_ratio NaN behavior change only triggers on denominator-degenerate inputs, which don't occur in the shipped synthetic. So the invariant survives all 15 rcs.

## Test suite

**853/853 in 227s** at rc15. mypy strict clean. doctest gate clean. 7 new tests in `tests/test_passthrough_reliability.py` for findings 3 and 4; 1 new test in `tests/test_doctest_gate.py` for finding 2.

## Item I deliberately did NOT touch

You flagged in passing that `docs/notebooks/speed_benchmark.py` still executes the full PyBLP benchmark at module import (not just at pytest discovery, which the rename to `speed_benchmark.py` already prevents). I agree this is real but I left it as-is for rc15 because the audit explicitly called it "lower priority now that default pytest discovery is fixed." Quick fix if you want it next round: wrap the body in `if __name__ == '__main__':`. I'll do that as a doc-only follow-up if you confirm.

Also still on the deferred list: C2 active half (safe-linalg with auto-fallback), C4 (instrument_channels field renaming). Both genuinely need coauthor input on threshold / fallback policy and on the final column names, per the design notes in the rc5→rc14 handover.

Chris
