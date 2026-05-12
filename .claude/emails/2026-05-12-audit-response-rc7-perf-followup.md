# Email draft — rc7 supplement, PT performance (round 2)

**To:** [Auditor 1]
**From:** Christopher Sullivan
**Subject:** pyRVtest rc7 — further PT performance, cumulative numbers

Draft below.

---

Hi [Auditor],

Quick follow-on to the rc6 supplement. After landing the markups cache I profiled what was left in `passthrough_summary` and found the next obvious win: even though markups assembly was now down to one call, `build_passthrough` was still hitting `backend.compute_hessian(market_id=t)` for every (model, market) pair — and demand derivatives only depend on demand state, not on the candidate. Hoisting them out of the model loop drops Hessian calls 9000 → 3000.

`v0.4.0rc7` at `17b921c` carries the patch.

```
pip install git+https://github.com/anyatarascina/pyRVtest@v0.4.0rc7
```

**Cumulative numbers since your audit (rc5 → rc7):**

| Call | rc5 (audit baseline) | rc6 (markups cache) | rc7 (demand-deriv hoist) |
|---|---|---|---|
| `passthrough_summary` | 12.5s | 6.0s | 5.0s |
| `instrument_channels` | 11.6s | 5.1s | 4.3s |
| Markups assembly calls | 5 | 1 | 1 |
| Hessian calls | 9000 | 9000 | 3000 |

(Shipped synthetic, T=3000, 4 candidates. My Python 3.9 / numpy 1.26 / pyblp 1.1.2 stack; your Python 3.13 / numpy 2.4 stress stack should see proportionally similar gains.)

Both improvements are pinned with regression tests in `TestMarkupsAssemblyCallCount`: one asserts the markups assembly count is exactly 1 across both diagnostics, the other asserts `compute_hessian` is called exactly `n_markets` times in `passthrough_summary` regardless of `n_models`.

Values bit-identical at every step. Public API unchanged; both optimization hooks are underscore-prefixed (`_precomputed_markups`, `_precomputed_demand_derivatives`).

**What I haven't fixed in this round.** The remaining ~5s on `passthrough_summary` is structurally harder:

- `compute_passthrough_numerical` (~1s) is a Python loop of 9000 (model, market) cells doing 2x2 `linalg.solve`/`inv` per cell. Python overhead dominates the actual linalg. Batching is possible but invasive — markets have variable J_t and the FOC dispatch is per-conduct.
- `_compute_markups` (~1.3s) runs once per diagnostic call. Inside, ~0.7s is unaccounted-for tottime (numpy/recarray overhead). Worth a v0.5 pass but no obvious single fix.
- Feature-metric loops (~1s) — Frobenius / column-norm work per pair. Could vectorize across markets.

The next item I'm planning to ship is what we discussed in the rc6 supplement: marking the slow PT tests `@pytest.mark.slow` so the dev loop isn't dominated by them.

Chris
