# Email draft — rc10, batched _compute_markups + scaffolding

**To:** [Auditor 1]
**From:** Christopher Sullivan
**Subject:** pyRVtest rc10 — batched _compute_markups, full-suite-gated

Draft below.

---

Hi [Auditor],

Last perf round. rc9 was sub-second already, but the coauthor concern about speed pushed me to batch the remaining hotspot: the per-(model, market) `_compute_markups` inner loop. This is more invasive than the rc6-rc9 PT-side changes because `_compute_markups` is load-bearing for every `Problem.solve` call, not just PT diagnostics. So before doing it I added scaffolding and ran a full-suite gate.

Tag: `v0.4.0rc10` at `d6ed670`. Install:

```
pip install git+https://github.com/anyatarascina/pyRVtest@v0.4.0rc10
```

## What rc10 did

**Batched downstream markups.** For the five conducts whose markup formula is a pure per-market closed form — `bertrand` / `cournot` / `monopoly` / `perfect_competition` / `constant_markup` — `_compute_markups` now groups markets by `J_t` and dispatches one batched LAPACK call per J-group. Falls back to the per-market scalar loop for vertical / mix / custom / user-supplied (so the existing Villas-Boas + custom-callable paths are untouched).

Numerically identical to the scalar path (same LAPACK routines, same input slices). Verified to `atol=1e-12` by parity tests across 9 parametrized random fixtures.

**PT bench (shipped synthetic, T=3000, n_models=4):**

| Call | rc9 | rc10 |
|---|---|---|
| `passthrough_summary` | 0.75s | ~0.66s |
| `instrument_channels` | 0.9s | ~0.83s |

Smaller delta than rc6-rc9 because most of `_compute_markups`'s time is now elsewhere (Hessian computation, batched setup overhead) — but real. The bigger payoff is in `Problem.solve(demand_adjustment=True)`, where `_compute_markups` is called `(n_theta + 1)` times per solve. With a 5-parameter random-coefficients spec that's 6×; the savings compound.

## Cumulative since your audit

| Tag | `passthrough_summary` | `instrument_channels` | Δ vs rc5 |
|---|---|---|---|
| rc5 (audit baseline) | 12.5s | 11.6s | — |
| rc6 (markups cache) | 6.0s | 5.1s | ~52% |
| rc7 (deriv hoist) | 5.0s | 4.3s | ~60% |
| rc8 (PT batching + metric batching) | 2.4s | 2.5s | ~81% |
| rc9 (index map + backend cache) | 0.75s | 0.9s | ~94% / ~92% |
| **rc10** | **~0.66s** | **~0.83s** | **~95% / ~93%** |

## Why I added scaffolding before this round

Unlike rc6-rc9 (PT-only blast radius), `_compute_markups` is called from `Problem.solve` itself, including the `demand_adjustment=True` path that perturbs demand parameters. A bug there breaks the entire test suite, not just PT. I didn't trust the existing tests alone to catch a subtle batching bug, so I added:

- **`tests/test_compute_markups_direct.py`** — 27 new tests. Hand-formula parity for each batchable conduct; mixed candidate sets; user-supplied short-circuit; variable J_t (mixed buckets); random-fixture scalar-vs-batched parity (9 cases); batchability decision matrix (11 cases pinning the gate logic).
- **Pre-batching baseline run of the full suite** at rc9 to make sure the baseline was actually 802/802 (it caught a pre-existing mypy strict failure from my rc5-rc8 type-annotation oversights, which I fixed).
- **Post-batching gate** on the full suite — **822/822 in 213s**, including the 20 new tests.

## What rc10 doesn't change

The batched path is *narrow*: only the conducts in `_BATCHABLE_DOWNSTREAM` and only when there's no upstream / mix flag / custom spec. Vertical models, `MixCournotBertrand`, `CustomConductModel`, and anything with `user_supplied_markups` go through the existing scalar per-market loop unchanged. The decision tree is pinned by `TestBatchabilityMatrix`.

## Public API

Unchanged. The new helper (`_compute_batchable_downstream_markups`) is underscore-prefixed; the batchability gate (`_is_batchable_downstream_model`) is similarly private. Existing public callers (`build_markups`, `evaluate_first_order_conditions`) work exactly as before.

## What's left after rc10

PT diagnostics at ~0.66s and ~0.83s is fast enough that I think we're done. Further perf wins would need Cython/numba or a full restructuring of the per-market data layout. Neither is right for v0.4.

If you find any value drift or test failure under rc10, send back a reproducer.

Chris
