# pyRVtest performance at scale: three O(N × n_markets) market scans on the markup / demand-adjustment path (+ optional stage-4 parallelism)

**Package:** pyRVtest `0.4.0rc15`
**Affected files:** `pyRVtest/backends/logit.py`, `pyRVtest/backends/nested_logit.py`, `pyRVtest/solve/demand_adjustment.py`
**Severity:** performance (no incorrect results) — but severe enough that a realistic problem does not finish.

**Summary:** Three routines on the (nested-)logit `Problem.solve` path recompute
per-market row indices with a full-length `np.where` / `np.sum` *inside a loop over
markets*, giving **O(N × n_markets)** behavior. On a problem with ~2.2M product-markets
across ~400k markets each spot is ~10¹² element comparisons and `solve` effectively never
returns. The base `LogitBackend` already solved exactly this elsewhere with an O(N log N)
stable-argsort groupby (`_ensure_market_indices`); these three spots were missed (one of
them a nested override that *regressed* an already-cached base method). All three have a
drop-in, **bit-identical** groupby fix.

This note also proposes an **optional, opt-in threaded execution** of the stage-4
per-market markup-gradient loop, for cases where even the linear-in-`n_markets` pass is
the wall-clock bottleneck.

Everything below was implemented as an external monkeypatch shim against `0.4.0rc15` and
checked for numerical equivalence (see *Equivalence checks*); the suggested patches are
written as in-library edits.

---

## Environment / scale where it bites

A single Nielsen product-module-year, one-level nested logit:

| quantity | value |
|---|---|
| `N` (product-markets / rows) | 2,166,670 |
| `n_markets` (`store × week`) | 401,218 |
| products per market (mean) | ~5.4 |
| nests | 3 |
| models tested | Bertrand, Monopoly |

Call: `problem.solve(demand_adjustment=True, clustering_adjustment=True)`.

Observed: `solve` enters the markups stage (`solve.markups.compute`), calls the demand
backend's `compute_jacobian()` once, and does not return. Stage-by-stage profiling
pinpointed the markups-stage full-Jacobian build as the first stall (Hotspot 1); after
fixing it the run reached — and stalled in — the demand-adjustment stage (Hotspots 2 & 3).

## The common anti-pattern

Each hotspot is the same shape:

```python
markets = np.unique(market_ids)        # n_markets distinct values
for t in markets:
    idx = np.where(market_ids == t)[0]   # or np.sum(market_ids == t): O(N) scan, PER market
    ...
```

`np.where(market_ids == t)` / `np.sum(market_ids == t)` scan all `N` rows once per market,
so the loop is **O(N × n_markets)**. With `N ≈ 2.2e6` and `n_markets ≈ 4.0e5` that is
~8.7×10¹¹ comparisons per spot.

The fix everywhere is the same stable-argsort groupby that `_ensure_market_indices`
(`logit.py` line ~599) already uses: sort once, and because the sort is *stable* each
market's contiguous segment of the sort order is **exactly** `np.where(market_ids == t)[0]`
in its original order. Cost drops to O(N log N) (+ the unavoidable `Σ_t O(J_t²)` per-market
dense block math). Output is bit-identical.

```python
order      = np.argsort(market_ids, kind='stable')
sorted_ids = market_ids[order]
change     = np.concatenate((
    [0], np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1, [len(market_ids)]))
# segments[k] == np.where(market_ids == sorted_ids[change[k]])[0]
segments   = [order[change[k]:change[k + 1]] for k in range(len(change) - 1)]
```

---

## Hotspot 1 — full-Jacobian build: `compute_analytical_jacobian`

`pyRVtest/backends/logit.py`, `compute_analytical_jacobian` (def line ~69). Called once
per `solve` from the markups stage (`LogitBackend.compute_jacobian()` /
`NestedLogitBackend.compute_jacobian()` with `market_id=None`).

```python
markets = np.unique(market_ids)
N       = len(market_ids)

J_max = max(np.sum(market_ids == t) for t in markets)   # line 142 — O(N) PER market

jacobian = np.full((N, J_max), np.nan)
for t in markets:                                       # line 167
    idx = np.where(market_ids == t)[0]                  # line 168 — O(N) PER market
    J_t = len(idx); s_t = shares[idx]
    ...
    jacobian[idx[:, None], np.arange(J_t)[None, :]] = D_t
```

**Two** O(N × n_markets) scans here (the `J_max` reduction and the loop index lookup).

### Suggested patch

```python
market_ids = np.asarray(product_data['market_ids']).flatten()
shares     = np.asarray(product_data['shares']).flatten()
N          = len(market_ids)

# One O(N log N) groupby; stable sort preserves within-market row order so each
# segment == np.where(market_ids == t)[0].
order      = np.argsort(market_ids, kind='stable')
sorted_ids = market_ids[order]
change     = np.concatenate((
    [0], np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1, [N]))
segments   = [order[change[k]:change[k + 1]] for k in range(len(change) - 1)]
J_max      = max((len(seg) for seg in segments), default=0)

jacobian = np.full((N, J_max), np.nan)
for idx in segments:                          # idx replaces np.where(market_ids == t)[0]
    J_t = len(idx)
    s_t = shares[idx]
    if L == 0:
        D_t = _logit_jacobian(alpha, s_t)
    else:
        nesting_t = [arr[idx] for arr in nesting_arrays]
        D_t = _nested_logit_jacobian(alpha, rho, s_t, nesting_t)
    jacobian[idx[:, None], np.arange(J_t)[None, :]] = D_t
return jacobian
```

(All validation and the `_logit_jacobian` / `_nested_logit_jacobian` math are unchanged.)

---

## Hotspot 2 — per-market slice: `NestedLogitBackend.compute_jacobian`

`pyRVtest/backends/nested_logit.py`, `compute_jacobian` (def line ~98):

```python
def compute_jacobian(self, market_id: Any = None) -> Array:
    if self._jacobian_cache is None:
        self._jacobian_cache = compute_analytical_jacobian(...)
    full = self._jacobian_cache
    if market_id is None:
        return full
    mids = np.asarray(self._product_data['market_ids']).flatten()
    idx  = np.where(mids == market_id)[0]    # line 108 — O(N) scan PER call
    block = full[idx]
    block = block[:, ~np.isnan(block).all(axis=0)]
    return block
```

`compute_jacobian(market_id=t)` is called once per market, throughout the
demand-adjustment stage. Each call is an O(N) `np.where`, so this is O(N × n_markets).

The **base** `LogitBackend.compute_jacobian` (`logit.py` line ~625) does **not** have this
problem — it slices through the cached map:

```python
idx = self._ensure_market_indices()[market_id]   # O(1) dict lookup
```

`_ensure_market_indices` (`logit.py` line ~599) builds that map in one O(N log N) groupby.
The nested override re-introduced the per-call `np.where` the base class had eliminated.

### Suggested patch — use the inherited cache, mirroring the base class

```python
def compute_jacobian(self, market_id: Any = None) -> Array:
    if self._jacobian_cache is None:
        self._jacobian_cache = compute_analytical_jacobian(
            self._alpha, self._rho, self._product_data,
            nesting_ids_columns=self._nesting_ids_columns,
        )
    full = self._jacobian_cache
    if market_id is None:
        return full
    idx = self._ensure_market_indices()[market_id]   # cached O(1), was np.where (O(N))
    block = full[idx]
    block = block[:, ~np.isnan(block).all(axis=0)]
    return block
```

---

## Hotspot 3 — stage-4 per-market loop: `compute_demand_adjustment`

`pyRVtest/solve/demand_adjustment.py`, in `compute_demand_adjustment` (the analytical
markup-gradient block, DMSS eq. (77) step 2a):

```python
dD_dtheta_per_market = backend.jacobian_gradient_all_markets()   # line 241  (pass A)
for t in markets:                                                # line 242  (pass B)
    idx = np.where(market_ids == t)[0]                           # line 243 — O(N) PER market
    J_t = idx.shape[0]
    s_t = shares_all[idx]
    D_t = backend.compute_jacobian(market_id=t)                  # line 246
    dD_dtheta = dD_dtheta_per_market[t]
    for m in analytical_models:
        ...
        gradient_markups[m][idx, :] = _analytical_markup_derivative(...)
```

Two issues compound here:

1. **Line 243 is a third O(N × n_markets) scan** — `np.where(market_ids == t)` once per
   market, identical to Hotspots 1 & 2.
2. **The markets are walked twice.** `jacobian_gradient_all_markets()` (line 241) is itself
   a per-market pass (`logit.py` line 720: `{t: self.jacobian_gradient(t) for t in markets}`),
   and pass B re-iterates the same markets. The two passes can be fused: pass B already needs
   `dD_dtheta_per_market[t]`, which is just `backend.jacobian_gradient(t)`.

### Suggested patch — precompute the index map once, and fuse the two passes

```python
# Build market -> row indices ONCE (replaces line-243 np.where).
order      = np.argsort(market_ids, kind='stable')
sorted_ids = market_ids[order]
change     = np.concatenate((
    [0], np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1, [len(market_ids)]))
idx_by_market = {sorted_ids[change[k]]: order[change[k]:change[k + 1]]
                 for k in range(len(change) - 1)}

for t in markets:
    idx = idx_by_market[t]                       # was np.where(market_ids == t)[0]
    J_t = idx.shape[0]
    s_t = shares_all[idx]
    D_t = backend.compute_jacobian(market_id=t)  # Hotspot-2 fix makes this O(1)
    dD_dtheta = backend.jacobian_gradient(t)     # fuse pass A into pass B (no separate dict)
    for m in analytical_models:
        ...
        gradient_markups[m][idx, :] = _analytical_markup_derivative(
            models_downstream[m], O_t, D_t, dD_dtheta, s_t, mu_t,
            mix_flag_all[m], idx, J_t,
        )
```

Fusing also drops the intermediate `dD_dtheta_per_market` dict (one `(J_t, J_t, n_theta)`
array per market — ~400k entries here). Note: fusion replaces the *batched*
`jacobian_gradient_all_markets()` with per-market `jacobian_gradient(t)`. For the analytical
logit/nested backends that method is itself just a per-market loop, so there is no loss; for
the **PyBLP** backend (whose `jacobian_gradient_all_markets` is genuinely batched) keep the
separate call and only apply the line-243 index-map fix.

---

## Optional enhancement — parallelize the stage-4 per-market loop

Once the three scans above are fixed, the stage-4 analytical block is a clean,
**embarrassingly parallel** loop: each market `t` reads shared, read-only state
(`_jacobian_cache`, the index map, ownership) and writes a **disjoint** row block
`gradient_markups[m][idx, :]`. It is a natural candidate for an opt-in parallel mode, e.g.
a `Problem.solve(..., n_jobs=...)` argument or a module option.

A thread-based implementation is attractive because the Jacobian cache can be several GB and
threads share it read-only with **zero serialization**; NumPy's per-market `np.linalg.solve`
/ `einsum` release the GIL, so threads give real (if sub-linear) speedup. Sketch:

```python
# warm caches in the main thread so workers only READ them (avoids a build race)
backend.compute_jacobian()
backend._ensure_market_indices()

def _process_market(t):
    idx = idx_by_market[t]
    D_t = backend.compute_jacobian(market_id=t)
    dD_dtheta = backend.jacobian_gradient(t)
    out = []
    for m in analytical_models:
        out.append((m, idx, _analytical_markup_derivative(... )))
    return out                                  # disjoint rows -> no locking

from concurrent.futures import ThreadPoolExecutor
chunks = np.array_split(markets, n_jobs * 4)    # chunk to amortize dispatch
with ThreadPoolExecutor(max_workers=n_jobs) as ex:
    for out in ex.map(lambda c: [r for t in c for r in _process_market(t)], chunks):
        for (m, idx, d_mu) in out:
            gradient_markups[m][idx, :] = d_mu   # scatter in the main thread
```

Defaulting `n_jobs=1` preserves current behavior exactly. (A process pool is also possible
but would need to ship the multi-GB cache to workers; threads avoid that.)

---

## Equivalence checks

All against stock `0.4.0rc15`.

**Hotspots 1 & 2 (synthetic backend, irregular market sizes 1–6, two nests, random shares):**

- full builder `compute_analytical_jacobian`: `np.array_equal(stock, fast, equal_nan=True)`
  — **True** for both `rho=[]` (plain logit) and `rho=[0.4]` (nested);
- per-market `NestedLogitBackend.compute_jacobian(market_id=t)`: `np.allclose` — **True**
  for every market.

**Hotspot 3 + parallel mode (end-to-end synthetic nested-logit conduct test, Bertrand +
Monopoly, `solve(demand_adjustment=True, clustering_adjustment=True)`):** the resulting
`TRV`, `F`, and `MCS_pvalues` are identical across **stock**, **patched-serial**, and
**patched-threaded (4 workers)** — `np.allclose(..., equal_nan=True)` **True** for all
pairs.

## Impact

All three hotspots sit on the critical path of `Problem.solve` whenever a (nested-)logit
backend is used with many markets:

- Hotspot 1 — once, in the markups stage (`solve.markups.compute`).
- Hotspot 2 — once per market, in the demand-adjustment stage.
- Hotspot 3 — once per market (plus the redundant second pass), in the demand-adjustment
  stage (`solve/demand_adjustment.py`).

For the ~400k-market problem above, eliminating the per-market full scans is the difference
between "does not finish" and a tractable run. Datasets with `store × week` (or any fine)
market definitions — common in retail-scanner demand work — hit this routinely.

## Suggested regression guard

A test that asserts near-constant *per-market* cost would catch any future reintroduction:
build two synthetic problems with the same `N` but `n_markets` differing by ~10×, and assert
the `solve()` (or `compute_jacobian()`) wall-time ratio stays ≈1 rather than scaling with
`n_markets`.
