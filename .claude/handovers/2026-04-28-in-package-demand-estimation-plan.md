# Handover — In-package demand estimation (logit + nested logit): design + plan

**Date:** 2026-04-28 (sibling to the F-reliability handover from the same session)
**Branch:** `feat/in-package-demand` (not yet created — off `v0.4-refactor`)
**Status:** **design approved, implementation not yet started.** This handover documents the plan and decisions; pickup begins by creating the branch and starting Phase 1.

## Goal

Add in-package 2SLS estimation of plain-logit and nested-logit demand parameters to `pyRVtest`, so users with simpler demand systems don't need to run PyBLP first. Both APIs:

- **A (standalone estimator):** `pyRVtest.LogitEstimator(...).solve()` returns a populated `demand_params` dict that the user passes to `Problem(demand_params=...)`.
- **B (inline shortcut):** `Problem(demand_params={'estimate': 'logit', ...})` — Problem detects the `'estimate'` key and runs the estimator internally before the existing `_construct_demand_backend` path.

Both ship in the same branch; B is a thin sugar layer over A.

## Background and motivation

This was issue #1 from the start of the 2026-04-28 session — Chris asked about (1) adding internal demand estimation, alongside (2) the F-stat fragility work. Issue #2 became `feat/f-reliability` (committed and pushed in the same session). This handover documents the plan for issue #1.

The 2026-04-20 handover (`2026-04-20-ci-matrix-green.md`) flagged this as v0.4-final or v0.5 work:
> 4. `Problem(demand_backend=...)` public kwarg — still v0.4.0-final or v0.5.

The design here doesn't add a `demand_backend=` kwarg — instead it extends the existing `demand_params=` dict path. Cleaner.

## Decisions made (during the design discussion)

1. **Build on the existing `demand_params=` path, not a new code path.** [pyRVtest/problem.py:1761-1806](pyRVtest/problem.py:1761) (`_construct_demand_backend`) already routes a populated `demand_params` dict to `LogitBackend` / `NestedLogitBackend`. The estimator just produces the same dict.
2. **Reuse existing analytic gradient machinery.** [LogitBackend](pyRVtest/backends/logit.py:530) and `NestedLogitBackend` already implement `compute_jacobian`, `compute_hessian`, `demand_moments`, `xi_gradient`, `jacobian_gradient`. Constructing them with estimated `alpha` (and `rho`) means demand-adjustment / DMSS eq. 77 first-stage correction works without touching backend code.
3. **Users pass instruments explicitly via `demand_instrument_columns`.** Auto-construction of the standard within-share IV (e.g. count-of-products-in-nest) is opt-in via a kwarg, not default.
4. **Both A and B in the same branch.** A is the canonical path; B is a ~20-line sugar layer on top.
5. **Two-stage least squares, not full BLP/MLE.** Berry inversion + linear 2SLS only. Plain logit is one endogenous regressor (price); nested logit is two (price + within-nest log-share). For nonlinear demand systems, users still go through PyBLP.
6. **No `xi` in the returned dict.** The backend recomputes ξ internally from `(alpha, beta, X, Z, prices, shares)`. Returning ξ from the estimator would be redundant. The estimator may expose ξ on its results object for inspection but the dict stays minimal.

## API design (concrete)

### A — standalone estimator

```python
import pyRVtest

# Plain logit
estimator = pyRVtest.LogitEstimator(
    product_data=df,
    formulation_X=pyRVtest.Formulation('1 + size + light'),
    formulation_Z=pyRVtest.Formulation('0 + cost1 + cost2'),
)
demand_params = estimator.solve()
# returns {
#     'alpha': -2.31,            # estimated price coefficient
#     'beta': array([...]),       # estimated coefficients on X (excl. price)
#     'x_columns': ['size', 'light'],
#     'demand_instrument_columns': ['cost1', 'cost2'],
#     'W_demand': 2x2 ndarray,    # 2SLS weight matrix (Z'Z/N)^-1 by default
# }

problem = pyRVtest.Problem(
    cost_formulation=...,
    instrument_formulation=...,
    product_data=df,
    demand_params=demand_params,
    models=[...],
)
results = problem.solve()
```

### Nested logit variant

```python
estimator = pyRVtest.NestedLogitEstimator(
    product_data=df,
    formulation_X=pyRVtest.Formulation('1 + size + light'),
    formulation_Z=pyRVtest.Formulation('0 + cost1 + cost2 + within_iv'),
    nesting_ids_column='nest',
)
demand_params = estimator.solve()
# returns dict with 'alpha', 'beta', 'rho' (single scalar; one-level nesting),
# 'x_columns', 'demand_instrument_columns', 'W_demand', 'nesting_ids_columns'

# Or with auto-constructed within-share IV:
estimator = pyRVtest.NestedLogitEstimator(
    product_data=df,
    formulation_X=pyRVtest.Formulation('1 + size + light'),
    formulation_Z=pyRVtest.Formulation('0 + cost1 + cost2'),
    nesting_ids_column='nest',
    auto_construct_within_share_iv=True,  # opt-in
)
demand_params = estimator.solve()
# Auto-appended IV (e.g., count of products in own nest) shows up in
# demand_params['demand_instrument_columns'].
```

### B — inline shortcut

```python
problem = pyRVtest.Problem(
    cost_formulation=...,
    instrument_formulation=...,
    product_data=df,
    demand_params={
        'estimate': 'logit',  # or 'nested_logit'
        'formulation_X': pyRVtest.Formulation('1 + size + light'),
        'formulation_Z': pyRVtest.Formulation('0 + cost1 + cost2'),
        # for nested_logit, also: 'nesting_ids_column': 'nest',
        # opt-in helper: 'auto_construct_within_share_iv': True
    },
    models=[...],
)
```

Internally, `Problem.__init__` detects the `'estimate'` key and replaces `demand_params` with the result of the appropriate estimator's `.solve()` call before running `_construct_demand_backend`.

## Math (standard 2SLS — for reference)

### Plain logit

Berry inversion on the inside good:
$$\delta_{jt} \equiv \log s_{jt} - \log s_{0t} = X_{jt}'\beta + \alpha p_{jt} + \xi_{jt}$$

Run 2SLS with $p$ endogenous, $X$ exogenous, $Z$ as the instrument matrix. Returns $(\hat\alpha, \hat\beta, \hat\xi)$ as in any textbook 2SLS.

### Nested logit (one-level)

$$\delta_{jt} = X_{jt}'\beta + \alpha p_{jt} + \rho \log s_{j|g_t} + \xi_{jt}$$

Two endogenous regressors: $p$ (price) and $\log s_{j|g}$ (within-nest log-share — endogenous because $\xi$ enters the within-nest share through demand). Need at least two excluded instruments. Common choices:
- Cost shifter for $p$
- Number-of-products-in-nest (or its standard variants) for $\log s_{j|g}$

The auto-construction option (`auto_construct_within_share_iv=True`) appends the count-of-products-in-own-nest column to `Z` automatically.

Standard 2SLS otherwise. Returns $(\hat\alpha, \hat\beta, \hat\rho, \hat\xi)$.

For higher-level nested logit, defer until v0.5+ — the literature mostly uses one-level nesting.

## Implementation plan

Three commits, each independently testable:

### Phase 1 — `LogitEstimator` (A path, plain logit)

Files:
- New: `pyRVtest/estimators/__init__.py`, `pyRVtest/estimators/logit.py`
- New: `tests/test_logit_estimator.py`

Estimator class:
- Constructor: `(product_data, formulation_X, formulation_Z, market_ids_column='market_ids', shares_column='shares', prices_column='prices', W_demand=None)`.
- Validate: shares sum to ≤ 1 per market (outside good has positive share); X and Z columns exist; Z has rank ≥ K (X's column count + 1 for price).
- Construct $\delta = \log s - \log s_0$.
- Construct design matrix $[X, p]$ and instrument matrix $[X, Z]$ (X is included in Z because it's exogenous).
- Run 2SLS with default weight $(Z'Z/N)^{-1}$ or user-supplied `W_demand`.
- Store estimates on the instance for inspection.
- `.solve()` returns the populated `demand_params` dict.

Tests:
- Recovery test on a synthetic logit DGP (estimator should hit $\alpha_{\text{true}}$ within MC noise at moderate N).
- Output dict has the expected keys and types.
- `demand_params` from this estimator successfully drives `Problem.solve()` end-to-end.
- Verify the resulting markups are consistent with the underlying DGP.
- Error cases: missing columns, rank-deficient Z, alpha estimated > 0 (warning, not error — let user inspect).

### Phase 2 — `NestedLogitEstimator` (A path, nested logit)

Files:
- New: `pyRVtest/estimators/nested_logit.py`
- New: `pyRVtest/estimators/_within_share.py` (helper for auto-constructed within-share IV)
- New: `tests/test_nested_logit_estimator.py`

Estimator class — extends Phase 1 with:
- Compute within-nest shares $s_{j|g}$ from `product_data` and the `nesting_ids_column`.
- Two endogenous regressors in the 2SLS: $p$ and $\log s_{j|g}$.
- Validate $\hat\rho \in [0, 1)$ (issue warning if outside; don't auto-clip).
- Optional `auto_construct_within_share_iv` flag → call into `_within_share.construct_count_in_nest_iv(product_data, nesting_ids_column)` and append.

Tests parallel to Phase 1, plus:
- Auto-IV path produces a different IV column from the explicit-IV path.
- Recovery test on a nested-logit DGP recovers $(\alpha, \rho)$ within MC noise.

### Phase 3 — Inline shortcut (B path)

Files:
- Modified: `pyRVtest/problem.py` — add detection branch in `Problem.__init__` (or `_construct_demand_backend`), route `demand_params['estimate']` to the appropriate estimator.
- Modified: `tests/test_logit_estimator.py` and `tests/test_nested_logit_estimator.py` — add a class for the inline path that constructs `Problem(demand_params={'estimate': ...})` and verifies the same end-to-end behavior as the standalone path.

The detection logic, ~20 lines:

```python
# Inside Problem.__init__, before _construct_demand_backend
if demand_params is not None and 'estimate' in demand_params:
    kind = demand_params['estimate']
    common = {
        'product_data': self._product_data_raw,
        'formulation_X': demand_params['formulation_X'],
        'formulation_Z': demand_params['formulation_Z'],
    }
    if kind == 'logit':
        from .estimators.logit import LogitEstimator
        demand_params = LogitEstimator(**common).solve()
    elif kind == 'nested_logit':
        from .estimators.nested_logit import NestedLogitEstimator
        demand_params = NestedLogitEstimator(
            **common,
            nesting_ids_column=demand_params['nesting_ids_column'],
            auto_construct_within_share_iv=demand_params.get('auto_construct_within_share_iv', False),
        ).solve()
    else:
        raise ValueError(
            f"Unknown demand_params['estimate']={kind!r}. Expected one of "
            f"'logit' or 'nested_logit'. Fix: pass one of these names, or "
            f"omit 'estimate' and supply alpha/rho directly in demand_params."
        )
    self.demand_params = demand_params  # so downstream code uses estimated values
```

## Open items / decisions to revisit

1. **Default weight matrix for 2SLS.** Default to $(Z'Z/N)^{-1}$ (standard 2SLS) or to a more efficient choice from the moment conditions? Default to standard 2SLS unless coauthors prefer otherwise.
2. **Standard errors.** The estimator computes $\hat\beta$ with standard 2SLS asymptotics. Should we also compute a heteroskedasticity-robust SE by default? Cluster-robust on `market_ids`? For the first version, ship with classical 2SLS SE (matches what most users expect from a textbook `ivregress`); add robust/cluster as a follow-up via an `se_kind` kwarg.
3. **Sanity on $\hat\rho$.** For nested logit, should we error if $\hat\rho \notin [0, 1)$? Or warn? Warning is more informative — let the user decide whether to investigate or override. Don't auto-clip.
4. **Multi-level nested logit.** Defer to v0.5+. The literature mostly uses one-level nesting; full multi-level adds complexity (the `rho` becomes a vector across levels, the within-share is a product of level-specific shares).
5. **Documentation.** A short section in `docs/tutorial.rst` showing the A path on a yogurt-like DGP. The B path is mentioned as a one-liner shortcut.
6. **PyBLP overlap check.** PyBLP can also estimate plain logit (degenerate-RC case). We need a sentence in the docs noting "if you're already using PyBLP, just continue with `demand_results=pyblp_results`; this estimator is for the lighter case where PyBLP is overkill."

## Branch + PR plan

Branch: `feat/in-package-demand` off `v0.4-refactor`. Independent of `feat/f-reliability` (which is also off `v0.4-refactor`).

PR sequence:
- `feat/f-reliability` → `v0.4-refactor` (already open)
- `feat/in-package-demand` → `v0.4-refactor` (this work; can proceed in parallel)

Both can ship in v0.4.1 if v0.4.0 finalizes first; otherwise as v0.5 features.

## Pickup for next session

1. Read this handover and `MEMO_F_reliability_diagnostic_2026-04-28.md` for context.
2. Read [pyRVtest/problem.py:1761-1806](pyRVtest/problem.py:1761) (`_construct_demand_backend`) and [pyRVtest/backends/logit.py:530+](pyRVtest/backends/logit.py:530) (`LogitBackend`) to verify the `demand_params` keys and `LogitBackend.__init__` signature.
3. `git checkout v0.4-refactor && git pull && git checkout -b feat/in-package-demand`.
4. Phase 1: implement `LogitEstimator`, write tests, commit.
5. Phase 2: implement `NestedLogitEstimator` + within-share IV helper, write tests, commit.
6. Phase 3: implement the inline shortcut in `Problem.__init__`, add tests for the inline path, commit.
7. Add tutorial section to `docs/tutorial.rst`.
8. Push branch, point coauthors at the diff for review.

Estimated total: 4-5 days of focused work.

## Files to create / modify (summary)

| Path | Phase | Purpose |
|---|---|---|
| `pyRVtest/estimators/__init__.py` | 1 | Package marker, re-exports |
| `pyRVtest/estimators/logit.py` | 1 | `LogitEstimator` class |
| `pyRVtest/estimators/nested_logit.py` | 2 | `NestedLogitEstimator` class |
| `pyRVtest/estimators/_within_share.py` | 2 | Auto-IV helper |
| `pyRVtest/problem.py` | 3 | `'estimate'` detection branch in `Problem.__init__` |
| `pyRVtest/__init__.py` | 1, 2 | Re-export `LogitEstimator`, `NestedLogitEstimator` |
| `tests/test_logit_estimator.py` | 1, 3 | Unit + inline-path tests |
| `tests/test_nested_logit_estimator.py` | 2, 3 | Unit + inline-path tests |
| `docs/tutorial.rst` | (post) | Worked example |

## Cross-references

- Sibling handover (same session, different feature): [.claude/handovers/2026-04-28-f-reliability-calibration.md](.claude/handovers/2026-04-28-f-reliability-calibration.md)
- F-reliability memo (sibling design doc): [MEMO_F_reliability_diagnostic_2026-04-28.md](../../MEMO_F_reliability_diagnostic_2026-04-28.md)
- Prior session handover (v0.4 release context): [.claude/handovers/2026-04-20-ci-matrix-green.md](2026-04-20-ci-matrix-green.md)
- v0.4 design plan (where the backend abstraction was introduced): [.claude/plans/v0.4-refactor.md](../plans/v0.4-refactor.md)
