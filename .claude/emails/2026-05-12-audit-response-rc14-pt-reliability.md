# Email draft — rc14, PT reliability vocabulary

**To:** [Auditor 2]
**From:** Christopher Sullivan
**Subject:** pyRVtest rc14 — PT reliability diagnostic (Audit 2 Findings C2 + C3)

Draft below.

---

Hi [Auditor],

Took the passive half of Findings C2 and C3 in rc14. You asked for vocabulary letting users distinguish *structural* pass-through degeneracy from *numerical* instability — that's what this method provides.

Tag: `v0.4.0rc14` at `e65fc8b`. Install:

```
pip install git+https://github.com/anyatarascina/pyRVtest@v0.4.0rc14
```

## What rc14 added

New diagnostic method on both `Problem` and `ProblemResults`:

```python
problem.passthrough_reliability() -> pd.DataFrame
```

Returns one row per (candidate model, market) with these columns (matching your C3 spec):

| Column | Type | Meaning |
|---|---|---|
| `model_index`, `model`, `market_id` | int, str, hashable | row identifier |
| `pt_method` | str | `'analytical_trivial'` / `'analytical_vertical'` / `'numerical_central_difference'` |
| `pt_condition_number` | float | `np.linalg.cond(P_m)` |
| `pt_rank` | int | `np.linalg.matrix_rank(P_m)` |
| `pt_status` | str | `'robust'` / `'ill-conditioned'` / `'near-degenerate'` / `'undefined'` |
| `pt_warning` | str | empty when robust; otherwise short description |

Thresholds are configurable kwargs (`cond_warn=1e6`, `cond_severe=1e12`, `cond_undefined=1e16`). Defaults correspond to "lose ~6 / ~12 / ~16 digits of precision relative to double."

By the identity `cond(A^{-1}) = cond(A)`, the reported `pt_condition_number` is exactly `cond(I - ∂Δ_m/∂p)` for the numerical-central-difference path. For the Vertical Villas-Boas path it bounds the conditioning of the `G` matrix being inverted.

## What this method is for

Your C3 ambiguity:
> A reported PT distance of zero should never be ambiguous between:
> 1. true structural degeneracy;
> 2. near-zero diagnostic denominator;
> 3. ill-conditioned numerical derivative calculation;
> 4. model/backend mis-specification.

With this method users can write:

```python
distances = problem.passthrough_summary().to_dataframe()
reliability = problem.passthrough_reliability()

# Are any cells unreliable?
bad = reliability[reliability['pt_status'] != 'robust']
```

If `bad` is empty, a zero distance in `passthrough_summary` means **structural** degeneracy (your case 1). If `bad` is populated, the distance may be contaminated by numerical issues (your cases 2-3).

## What I did NOT do in this round

**The active half of C2** — auto-fallback to `pinv` when ill-conditioned, condition reporting threaded through the existing PT methods themselves. That would *change* computed values, which would break the bit-identical-vs-rc5 parity I've maintained across rc6-rc14. Doing it well requires:
- A policy decision on when to fall back (right now nothing does; the user sees raw `inv` output).
- Reflecting the fallback in the rest of the diagnostic chain (`passthrough_summary` metrics, `instrument_channels` projections).
- Documenting the new behavior so users know whether the reported `P_m` is exact or pseudoinverse-regularized.

I'd want coauthor input before changing the math layer. That's v0.5.

**Finding C4** (rename `instrument_channels` fields from `structural`/etc. to `pt_feature_gap`/`projected_pt_gap`/`data_slope`/`direct_beta_gap`) — also v0.5, also needs coauthor agreement on names. Public API surface, not something I want to ship unilaterally.

**Finding D** (derivative validation for custom backends) — v0.5; `UserSuppliedBackend` is labeled experimental in v0.4 anyway.

## Value preservation

Cross-rc bit-identical parity check, rc5 vs rc14:

| Quantity | max |Δ| |
|---|---|
| TRV (constant-cost) | 0 |
| F (constant-cost) | 0 |
| markups | 0 |
| MCS p-values (constant-cost) | 0 |
| TRV (demand-adj) | 0 |
| F (demand-adj) | 0 |
| MCS p-values (demand-adj) | 0 |

The 14 release candidates between rc5 and rc14 made zero changes to any computed output on the shipped synthetic example. PT diagnostics ~20× faster, full test suite ~6× faster, and the new diagnostic is purely additive.

## Tests + verification

- 24 new tests in `tests/test_passthrough_reliability.py`: column contract, method classification, condition-number-matches-direct-call, threshold logic (all four states + rank-deficient + non-finite), threshold kwarg behavior, market_id filter, value preservation, ProblemResults mirror parity, per-conduct classifier.
- Full test suite: 846/846 in 242s. mypy strict clean.
- `docs/api.rst` updated to list `Problem.passthrough_reliability` and `ProblemResults.passthrough_reliability` under a new "Numerical reliability" subsection of the PT block added in rc13.

If anything regresses or the threshold defaults disagree with what you'd want, send back a follow-up.

Chris
