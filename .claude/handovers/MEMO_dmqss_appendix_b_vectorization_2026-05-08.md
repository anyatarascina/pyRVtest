# MEMO — DMQSS Appendix B vectorization for K_endog ≥ 1

**To:** Marco, Lorenzo, Daniel, Mikkel
**From:** Chris (via pyRVtest dev session, 2026-05-08)
**Status:** DRAFT — awaits Chris's review before sending

## TL;DR

pyRVtest now ships multi-column `endogenous_cost_component` (paper
Appendix A.4 use cases — quadratic cost, scale + scope, and any
linear-in-basis-columns cost regression with K_endog ≥ 1). The code
is a mechanical generalization of the K_endog = 1 path:
two-stage least squares with a length-K_endog gamma block, a
`(K, K_endog)` Λ_q matrix in the influence-function correction, and
the same `K + 1`-instrument identifying-rank requirement enforced in
`_validate_solve_args`.

We'd like to add a short remark to Appendix B that explicitly states
this vectorization, plus the rank-K+1 decomposition that the package's
unified pre-test diagnostic uses. Ideally before tagging v0.4.0 final.

## What changed in pyRVtest

`endogenous_cost_component` widens from `Optional[str]` to
`Optional[Union[str, Sequence[str]]]`. Existing single-string callers
see no behavior change (K_endog == 1 path is bit-identical, verified
on the existing snapshot suite). Two new APIs unlock for users:

```python
# Quadratic cost: c = γ_1 q + γ_2 q² + w'τ + ω
endogenous_cost_component=['q', 'q_sq']

# Scale + scope: log(c) = γ_1 log(q) + γ_2 log(Q⁻) + w'τ + ω
endogenous_cost_component=['log_q', 'log_Q_minus']
```

Plus the `costs_type='log' + demand_adjustment=True` combination is
now properly supported (chain-rule rescaling of `gradient_markups` by
`f'(p − Δ_m) = 1/(p − Δ_m)`), retiring the soft-warning fallback.

The ex-ante diagnostic `instrument_channels(column=...)` automatically
uses DMQSS-Appendix-B z^e residualization when
`endogenous_cost_component` is set: the data-side regression and FWL
partialling project on `(g(q̃), w_exog)` rather than raw `(q, w_exog)`.
The first stage uses the FULL declared instrument set (union over
all `L` test IV bundles) plus `w_exog`. This makes the diagnostic
produce a magnitude that reproduces the residualized moment driving
the actual test, rather than a raw-q-contaminated proxy.

## What we'd like in the paper

A short remark in Appendix B saying:

1. **Lemma B.1 extends to vector γ_m by component-wise replacement.**
   `(γ_m − γ̂_m)` becomes a `(K_endog,)`-vector, `λ_q ∈ ℝ^{d_z}`
   becomes `Λ_q ∈ ℝ^{d_z × K_endog}`, and `q^e ∈ ℝ` becomes
   `q^e ∈ ℝ^{K_endog}`. The proof steps go through with each
   component being `O_p(n^{−1/2})` and the contraction being
   `O_p(n^{−1})`. Standard multi-endogenous-variable IV asymptotics
   under Assumption 5 generalized to require
   `E[(z',w')'(Q,w)]` to have rank `K_endog + K_w`.

2. **The Appendix A.4 rank-K+1 condition decomposes** into

   - First-stage rank K (instruments jointly identify Q_endog), and
   - `Cov(z^e, f(p − Δ_m) − f(p − Δ_0)) ≠ 0` where
     `z^e = z − Λ̂_q g(q̃) − Λ̂_w w` is the residualized instrument
     from Appendix B.

   The equivalence holds under the parametric linear-in-basis-columns
   cost regression: cost-coefficient differences `(γ_{m,k} − γ_{0,k})`
   are constants that come out of the moment-level covariance, leaving
   the residualized term. So one diagnostic — Cov(z^e, f-difference)
   — subsumes both the Dearing pass-through condition and the
   "economic distinctness" rank check from A.4.

   This is what `instrument_channels` reports when
   `endogenous_cost_component` is set: it's the unified pre-test
   diagnostic the framework supports.

The first piece (vector-γ Lemma B.1) is uncontroversial multi-endog
IV. The second piece (rank-K+1 decomposition) is a clean restatement
of A.4 in moment-condition terms; both follow from standard linear-
projection arguments.

## Why we're asking now

The pyRVtest implementation is paper-supported by Appendices A.4 and
B as written, but those sections write Lemma B.1 for K_endog = 1 with
scalar γ_m and `λ_q ∈ ℝ^{d_z}`. The vector-γ generalization is
"obvious" by component-wise replacement, but a strict reader of the
paper would want it stated. Since pyRVtest now ships this and is
about to land in v0.4.0 final, we'd like the paper-side reference to
match.

## Asks

1. **Confirm the vector-γ generalization is correct.** Specifically:
   the Λ_q ∈ ℝ^{d_z × K_endog} replacement, the Assumption 5 rank
   adjustment, and that no uniform-convergence step in the proof
   has K_endog = 1 baked in.
2. **Confirm the rank-K+1 decomposition.** Specifically, that under
   the parametric linear-in-basis-columns cost regression the rank
   condition factors into first-stage rank + Cov(z^e, f-difference).
3. **Add the one-paragraph remark to Appendix B** before the v0.4.0
   final tag. Ideal text below; happy to iterate.

## Suggested wording

> **Remark (vector γ_m generalization).** Lemma B.1 extends to the
> case of K_endog ≥ 1 endogenous variables in marginal cost (paper
> Appendix A.4 cases — economies of scale and scope; quadratic cost;
> general linear-in-basis-columns parametric cost regressions) by
> component-wise replacement: `(γ_m − γ̂_m)` becomes a
> `(K_endog,)`-vector, `λ_q ∈ ℝ^{d_z}` becomes
> `Λ_q ∈ ℝ^{d_z × K_endog}`, and `q^e_jt = q_jt − q̃_jt` becomes
> `q^e_jt ∈ ℝ^{K_endog}`. Assumption 5(ii) generalizes to require
> `E[(z',w')'(Q,w)]` to have rank `K_endog + K_w`. The proof steps go
> through with each component being `O_p(n^{−1/2})` and the matrix
> contractions being `O_p(n^{−1})`. The rank-(K_endog + 1) condition
> in Appendix A.4 then decomposes into first-stage rank K_endog plus
> `Cov(z^e, f(p − Δ_m) − f(p − Δ_0)) ≠ 0`, where
> `z^e = z − Λ̂_q g(q̃) − Λ̂_w w` is the residualized instrument used
> in the variance machinery.

(Two-three sentences if you want it shorter.)

## What pyRVtest does today

For reference, here are the concrete code locations that implement
the vectorization:

- `pyRVtest/solve/endogenous_cost.py::iv_correct` — multi-column 2SLS
  first stage and length-K_endog gamma block in `cost_param`.
- `pyRVtest/solve/demand_adjustment.py::_compute_gamma_gradient` —
  finite-diff over the last K_endog elements of `cost_param`;
  output shape `(M, n_theta)` for K_endog == 1, `(M, n_theta, K_endog)`
  for K_endog > 1.
- `pyRVtest/solve/test_engine.py::compute_instrument_results` —
  K_effective = K_inst − K_endog; Appendix B influence-function
  correction with `Λ_q ∈ ℝ^{d_z × K_endog}` and the variance
  contraction generalized accordingly.
- `pyRVtest/solve/passthrough.py::compute_instrument_channels` —
  data-side regression and FWL partialling project on
  `(g(q̃), w_exog)` when `endogenous_cost_component` is set; first
  stage uses full declared instrument set.
- `pyRVtest/problem.py::_validate_solve_args` — hard-error when
  `K_inst ≤ K_endog` for any instrument set (paper Remark 1 gate).

Tests are in `tests/test_analytical.py` (multi-column round-trip and
smoke), `tests/test_demand_adjustment.py` (log-cost +
demand_adjustment), `tests/test_instrument_channels.py` (z^e
residualization).

## Timing

We'd like to have the paper remark in place before tagging v0.4.0
final. If you confirm by reply, Chris can hold the tag for a week
while you draft. If the math needs more thought, happy to keep
v0.4.0 final tagged with a "vector-γ generalization pending paper
remark" CHANGELOG note and add the cross-reference in a v0.4.1
patch later.
