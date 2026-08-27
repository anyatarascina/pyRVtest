# Handover: alignment with the revised Duarte et al. (2026), Appendix B (2026-08-27)

**Deliberate >1e-7 snapshot change** (`analytical_scale`, field `F` only).
This note is the justification + provenance record required by the
AGENTS.md snapshot decision rule.

## Source

`notes/manuscript_EditsByMikkel_Highlighted.tex` (private, gitignored),
Appendix B "Distributions of the RV- and F-statistics", change-tracked
edits by a coauthor. Summary of the revision, for a cost specification
with `d_q` endogenous regressors instrumented by `d_z` excluded
instruments:

* Full-rank representation: `z^{e,0}` (instruments residualized on
  `(q_tilde, w)`) has rank `r = d_z - d_q`; `S_e` selects `r` of them,
  `z^e = S_e z^{e,0}`, `W = (E z^e z^e')^{-1}` (true inverse). Every
  `d_z - 1` becomes `r`.
* New influence function (tex lines 1288-1305):

  ```
  psi_m,i = W^{1/2}( z^e_i omega_mi - g_m - S_e Lambda_q q^e_i z^r_i' Z g^0_m )
          + (g_m' (x) I_r) A_W vec( z^e_i z^e_i' - W^{-1}
                                    - S_e Sigma^0_e Z z^r_i q^e_i' Lambda_q' S_e'
                                    - S_e Lambda_q q^e_i z^r_i' Z Sigma^0_e S_e' )
  A_W = -(I_r (x) W^{-1/2} + W^{-1/2} (x) I_r)^{-1} (W^{1/2} (x) W^{1/2})
  ```

  The `W^{3/4}` term is gone (exact Sylvester derivative instead) and the
  q_tilde-estimation effect enters through both `g_hat` and `W_hat`.
* F: `sigma_m^2 = trace(V^AR_mm W^{-1}) / r`, `pi_m = W g_m` in the
  `r`-dimensional representation.

## Result: the package already matched, up to one numerical detail

Contracting the revised `psi_m,i` with `2 (W^{1/2} g_m)'` gives exactly
the scalar score the package computes since commit `f3879b6`,
`phi_mi = 2 v_mi omega_mi - v_mi^2 - Q_m`, observation by observation:

1. `2 g' W^{1/2} X g = g' (dW) g = -(Wg)' (dW^{-1}) (Wg)` for the
   Sylvester solution `X` (`dW = X W^{1/2} + W^{1/2} X`), i.e. the exact
   derivative of `Q = g'Wg` in `W`;
2. `g^0_m = Sigma^0_e S_e' W g_m` (because `z^{e,0}` lies a.s. in the
   column space of `Sigma^0_e S_e'`), so the q_tilde term in the g-hat
   channel, `-2 pi' S_e Lambda_q q^e z^r' Z g^0`, and the two q_tilde
   terms in the W-hat channel, `+2 pi' S_e Lambda_q q^e z^r' Z Sigma^0_e
   S_e' W g`, are equal and opposite;
3. what remains is `2 v omega - 2Q - (v^2 - Q) = phi`.

The vector covariance `V^RV` itself does change with the q_tilde terms
(they matter for the vector-level Lemma) but nothing the package reports
depends on it. The F path already used `K_effective = K - K_endog = r`
and `trace(V^AR Sigma_e)` in the pseudo-inverse basis, which equals the
`r`-form for any admissible `S_e`.

Decision (user): keep the scalar score in production; the vector formula
lives only in `notes/mc_rv_variance_appendix_b.py` and, hand-coded, in
`tests/test_appendix_b_reference.py` (no package module).

## The one numerical fix: F first-stage residual (endogenous-cost path)

`test_engine.py` obtained `e_m = omega_m - P_Z omega_m` with `Q_Z` from
`np.linalg.qr(Z_orthogonal)`. On the endogenous-cost path
`Z_orthogonal` is rank-deficient (rank `K - K_endog`; last `R` diagonal
~1e-13) and the plain QR is not rank-revealing: `Q_Z` carries a spurious,
rounding-determined direction that the projection also removes,
perturbing `V^AR` and `F` by O(1/N) with a random sign/size. The revised
appendix defines `e_m = omega_m - z^e' pi_m`, `pi_m = W g_m`; with the
pseudo-inverse `W`, `Z W g_m = Z (Z'Z)^+ Z' omega_m` is the exact
projection. That is now the code (`test_engine.py`, F block). On the
full-rank path the two coincide to rounding.

Verified: on the Monte Carlo cross-check the package `F` differed from the
reference by 0.2% before the fix (2504.37 vs 2499.05) and agrees to 13
digits after it; TRV, RV denominator and Q agreed to 12 digits throughout.

### Snapshot deltas (old -> new baseline, max |delta| per field)

| snapshot          | markups | marginal_cost | taus    | g       | Q       | TRV     | F        | MCS_pvalues |
|-------------------|---------|---------------|---------|---------|---------|---------|----------|-------------|
| analytical_scale  | 0       | 1.5e-14       | 1.5e-14 | 1.1e-16 | 3.9e-19 | 3.7e-14 | 1.07e-02 | 2.8e-14     |
| all other snapshots | unchanged (suite passes without regeneration) |

`analytical_scale` has N = 60, so a one-direction O(1/N) perturbation of
`V^AR` is ~1-2%: `F` 0.98605 -> 0.99674. `tests/test_analytical.py::
_hand_compute_scale` was updated to the `pi_m = W g_m` form; the
hand-computed F now matches the package to 1e-7 again.

## Monte Carlo evidence

`notes/mc_rv_variance_appendix_b.py` (gitignored) -> `notes/mc_rv_variance_appendix_b.md`,
2000 replications, n = 2000, seed 0. Nondegenerate fixed-g null
`Q_1 = Q_2 = 0.35^2`, endogenous `log q` (corr(u, omega_0) = 0.6),
skewed correlated instruments with eigenvalues 25/4/1/0.25.

| design                 | estimator      | mean sigma-hat / sigma_MC | sd(T_RV) | reject 5% |
|------------------------|----------------|---------------------------|----------|-----------|
| A: d_z=4, d_q=1, r=3   | OLD (pre-revision, as shipped before f3879b6) | 1.186 | 0.844 | 1.8% |
|                        | OLD_NO_QTILDE (DMSS W^{3/4})                  | 0.922 | 1.082 | 6.8% |
|                        | NEW (revised Appendix B vector psi)           | 0.973 | 1.027 | 5.1% |
|                        | SCALAR (package)                              | 0.973 | 1.027 | 5.1% |
| B: d_z=5, d_q=2, r=3   | NEW = SCALAR                                  | 0.979 | 1.018 | 5.2% |
| C: d_z=2, d_q=1, r=1   | all five coincide                             | 1.037 | 0.962 | 4.1% |

`NEW`, `NEW_NO_QTILDE` and `SCALAR` agree to ~1e-16 in every replication.
Design D (`pi_m = 0`): the simulated F has sd 0.450; the Proposition's
limit with `I_r`, r = 3, has sd 0.443, the old `I_{d_z-1}` (4) has sd
0.384 — `r` enters through the critical-value row (the value of F is
algebraically invariant to the divisor, which cancels).

## Code / test surface

- `pyRVtest/solve/test_engine.py` — F first-stage residual via
  `pi_m = W g_m`; effective-rank `UserWarning`; docstring/comments cite
  the revised Duarte et al. (2026), Appendix B. RV/MCS numerics untouched.
- `tests/test_appendix_b_reference.py` (new) — hand-coded revised
  `psi`, `A_W` finite-difference check, scalar contraction == package
  (`d_q = 1`, `d_q = 2`, absorbed FE), q_tilde cancellation, `S_e`
  invariance == pseudo-inverse form, rank-audit warning.
- `tests/test_analytical.py::_hand_compute_scale` — `pi_m = W g_m` form.
- `tests/snapshots/analytical_scale.json` — regenerated (Linux,
  ScalableRV env: numpy 2.5.0, pyblp 1.2.0).
- `docs/math.rst`, `CHANGELOG.md`.

## Follow-ups for coauthors

- `tests/replication/test_dmss_yogurt.py` EXPECTED_* values (still
  None) must be computed with the current code.
- The manuscript's main text (Section "Step 4") still says a separate
  q_tilde adjustment is needed in `sigma_RV`; per the revised appendix
  the adjustment is present in the vector `psi` but cancels in the
  scalar variance — consider rewording.
