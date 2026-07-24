# Handover: RV variance corrected to the scalar-score formula (2026-07-24)

**Deliberate >1e-7 snapshot change.** This note is the justification +
provenance record required by the AGENTS.md snapshot decision rule.

## What changed and why

Two coauthor notes (checked in under `notes/`) established that the RV
denominator implemented in `pyRVtest/solve/test_engine.py` was incorrect
for K > 1 instruments:

1. `notes/variance_proof_note.pdf` — the published Appendix J proof uses
   `(ABA)^{1/2} = A^{1/2} B^{1/2} A^{1/2}`, which requires commuting
   matrices. The `W^{3/4}` influence-function term derived from it is
   valid only in the scalar / same-eigenbasis case.
2. `notes/Memo_appendixB.pdf` — additionally, the explicit q_tilde
   first-stage correction (endogenous-cost path) does not belong in the
   scalar RV variance: by the exact FWL projection identity
   `P_{z_e} = P_{(z,w)} - P_{(q_hat,w)}`, its first-order effect through
   W-hat is exactly cancelled by the omitted effect through g-hat.

Replacement (exact delta method on `Q_m = g_m' W g_m`, memo eq. (1)-(2)):

```
phi_mi = 2 v_mi omega_mi - v_mi^2 - Q_m,    v_mi = z_i' W g_m
sigma^2(m,m') = C_mm + C_m'm' - 2 C_mm'
C_lk = (1/N) sum_i phi_li phi_ki            (cluster-summed when clustering)
```

The demand adjustment (DMSS Appendix C) is preserved, contracted onto the
scalar score as `-2 (h_i - h_bar)' B_m' W g_m` with
`B_m = G_k (H'_wd H)^{-1} H'_wd` (the old `adjustment_value` without its
`W^{1/2}` prefactor; the F-path uses `W B_m`, same math, float
reassociation only). MCS covariances now come from the scalar scores;
`model_confidence_set_variance` dropped its `/2` (exact cancellation in
the correlation normalization — the sigma_mcs matrix is unchanged).

I verified the math independently before implementing: the memo's scalar
score is exactly the delta method on `Q_hat` in both `g_hat` and `W_hat`,
and the Sylvester-equation correction in `variance_proof_note.pdf`
collapses to the same scalar score after contraction with `2 g' W^{1/2}`.

## Monte Carlo evidence

Adverse 3-instrument design (eigenvalues of E[zz'] = 25 / 1 / 0.04,
generic non-eigenvector moment directions, fixed-g null Q1 = Q2 > 0,
n = 4000, 4000 reps):

|                       | old W^{3/4} formula | scalar score |
|-----------------------|---------------------|--------------|
| sd(T_RV) under null   | 1.49                | 0.98         |
| rejection @ nominal 5%| 18.9%               | 4.2%         |

In the well-conditioned shipped synthetics the change is tiny (~1e-6
absolute on TRV), which is why the old formula survived the snapshot
suite: the old nominal-size test had g1 = g2 = 0, a degenerate design in
which every disputed term vanishes.

## Snapshot deltas (old → new baseline, max |delta| per field)

| snapshot                        | TRV     | MCS_pvalues | F       | g       | Q       |
|---------------------------------|---------|-------------|---------|---------|---------|
| analytical_base                 | 2.4e-06 | 1.8e-06     | 4e-16   | 2e-17   | 1e-18   |
| analytical_clustering           | 2.4e-06 | 1.9e-06     | 9e-16   | 2e-17   | 1e-18   |
| analytical_base_fe              | 5.2e-06 | 4.0e-06     | 3e-15   | 7e-18   | 7e-19   |
| analytical_scale                | 9.2e-02 | 6.6e-02     | 4.6e-02 | 3e-16   | 4e-19   |
| first_stage_pyblp_path          | 5.7e-09 | 4.5e-09     | 2.3e-11 | 2e-14   | 1e-16   |
| first_stage_demand_params_path  | 5.7e-09 | 4.5e-09     | 2.3e-11 | 2e-14   | 1e-16   |
| nested_logit_vertical           | 8e-16   | 7e-16       | 1e-15   | 4e-17   | 5e-18   |

Reading guide:
- TRV / MCS moves are the deliberate formula change.
- F moves only at float-reassociation level (2.3e-11) on the
  demand-adjustment snapshots — `W^{1/2}(W^{1/2} B)` became `W B`.
- g / Q / markups / taus / marginal_cost are bit-level unchanged
  (the 1e-14 g deltas on the first_stage snapshots are pyblp optimizer
  run-to-run noise, present before this change).
- **analytical_scale caveat:** its old baseline was macOS + numpy 1.x and
  it was xfailed on Linux/numpy2 because of a documented ~2.5-3% LAPACK
  cancellation drift on F. Regenerating on this box (Linux + numpy 2.3.5,
  OpenBLAS wheel) makes Linux+numpy2 the new baseline; its F delta above
  is that platform drift, not the variance change (the scale fixture runs
  with demand_adjustment=False, so the F path is untouched there). The
  xfail condition in `tests/test_snapshots.py` was inverted accordingly
  (`_analytical_scale_expected_to_drift` now marks non-Linux / numpy<2 as
  drift-expected). Its TRV delta (9.2e-2) combines the q_tilde-correction
  removal (the endogenous-cost path) with the platform drift.

## Code / test surface

- `pyRVtest/solve/test_engine.py` — core change; deleted W_12/W_34
  eigendecomposition, psi, the q_tilde term1/term2 correction, and the
  `_skip_appendix_b` gate. `compute_block_gram` reused with (M, N, 1)
  scores. F/rho/CV/reliability path untouched.
- `tests/test_analytical.py` — the three hand-compute references now
  re-derive the scalar-score sigma^2 independently.
- `tests/test_rv_variance.py` (new) — basis invariance under Z -> Z T
  (old formula was not invariant), K=1 exact equivalence with the old
  formula, cluster-sum equality (memo eq. (13)), MCS covariance
  consistency incl. exact reproduction of package MCS p-values.
- `tests/test_size_power.py` — degenerate size arm (Bertrand vs
  Bertrand + 1e-6 noise, g1 = g2 = 0) replaced with a nondegenerate
  fixed-g null (equal-fit generic deviations, unequal-eigenvalue
  instruments); added a sd(TRV) in [0.85, 1.15] assertion.
- `docs/math.rst` RV section rewritten; `docs/notebooks/replication_CarRV.py`
  legacy `_skip_appendix_b` section removed; CHANGELOG entry under 0.4.0.

## Follow-ups for coauthors

- The paper's Appendix B / Appendix J text needs the corresponding
  corrections (see Section 7 of `notes/Memo_appendixB.pdf`).
- `tests/replication/test_dmss_yogurt.py` EXPECTED_* values are still
  None ("Awaiting Lorenzo"); when filled, they must be computed under
  the scalar-score variance.
