"""Tests for the RV first-stage correction when demand is estimated.

These tests verify that pyRVtest correctly implements the DMSS (2024, QE) Appendix C
equation (77) first-stage correction:

    psi_tilde_m,i = psi_hat_m,i - W^{1/2} G_m Lambda [h(theta_hat) - h_i(theta_hat)]

where G_m = -(1/n) z_hat' grad_theta Delta_hat_m(theta_hat_D), i.e. the gradient of
the sample markup w.r.t. demand parameters theta.

The canonical object that both pyRVtest paths must compute is d(markup)/d(theta) as a
sample function of theta, with observed data held fixed.

Two paths exist:

1. PyBLP finite-diff: perturb theta, call compute_delta() (which runs the BLP
   contraction so s_impl(delta(theta+eps), theta+eps) = s_obs), then recompute markup
   via FOC at observed shares.

2. demand_params analytical: compute d(markup)/d(theta) in closed form via implicit
   differentiation of the FOC, using analytical d(D)/d(theta).

The claim: both paths compute the same d(markup)/d(theta) object to first order.

Test 1 verifies the PyBLP path premise empirically (compute_delta restores observed
shares at perturbed theta). Test 2 verifies end-to-end equivalence of TRV/F when
demand_adjustment=True. Test 3 verifies the premise at the FOC level (same g, same h).

See .claude/plans/v0.4-refactor.md Section 3 note 4 for the theoretical discussion.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyRVtest


# ---------------------------------------------------------------------------
# Shared fixture: logit DGP estimated via PyBLP, with nonzero xi for a nontrivial
# demand adjustment. Used by all tests in this module.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def logit_dgp_and_estimation():
    """Logit DGP with an explicit excluded demand instrument.

    The demand-adjustment first-stage correction (DMSS Appendix C eq. 77) is only
    non-trivial when there is variation in the instruments beyond the exogenous
    regressors in X1. Under pure logit with X1 = '1 + prices + x1' and no
    demand_instruments, PyBLP's ZD reduces to (intercept, x1) = X1_exogenous; the
    2SLS-style projection in the first-stage correction zeros out H by
    construction, and the correction vanishes. We add an explicit
    demand_instruments0 column (rival_x1) to avoid this degenerate case.

    ZD columns after this fixture: (intercept, x1, rival_x1). Both paths reference
    these same columns by name.

    Returns (data, pyblp_results, alpha_hat, beta_x_hat).
    """
    import pyblp
    pyblp.options.verbose = False

    # Larger sample for stable GMM estimation. Small T leads to wildly wrong alpha
    # estimates (e.g., +21 instead of the true -2) which crashes downstream
    # demand_params validation (requires alpha < 0).
    T, J = 200, 3
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)
    id_data = pd.DataFrame({"market_ids": market_ids, "firm_ids": firm_ids})

    X1 = pyblp.Formulation("1 + prices + x1")
    X3 = pyblp.Formulation("1 + z1")

    simulation = pyblp.Simulation(
        product_formulations=(X1, None, X3),
        beta=[0, -2, 1],
        gamma=[1, 0.5],
        xi_variance=0.3,
        omega_variance=0.2,
        correlation=0.0,
        product_data=id_data,
        seed=99,
    )
    sim_results = simulation.replace_endogenous()
    data = pd.DataFrame(pyblp.data_to_dict(sim_results.product_data))

    # Rival characteristics for conduct-testing instruments AND as an excluded
    # demand instrument (so the demand estimation is over-identified).
    # Vectorize to avoid per-row DataFrame ops on larger samples.
    for t in range(T):
        idx = np.where(data["market_ids"] == t)[0]
        for j in idx:
            rival = [i for i in idx if i != j]
            data.loc[j, "rival_x1"] = data.loc[rival, "x1"].mean()
            data.loc[j, "rival_z1"] = data.loc[rival, "z1"].mean()
    # Additional cost-based instrument: rival z1 at a different moment
    data["rival_z1_sq"] = data["rival_z1"] ** 2

    # Explicit intercept for demand_params to reference by name.
    data["intercept"] = 1.0
    # Add two excluded demand instruments to over-identify demand (4 moments,
    # 3 parameters). Over-identification makes the first-stage-correction Λ
    # depend on the weight matrix, so the demand_adjustment_weight option toggle
    # produces a meaningful TRV difference.
    data["demand_instruments0"] = data["rival_x1"]
    data["demand_instruments1"] = data["rival_z1"] ** 2

    problem = pyblp.Problem((X1,), product_data=data)
    pyblp_results = problem.solve(method="1s")

    alpha_hat = float(pyblp_results.beta[pyblp_results.beta_labels.index("prices")])
    beta_x_hat = float(pyblp_results.beta[pyblp_results.beta_labels.index("x1")])

    return data, pyblp_results, alpha_hat, beta_x_hat


# ---------------------------------------------------------------------------
# Test 1: compute_delta restores observed shares at perturbed theta.
#
# This is the mechanical premise underlying the equivalence between the PyBLP
# finite-diff path and the demand_params analytical path. If compute_delta does
# NOT restore observed shares after perturbation, the paths compute different
# objects and equivalence fails.
# ---------------------------------------------------------------------------


class TestComputeDeltaRestoresShares:
    """Verify that after perturbing a demand parameter and calling compute_delta,
    the resulting implied shares match observed shares.

    If this test passes, the Jacobian at the perturbed (theta, delta) point is
    effectively evaluated at observed shares, which matches what the analytical
    path does by construction.
    """

    def test_alpha_perturbation_restores_shares(self, logit_dgp_and_estimation):
        """Perturb alpha (price coefficient), verify implied shares = observed."""
        import pyblp
        pyblp.options.verbose = False

        _, r, alpha_hat, _ = logit_dgp_and_estimation

        observed_shares = r.problem.products.shares.flatten().copy()
        original_beta = r.beta.copy()
        original_delta = r.delta.copy()

        price_idx = r.beta_labels.index("prices")
        epsilon = 1e-3

        # Perturb alpha, restore delta via compute_delta, check implied shares
        try:
            r._beta[price_idx] = alpha_hat + epsilon
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                r._delta = r.compute_delta()
                implied = r.compute_shares().flatten()

            # Core assertion: after compute_delta, implied shares match observed
            np.testing.assert_allclose(
                implied, observed_shares, atol=1e-6,
                err_msg="compute_delta did not restore observed shares after alpha perturbation"
            )
        finally:
            r._beta[:] = original_beta
            r._delta = original_delta

    def test_alpha_perturbation_both_directions(self, logit_dgp_and_estimation):
        """Verify behavior is symmetric: +eps and -eps both restore shares."""
        import pyblp
        pyblp.options.verbose = False

        _, r, alpha_hat, _ = logit_dgp_and_estimation

        observed_shares = r.problem.products.shares.flatten().copy()
        original_beta = r.beta.copy()
        original_delta = r.delta.copy()

        price_idx = r.beta_labels.index("prices")
        epsilon = 1e-3

        try:
            for direction, sign in [("positive", +1), ("negative", -1)]:
                r._beta[price_idx] = alpha_hat + sign * epsilon
                with contextlib.redirect_stdout(open(os.devnull, "w")):
                    r._delta = r.compute_delta()
                    implied = r.compute_shares().flatten()
                np.testing.assert_allclose(
                    implied, observed_shares, atol=1e-6,
                    err_msg=f"compute_delta failed to restore shares at {direction} perturbation"
                )
        finally:
            r._beta[:] = original_beta
            r._delta = original_delta


# ---------------------------------------------------------------------------
# Test 2: End-to-end equivalence of TRV/F with demand_adjustment=True.
#
# The existing TestDemandParamsVsPyBLP in test_demand_params.py uses
# demand_adjustment=False. This fills the gap by exercising the DMSS Appendix C
# correction on the same DGP.
# ---------------------------------------------------------------------------


class TestFirstStageEquivalenceWithDemandAdjustment:
    """End-to-end equivalence: same logit DGP, both paths, demand_adjustment=True.

    If DMSS Appendix C equation (77) is implemented identically in both paths,
    TRV, F, and g should match to finite-difference precision.
    """

    @pytest.fixture(scope="class")
    def comparison_results(self, logit_dgp_and_estimation):
        """Solve the same DGP via both paths with demand_adjustment=True."""
        data, pyblp_results, alpha, beta_x = logit_dgp_and_estimation

        models = (
            pyRVtest.ModelFormulation(
                model_downstream="bertrand", ownership_downstream="firm_ids"
            ),
            pyRVtest.ModelFormulation(model_downstream="perfect_competition"),
        )
        common = dict(
            cost_formulation=pyRVtest.Formulation("1 + z1"),
            instrument_formulation=pyRVtest.Formulation("0 + rival_x1 + rival_z1"),
            model_formulations=models,
            product_data=data,
        )

        pyRVtest.options.verbose = False

        # Path 1: PyBLP with demand_adjustment=True
        r1 = pyRVtest.Problem(
            **common, demand_results=pyblp_results
        ).solve(demand_adjustment=True)

        # Path 2: demand_params with demand_adjustment=True
        # Must pass the full beta vector matching pyblp's X1 columns, EXCLUDING prices.
        # pyblp X1 = '1 + prices + x1'; X1 exogenous = (intercept, x1); so we need both.
        beta_0 = float(
            pyblp_results.beta[pyblp_results.beta_labels.index("1")]
        )
        r2 = pyRVtest.Problem(
            **common,
            demand_params={
                "alpha": alpha,
                "sigma": [],
                "beta": np.array([beta_0, beta_x]),
                "x_columns": ["intercept", "x1"],
                # PyBLP orders ZD with demand_instruments first, then X1 exogenous.
                "demand_instrument_columns": ["rival_x1", "rival_z1_sq", "intercept", "x1"],
            },
        ).solve(demand_adjustment=True)

        pyRVtest.options.verbose = True
        return r1, r2

    def test_markups_match_with_adjustment(self, comparison_results):
        """Markups should still match to machine precision (they don't depend on adj)."""
        r1, r2 = comparison_results
        for m in range(2):
            np.testing.assert_allclose(
                r1.markups[m].flatten(), r2.markups[m].flatten(), atol=1e-8,
                err_msg=f"Markups differ for model {m}"
            )

    def test_g_matches_with_adjustment(self, comparison_results):
        """GMM moments should match to machine precision."""
        r1, r2 = comparison_results
        np.testing.assert_allclose(
            r1.g[0], r2.g[0], atol=1e-8, err_msg="g differs between paths"
        )

    def test_trv_matches_with_adjustment(self, comparison_results):
        """TRV with demand adjustment must agree to finite-diff precision.

        This is the core DMSS Appendix C correctness check. If this fails, either
        the paths compute different G_m or different Lambda.
        """
        r1, r2 = comparison_results
        np.testing.assert_allclose(
            r1.TRV[0][0, 1], r2.TRV[0][0, 1], atol=1e-4,
            err_msg=(
                "TRV with demand_adjustment=True differs between PyBLP and demand_params. "
                "DMSS Appendix C eq (77) implementation may disagree across paths."
            )
        )

    def test_f_matches_with_adjustment(self, comparison_results):
        """F-statistic with demand adjustment must agree."""
        r1, r2 = comparison_results
        np.testing.assert_allclose(
            r1.F[0][0, 1], r2.F[0][0, 1], atol=1e-4,
            err_msg="F differs with demand_adjustment=True"
        )

    def test_updated_W_option_reproduces_prior_behavior(self, logit_dgp_and_estimation):
        """With options.demand_adjustment_weight='updated_W' the PyBLP path uses the
        pre-v0.3.3 weight matrix. This gives a different TRV than the default 'W',
        and the difference should be non-trivial (confirming the option is wired).

        Users who need to reproduce prior output for validation can flip this option.
        """
        data, pyblp_results, alpha, beta_x = logit_dgp_and_estimation
        models = (
            pyRVtest.ModelFormulation(model_downstream="bertrand", ownership_downstream="firm_ids"),
            pyRVtest.ModelFormulation(model_downstream="perfect_competition"),
        )
        common = dict(
            cost_formulation=pyRVtest.Formulation("1 + z1"),
            instrument_formulation=pyRVtest.Formulation("0 + rival_x1 + rival_z1"),
            model_formulations=models, product_data=data,
        )
        pyRVtest.options.verbose = False
        orig_weight = getattr(pyRVtest.options, "demand_adjustment_weight", "W")
        try:
            pyRVtest.options.demand_adjustment_weight = "W"
            r_new = pyRVtest.Problem(**common, demand_results=pyblp_results).solve(demand_adjustment=True)

            pyRVtest.options.demand_adjustment_weight = "updated_W"
            r_old = pyRVtest.Problem(**common, demand_results=pyblp_results).solve(demand_adjustment=True)
        finally:
            pyRVtest.options.demand_adjustment_weight = orig_weight
            pyRVtest.options.verbose = True

        # Both should be finite
        assert not np.isnan(r_new.TRV[0][0, 1])
        assert not np.isnan(r_old.TRV[0][0, 1])
        # And different (otherwise the option is a no-op)
        assert not np.isclose(r_new.TRV[0][0, 1], r_old.TRV[0][0, 1], atol=1e-10), (
            f"demand_adjustment_weight option is a no-op: "
            f"W={r_new.TRV[0][0, 1]}, updated_W={r_old.TRV[0][0, 1]}"
        )

    def test_adjustment_actually_changes_trv(self, logit_dgp_and_estimation):
        """Sanity: demand_adjustment=True should give a different TRV than =False.

        If this test shows TRV is unchanged, then the demand adjustment machinery
        is silently a no-op and the equivalence tests above are not checking anything.
        """
        data, pyblp_results, alpha, beta_x = logit_dgp_and_estimation

        models = (
            pyRVtest.ModelFormulation(
                model_downstream="bertrand", ownership_downstream="firm_ids"
            ),
            pyRVtest.ModelFormulation(model_downstream="perfect_competition"),
        )
        common = dict(
            cost_formulation=pyRVtest.Formulation("1 + z1"),
            instrument_formulation=pyRVtest.Formulation("0 + rival_x1 + rival_z1"),
            model_formulations=models,
            product_data=data,
            demand_params={
                "alpha": alpha,
                "sigma": [],
                "beta": np.array([beta_x]),
                "x_columns": ["x1"],
                # PyBLP orders ZD with demand_instruments first, then X1 exogenous.
                "demand_instrument_columns": ["rival_x1", "rival_z1_sq", "intercept", "x1"],
            },
        )

        pyRVtest.options.verbose = False
        r_no = pyRVtest.Problem(**common).solve(demand_adjustment=False)
        r_adj = pyRVtest.Problem(**common).solve(demand_adjustment=True)
        pyRVtest.options.verbose = True

        assert not np.isclose(r_no.TRV[0][0, 1], r_adj.TRV[0][0, 1], atol=1e-10), (
            f"demand_adjustment appears to be a no-op: TRV = {r_no.TRV[0][0, 1]} either way"
        )


# ---------------------------------------------------------------------------
# Test 3: Isolated d(markup)/d(theta) equivalence.
#
# If Test 2 fails, this test localizes the disagreement. If the markup gradient
# agrees between paths but TRV/F don't, the bug is downstream (in how H, h, or
# Lambda are assembled). If this test fails, the bug is in the gradient itself.
# ---------------------------------------------------------------------------


class TestMarkupGradientEquivalence:
    """Directly compare d(markup)/d(theta) between the two paths.

    Both paths expose a _compute_*_demand_adjustment method that returns
    (gradient_markups, H_prime_wd, H, h_i, h). Compare gradient_markups arrays.
    """

    def test_gradient_markups_match(self, logit_dgp_and_estimation):
        """The two demand-adjustment methods must produce the same gradient_markups.

        Finite-diff (PyBLP) vs. analytical (demand_params) — both target
        d(markup)/d(alpha). Agreement expected to O(eps) finite-diff precision.
        """
        data, pyblp_results, alpha, beta_x = logit_dgp_and_estimation

        models = (
            pyRVtest.ModelFormulation(
                model_downstream="bertrand", ownership_downstream="firm_ids"
            ),
            pyRVtest.ModelFormulation(model_downstream="perfect_competition"),
        )
        common = dict(
            cost_formulation=pyRVtest.Formulation("1 + z1"),
            instrument_formulation=pyRVtest.Formulation("0 + rival_x1 + rival_z1"),
            model_formulations=models,
            product_data=data,
        )

        pyRVtest.options.verbose = False

        # Build both Problem instances
        problem_pyblp = pyRVtest.Problem(**common, demand_results=pyblp_results)
        problem_dp = pyRVtest.Problem(
            **common,
            demand_params={
                "alpha": alpha,
                "sigma": [],
                "beta": np.array([beta_x]),
                "x_columns": ["x1"],
                # PyBLP orders ZD with demand_instruments first, then X1 exogenous.
                "demand_instrument_columns": ["rival_x1", "rival_z1_sq", "intercept", "x1"],
            },
        )

        # Compute markups first (required input for both adjustment methods)
        markups_pyblp, _, _ = problem_pyblp._perturb_and_build_markups()
        markups_dp, _, _ = problem_dp._perturb_and_build_markups()

        M = problem_pyblp.M
        N = problem_pyblp.N
        advalorem_tax_adj = [np.ones((N, 1)) for _ in range(M)]
        cost_scaling = [np.zeros((N, 1)) for _ in range(M)]

        # Call the two gradient methods directly
        grad_pyblp, *_ = problem_pyblp._compute_demand_adjustment_gradient(
            N, advalorem_tax_adj, cost_scaling, marginal_cost_base=None
        )
        grad_dp, *_ = problem_dp._compute_analytical_demand_adjustment(
            M, N, markups_dp, advalorem_tax_adj, cost_scaling
        )

        pyRVtest.options.verbose = True

        # Both paths should have same theta ordering in this logit DGP
        # (no sigma, no pi, no rho; just alpha → single parameter)
        assert grad_pyblp.shape == grad_dp.shape, (
            f"gradient_markups shape mismatch: {grad_pyblp.shape} vs {grad_dp.shape}"
        )

        # Compare per-model gradients. Tolerance: O(eps) finite-diff error.
        for m in range(M):
            np.testing.assert_allclose(
                grad_pyblp[m], grad_dp[m], atol=1e-4,
                err_msg=(
                    f"d(markup)/d(theta) disagrees for model {m}. "
                    f"PyBLP finite-diff vs demand_params analytical diverged. "
                    f"DMSS Appendix C requires both to compute nabla_theta Delta_hat_m."
                )
            )
