"""Economy-level conduct testing problem functionality."""

import abc
import contextlib
import itertools
import math
import os
import time
from typing import Mapping, Optional, Sequence

import numpy as np
from pyblp.utilities.algebra import precisely_identify_collinearity
from pyblp.utilities.basics import Array, RecArray, format_seconds, output
from scipy.linalg import inv, fractional_matrix_power
from scipy.stats import norm
import statsmodels.api as sm

from .economy import Economy
from .. import options
from ..configurations.formulation import Formulation, ModelFormulation
from ..construction import build_markups
from ..primitives import Models, Products, read_critical_values_tables
from ..results.problem_results import ProblemResults


class ProblemEconomy(Economy):
    """An abstract firm conduct testing problem."""

    @abc.abstractmethod
    def __init__(
            self, cost_formulation: Formulation, instrument_formulation: Sequence[Formulation],
            model_formulations: Sequence[ModelFormulation], products: RecArray, models: RecArray,
            demand_results: Mapping, markups: RecArray) -> None:
        """Initialize the underlying economy with product and agent data."""
        super().__init__(
            cost_formulation, instrument_formulation, model_formulations, products, models, demand_results, markups
        )

    def solve(
            self, demand_adjustment: Optional[bool] = False, 
            clustering_adjustment: Optional[bool] = False,
            costs_type: Optional[str] = 'linear',
            mc_correction:Optional[Array] = None
    ) -> ProblemResults:
        r"""Solve the problem.

        Given demand estimates from PyBLP, we compute implied markups for each model :math:`m` being tested. Marginal
        cost is a linear function of observed cost shifters and an unobserved shock.

        The rest of the testing procedure is done for each pair of models, for each set of instruments. A GMM measure of
        fit is computed for each model-instrument pair. This measure of fit is used to construct the test statistic.

        Parameters
        ----------
        demand_adjustment: Optional[bool]
            (optional, default is False) Configuration that allows user to specify whether to compute a two-step demand
            adjustment. Options are True or False.
        clustering_adjustment: Optional[str]
            (optional, default is unadjusted) Configuration that specifies whether to compute clustered standard errors.
            Options are True or False.

        Returns
        -------
        `ProblemResults`
            :class:`ProblemResults` of the solved problem.

        """

        # keep track of how long it takes to solve the problem
        output("Solving the problem ...")
        step_start_time = time.time()

        # initialize constants and precomputed values
        M = self.M
        N = self.N
        L = self.L
        markups = self.markups
        critical_values_power, critical_values_size = read_critical_values_tables()

        # validate settings
        if not isinstance(demand_adjustment, bool):
            raise TypeError("demand_adjustment must be a boolean (one of True or False).")
        if not isinstance(clustering_adjustment, bool):
            raise TypeError("clustering_adjustment must be a boolean (one of True or False).")
        if clustering_adjustment and np.shape(self.products.clustering_ids)[1] != 1:
            raise ValueError("product_data.clustering_ids must be specified with clustering_adjustment True.")
        for m in range(M):
            if self.model_formulations[m]._user_supplied_markups is not None:
                if clustering_adjustment or demand_adjustment:
                    raise ValueError(
                        "If using own markups, demand_adjustment and clustering_adjustment should be False."
                    )

        # initialize variables to be computed
        markups_upstream = np.zeros(M, dtype=options.dtype)
        markups_downstream = np.zeros(M, dtype=options.dtype)
        markups_orthogonal = np.zeros((M, N), dtype=options.dtype)
        prices_orthogonal = np.zeros((M, N), dtype=options.dtype)
        marginal_cost_orthogonal = np.zeros((M, N), dtype=options.dtype)
        omega = np.zeros((M, N), dtype=options.dtype)
        tau_list = np.zeros((M, self.products.w.shape[1]), dtype=options.dtype)
        prices_errors = np.zeros(M, dtype=options.dtype)
        markups_errors = np.zeros(M, dtype=options.dtype)
        marginal_cost_errors = np.zeros(M, dtype=options.dtype)

        # initialize tax-related variables
        prices_effective = [None] * M
        markups_effective = [None] * M
        advalorem_tax_adj = [None] * M

        # if there are no markups, compute them
        if markups[0] is None:
            print('Computing Markups ... ')
            markups, markups_downstream, markups_upstream = build_markups(
                self.products, self.demand_results, self.models["models_downstream"],
                self.models["ownership_downstream"], self.models["models_upstream"], self.models["ownership_upstream"],
                self.models["vertical_integration"], self.models["custom_model_specification"],
                self.models["user_supplied_markups"]
            )

        # for each model, use computed markups to compute the marginal costs
        marginal_cost = self.products.prices - markups

        # for the setting with taxes, adjust the markup computation to account for marginal costs
        unit_tax = self.models["unit_tax"]
        advalorem_tax = self.models["advalorem_tax"]
        cost_scaling = self.models["cost_scaling"]

        # for each model, compute tax adjustment
        for m in range(M):
            condition = self.models["advalorem_payer"][m] == "consumer"
            advalorem_tax_adj[m] = 1 / (1 + advalorem_tax[m]) if condition else (1 - advalorem_tax[m])
            prices_effective[m] = (advalorem_tax_adj[m] * self.products.prices / (1 + cost_scaling[m]) - unit_tax[m])
            markups_effective[m] = (advalorem_tax_adj[m]/(1 + cost_scaling[m])) * markups[m]
            marginal_cost[m] = prices_effective[m] - markups_effective[m]

	# transform marginal cost based on specificiation assumption
        if costs_type=="log" and not demand_adjustment:
            if np.any(marginal_cost<0):
                raise ValueError(
                    "Can't generate log costs with negative marginal cost!"
                    )
            else:                
                marginal_cost=np.log(marginal_cost)
        
        # include any marginal cost correction
        if mc_correction is not None:
            if not marginal_cost.shape==mc_correction.shape:
                raise ValueError("The dimensions of marginal cost correction don't match the dimensions of the marginal cost.")
            marginal_cost=marginal_cost + mc_correction

        # absorb any cost fixed effects from prices, markups, and instruments
        if self._absorb_cost_ids is not None:
            output("Absorbing cost-side fixed effects ...")
            self.products.w, w_errors = self._absorb_cost_ids(self.products.w)

            # TODO: put this into a loop
            for m in range(M):
                value, error = self._absorb_cost_ids(prices_effective[m])
                prices_errors[m] = np.nan if not error else error

                value, error = self._absorb_cost_ids(markups_effective[m])
                markups_orthogonal[m] = np.squeeze(value)
                markups_errors[m] = np.nan if not error else error

                value, error = self._absorb_cost_ids(marginal_cost[m])
                marginal_cost_orthogonal[m] = np.squeeze(value)
                marginal_cost_errors[m] = np.nan if not error else error
        else:
            markups_orthogonal = markups_effective.copy()
            marginal_cost_orthogonal = marginal_cost

        # residualize prices, markups, and instruments w.r.t cost shifters w and recover the tau parameters in cost
        #   regression on w
        # TODO: only do this if w is not 0
        if self.products.w.any():
            for m in range(M):
                results = sm.OLS(markups_orthogonal[m], self.products.w).fit()
                markups_orthogonal[m] = results.resid
                results = sm.OLS(marginal_cost_orthogonal[m], self.products.w).fit()
                omega[m] = results.resid
                tau_list[m] = results.params
        else:
            markups_orthogonal = np.reshape(markups_orthogonal, [M, N])
            omega = np.reshape(marginal_cost_orthogonal, [M, N])

        # if user specifies demand adjustment, account for two-step estimation in the standard errors by computing the
        #   finite difference approximation to the derivative of markups with respect to theta
        if demand_adjustment:
            ZD = self.demand_results.problem.products.ZD
            WD = self.demand_results.updated_W
            h = self.demand_results.moments
            h_i = ZD * self.demand_results.xi
            K2 = self.demand_results.problem.K2
            D = self.demand_results.problem.D

            # check whether price is in linear parameters, and if it is remove it
            XD = self.demand_results.problem.products.X1
            XD_column_names = self.demand_results.problem.products.dtype.fields['X1'][2]
            price_in_linear_parameters = 'prices' in XD_column_names
            if price_in_linear_parameters:
                XD = np.delete(XD, XD_column_names.index('prices'), 1)

            # add price to the gradient
            partial_y_theta = (np.append(
                    self.demand_results.xi_by_theta_jacobian, -self.demand_results.problem.products.prices, axis=1
                ) if price_in_linear_parameters else self.demand_results.xi_by_theta_jacobian)

            # absorb fixed effects if they are specified
            if self.demand_results.problem.ED > 0:
                partial_y_theta = self.demand_results.problem._absorb_demand_ids(partial_y_theta)
                partial_y_theta = np.reshape(
                    partial_y_theta[0], [N, len(self.demand_results.theta) + int(price_in_linear_parameters)]
                )

            # if there are linear parameters, adjust them
            if not XD.shape[1]:
                partial_xi_theta = partial_y_theta
            else:
                try:
                    product = XD @ inv(XD.T @ ZD @ WD @ ZD.T @ XD) @ (XD.T @ ZD @ WD @ ZD.T @ partial_y_theta)
                    partial_xi_theta = partial_y_theta - product
                except Exception:
                    output(
                        "Dimension mismatch occurred. This can happen if you specify a supply side in the demand "
                        "estimation."
                    )
            H = 1 / N * (np.transpose(ZD) @ partial_xi_theta)
            H_prime = np.transpose(H)
            H_prime_wd = H_prime @ WD

            # build adjustment to psi for each model
            epsilon = options.finite_differences_epsilon
            G_m = [None] * M
            gradient_markups = np.zeros(
                (M, N, len(self.demand_results.theta) + int(price_in_linear_parameters)), dtype=options.dtype
            )

            # compute sigma
            theta_index = 0
            delta_estimate = self.demand_results.delta

            def markups_computation(markups_m):
                """Compute markups for setting with taxes."""
                denominator = (1 + cost_scaling[m])
                computation = (advalorem_tax_adj[m] * markups_m)
                return computation / denominator

            # loop over pairs of nonlinear demand characteristics, and recompute markups with perturbations if
            #   the demand coefficient estimates for sigma are not zero
            for (i, j) in itertools.product(range(K2), range(K2)):
                if not self.demand_results.sigma[i, j] == 0:
                    sigma_initial = self.demand_results.sigma[i, j]

                    # reduce sigma by small increment, update delta, and recompute markups
                    self.demand_results._sigma[i, j] = sigma_initial - epsilon / 2
                    with contextlib.redirect_stdout(open(os.devnull, 'w')):
                        delta_new = self.demand_results.compute_delta()
                    self.demand_results._delta = delta_new
                    markups_l, md, ml = build_markups(
                        self.products, self.demand_results, self.models["models_downstream"],
                        self.models["ownership_downstream"], self.models["models_upstream"],
                        self.models["ownership_upstream"], self.models["vertical_integration"],
                        self.models["custom_model_specification"], self.models["user_supplied_markups"]
                    )

                    # increase sigma by small increment, update delta, and recompute markups
                    self.demand_results._sigma[i, j] = sigma_initial + epsilon / 2
                    with contextlib.redirect_stdout(open(os.devnull, 'w')):
                        delta_new = self.demand_results.compute_delta()
                    self.demand_results._delta = delta_new
                    markups_u, mu, mu = build_markups(
                        self.products, self.demand_results, self.models["models_downstream"],
                        self.models["ownership_downstream"], self.models["models_upstream"],
                        self.models["ownership_upstream"], self.models["vertical_integration"],
                        self.models["custom_model_specification"], self.models["user_supplied_markups"]
                    )

                    # compute markup perturbations for taxes
                    for m in range(M):
                        markups_u[m] = markups_computation(markups_u[m])
                        markups_l[m] = markups_computation(markups_l[m])

                    # compute first difference approximation of derivative of markups
                    gradient_markups = self._compute_first_difference_markups(
                        markups_u, markups_l, epsilon, theta_index, gradient_markups
                    )
                    self.demand_results._sigma[i, j] = sigma_initial
                    theta_index = theta_index + 1

            # loop over nonlinear demand characteristics and demographics, and recompute markups with perturbations if
            #   the demand coefficient estimates for pi are not zero
            for (i, j) in itertools.product(range(K2), range(D)):
                if not self.demand_results.pi[i, j] == 0:
                    pi_initial = self.demand_results.pi[i, j]
                    perturbations = [pi_initial - epsilon / 2, pi_initial + epsilon / 2]
                    markups_l, md, ml = self._compute_perturbation(i, j, perturbations[0])
                    markups_u, mu, mu = self._compute_perturbation(i, j, perturbations[1])
                    for m in range(M):
                        markups_u[m] = markups_computation(markups_u[m])
                        markups_l[m] = markups_computation(markups_l[m])

                    gradient_markups = self._compute_first_difference_markups(
                        markups_u, markups_l, epsilon, theta_index, gradient_markups
                    )
                    self.demand_results._pi[i, j] = pi_initial
                    theta_index = theta_index + 1
            self.demand_results._delta = delta_estimate

            # perturb alpha in negative (positive) direction and recompute markups
            price_index = [index for index, value in enumerate(self.demand_results.beta_labels) if value == 'prices']
            if price_index:
                alpha_initial = self.demand_results.beta[price_index].copy()
                self.demand_results._beta[price_index] = alpha_initial - epsilon / 2
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    delta_new = self.demand_results.compute_delta()
                self.demand_results._delta = delta_new
                markups_l, md, ml = build_markups(
                    self.products, self.demand_results, self.models["models_downstream"],
                    self.models["ownership_downstream"], self.models["models_upstream"],
                    self.models["ownership_upstream"], self.models["vertical_integration"],
                    self.models["custom_model_specification"], self.models["user_supplied_markups"]
                )
                self.demand_results._beta[price_index] = alpha_initial + epsilon / 2
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    delta_new = self.demand_results.compute_delta()
                self.demand_results._delta = delta_new
                markups_u, mu, mu = build_markups(
                    self.products, self.demand_results, self.models["models_downstream"],
                    self.models["ownership_downstream"], self.models["models_upstream"],
                    self.models["ownership_upstream"], self.models["vertical_integration"],
                    self.models["custom_model_specification"], self.models["user_supplied_markups"]
                )

                # compute markup perturbations for taxes
                for m in range(M):
                    markups_u[m] = markups_computation(markups_u[m])
                    markups_l[m] = markups_computation(markups_l[m])

                gradient_markups = self._compute_first_difference_markups(
                    markups_u, markups_l, epsilon, theta_index, gradient_markups
                )

                self.demand_results._beta[price_index] = alpha_initial
                theta_index = theta_index + 1

            # first differencing for the nesting parameter rho
            if len(self.demand_results.rho) != 0:
                rho_initial = self.demand_results.rho.copy()

                # perturb rho in the negative direction and recompute markups
                self.demand_results._rho = rho_initial - epsilon / 2
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    delta_new = self.demand_results.compute_delta()
                self.demand_results._delta = delta_new
                markups_l, md, ml = build_markups(
                    self.products, self.demand_results, self.models["models_downstream"],
                    self.models["ownership_downstream"], self.models["models_upstream"],
                    self.models["ownership_upstream"], self.models["vertical_integration"],
                    self.models["custom_model_specification"], self.models["user_supplied_markups"]
                )

                # perturb rho in the positive direction and recompute markups
                self.demand_results._rho = rho_initial + epsilon / 2
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    delta_new = self.demand_results.compute_delta()
                self.demand_results._delta = delta_new
                markups_u, mu, mu = build_markups(
                    self.products, self.demand_results, self.models["models_downstream"],
                    self.models["ownership_downstream"], self.models["models_upstream"],
                    self.models["ownership_upstream"], self.models["vertical_integration"],
                    self.models["custom_model_specification"], self.models["user_supplied_markups"]
                )

                # compute markup perturbations for taxes
                for m in range(M):
                    markups_u[m] = markups_computation(markups_u[m])
                    markups_l[m] = markups_computation(markups_l[m])

                gradient_markups = self._compute_first_difference_markups(
                    markups_u, markups_l, epsilon, theta_index, gradient_markups
                )
                self.demand_results._rho = rho_initial

        # initialize empty lists to store statistic-related values for each model
        g_list = [None] * L
        Q_list = [None] * L
        RV_numerator_list = [None] * L
        RV_denominator_list = [None] * L
        test_statistic_RV_list = [None] * L
        F_statistic_list = [None] * L
        unscaled_F_statistic_list = [None] * L
        MCS_p_values_list = [None] * L
        rho_list = [None] * L
        F_cv_size_list = [None] * L
        F_cv_power_list = [None] * L
        symbols_size_list = [None] * L
        symbols_power_list = [None] * L

        # compare models of conduct for each set of instruments
        for instrument in range(L):
            instruments = self.products["Z{0}".format(instrument)]
            K = np.shape(instruments)[1]

            # absorb any cost fixed effects from prices, markups, and instruments
            if self._absorb_cost_ids is not None:
                Z_orthogonal, Z_errors = self._absorb_cost_ids(instruments)
            else:
                Z_orthogonal = instruments
            if self.products.w.any():
                Z_residual = sm.OLS(Z_orthogonal, self.products.w).fit().resid
                Z_orthogonal = np.reshape(Z_residual, [N, K])

            # initialize variables to store GMM measure of fit Q_m for each model
            g = np.zeros((M, K), dtype=options.dtype)
            Q = np.zeros(M, dtype=options.dtype)

            # compute the weight matrix
            W_inverse = 1 / N * (Z_orthogonal.T @ Z_orthogonal)
            W_inverse = np.reshape(W_inverse, [K, K])
            weight_matrix = np.linalg.pinv(W_inverse) #Moore-Penrose pseudo-inverse

            # for each model compute GMM measure of fit
            for m in range(M):
                g[m] = 1 / N * (Z_orthogonal.T @ (omega[m]))
                Q[m] = g[m].T @ weight_matrix @ g[m]

            # compute the pairwise RV numerator
            test_statistic_numerator = np.zeros((M, M))
            for m in range(M):
                for i in range(m):
                    if i < m:
                        test_statistic_numerator[i, m] = math.sqrt(N) * (Q[i] - Q[m])

            # initialize the RV test statistic denominator and construct weight matrices
            test_statistic_denominator = np.zeros((M, M))
            covariance_mc = np.zeros((M, M))
            W_12 = fractional_matrix_power(weight_matrix, 0.5)
            W_34 = fractional_matrix_power(weight_matrix, 0.75)

            # compute psi, which is used in the estimator of the covariance between weighted moments
            psi = np.zeros((M, N, K), dtype=options.dtype)
            if demand_adjustment:
                adjustment_value = np.zeros((M, K, H_prime_wd.shape[1]), dtype=options.dtype)
            for m in range(M):
                psi_bar = W_12 @ g[m] - .5 * W_34 @ W_inverse @ W_34 @ g[m]
                W_34_Zg = Z_orthogonal @ W_34 @ g[m]
                W_34_Zg = W_34_Zg[:, np.newaxis]
                marginal_cost_orthogonal = (omega[m])
                marginal_cost_orthogonal = marginal_cost_orthogonal[:, np.newaxis]
                psi_i = (marginal_cost_orthogonal * Z_orthogonal) @ W_12 - 0.5 * W_34_Zg * (Z_orthogonal @ W_34.T)
                psi[m] = psi_i - np.transpose(psi_bar)

                # make a demand adjustment
                if demand_adjustment:
                    G_k = -1 / N * np.transpose(Z_orthogonal) @ gradient_markups[m]
                    G_m[m] = G_k
                    adjustment_value[m] = W_12 @ G_m[m] @ inv(H_prime_wd @ H) @ H_prime_wd
                    psi[m] = psi[m] - (h_i - np.transpose(h)) @ np.transpose(adjustment_value[m])

            # initialize model confidence set containers
            model_confidence_set = np.array(range(M))
            all_model_combinations = list(itertools.combinations(model_confidence_set, 2))
            number_model_combinations = np.shape(all_model_combinations)[0]
            model_confidence_set_variance = np.zeros([number_model_combinations, 1])

            # compute the RV test statistic denominator
            for m in range(M):
                for i in range(m):
                    if i < m:
                        variance_covariance = self._compute_variance_covariance(m, i, N, clustering_adjustment, psi)
                        weighted_variance = W_12 @ variance_covariance @ W_12
                        operations = np.array([1, 1, -2])
                        moments = np.array([
                            g[i].T @ weighted_variance[0] @ g[i],
                            g[m].T @ weighted_variance[1] @ g[m],
                            g[i].T @ weighted_variance[2] @ g[m]
                        ]).flatten()
                        sigma_squared = 4 * (operations.T @ moments)

                        # compute the covariance matrix for marginal costs
                        covariance_mc[i, m] = moments[2]
                        covariance_mc[m, i] = covariance_mc[i, m]
                        covariance_mc[m, m] = moments[1]
                        covariance_mc[i, i] = moments[0]
                        test_statistic_denominator[i, m] = math.sqrt(sigma_squared)

            # compute the pairwise RV test statistic
            rv_test_statistic = np.zeros((M, M))
            for (m, i) in itertools.product(range(M), range(M)):
                if i < m:
                    rv_test_statistic[i, m] = test_statistic_numerator[i, m] / test_statistic_denominator[i, m]
                else:
                    rv_test_statistic[i, m] = "NaN"

            # compute the pairwise F-statistic for each model
            unscaled_F = np.zeros((M, M))
            F = np.zeros((M, M))
            pi = np.zeros((K, M))
            phi = np.zeros([M, N, K])
            rho = np.zeros((M, M))
            F_cv_size = np.empty((M, M), dtype=object)
            F_cv_power = np.empty((M, M), dtype=object)
            symbols_size = np.empty((M, M), dtype=object)
            symbols_power = np.empty((M, M), dtype=object)
            for m in range(M):
                ols_results = sm.OLS(omega[m], Z_orthogonal).fit()
                pi[:, m] = ols_results.params
                e = np.reshape(ols_results.resid, [N, 1])
                phi[m] = (e * Z_orthogonal) @ weight_matrix
                if demand_adjustment:
                    phi[m] = phi[m] - (h_i - np.transpose(h)) @ np.transpose(W_12 @ adjustment_value[m])

            # compute the F statistic for each pair of models
            for (m, i) in itertools.product(range(M), range(M)):
                if i < m:
                    variance = self._compute_variance_covariance(m, i, N, clustering_adjustment, phi)
                    sigma = 1 / K * np.array([
                        np.trace(variance[0] @ W_inverse), np.trace(variance[1] @ W_inverse),
                        np.trace(variance[2] @ W_inverse)
                    ])
                    numerator_sqrt = (sigma[0] - sigma[1])
                    denominator_sqrt = np.sqrt((sigma[0] + sigma[1]) * (sigma[0] + sigma[1]) - 4 * sigma[2] ** 2) #TODO: handle negative values
                    rho[i, m] = numerator_sqrt / denominator_sqrt
                    rho_squared = np.square(rho[i, m])

                    # construct F statistic
                    operations = np.array([sigma[1], sigma[0], -2 * sigma[2]])
                    moments = np.array([
                        g[i].T @ weight_matrix @ g[i],
                        g[m].T @ weight_matrix @ g[m],
                        g[i].T @ weight_matrix @ g[m]
                    ]).flatten()
                    F_numerator = operations @ moments
                    F_denominator = (sigma[0] * sigma[1] - sigma[2] ** 2)
                    unscaled_F[i, m] = N / (2 * K) * F_numerator / F_denominator
                    F[i, m] = (1 - rho_squared) * N / (2 * K) * F_numerator / F_denominator

                    # pull out critical values for size and power
                    rho_lookup = np.round(np.abs(rho[i, m]), 2)
                    if rho_lookup > .99:
                        rho_lookup = .99
                    if K<=30:
                        ind = np.where((critical_values_size['K'] == K) & (critical_values_size['rho'] == rho_lookup))[0][0]
                    else:
                        ind = np.where((critical_values_size['K'] == 30) & (critical_values_size['rho'] == rho_lookup))[0][0]
                    F_cv_size[i, m] = np.array([
                        critical_values_size['r_125'][ind],
                        critical_values_size['r_10'][ind],
                        critical_values_size['r_075'][ind]
                    ], dtype=object)
                    F_cv_power[i, m] = np.array([
                        critical_values_power['r_50'][ind],
                        critical_values_power['r_75'][ind],
                        critical_values_power['r_95'][ind]
                    ], dtype=object)

                    # determine F-stat critical values for size
                    if F[i, m] < F_cv_size[i, m][0]:
                        symbols_size[i, m] = " "
                    elif F[i, m] < F_cv_size[i, m][1]:
                        symbols_size[i, m] = "*"
                    elif F[i, m] < F_cv_size[i, m][2]:
                        symbols_size[i, m] = "**"
                    else:
                        symbols_size[i, m] = "***"

                    # determine F-stat critical values for power
                    if F[i, m] < F_cv_power[i, m][0]:
                        symbols_power[i, m] = " "
                    elif F[i, m] < F_cv_power[i, m][1]:
                        symbols_power[i, m] = "^"
                    elif F[i, m] < F_cv_power[i, m][2]:
                        symbols_power[i, m] = "^^"
                    else:
                        symbols_power[i, m] = "^^^"

                if i >= m:
                    F[i, m] = "NaN"
                    symbols_size[i, m] = ""
                    symbols_power[i, m] = ""

            # compute the sigma model confidence set
            sigma_model_confidence_set = np.zeros([number_model_combinations, number_model_combinations])
            for index_i, model_i in enumerate(all_model_combinations):
                model_confidence_set_variance[index_i] = test_statistic_denominator[model_i[0], model_i[1]] / 2
                for index_j, model_j in enumerate(all_model_combinations):
                    term1 = covariance_mc[model_i[0], model_j[0]] - covariance_mc[model_i[1], model_j[0]]
                    term2 = covariance_mc[model_i[0], model_j[1]] - covariance_mc[model_i[1], model_j[1]]
                    sigma_model_confidence_set[index_j, index_i] = term1 - term2
            denominator = model_confidence_set_variance @ model_confidence_set_variance.T
            sigma_model_confidence_set = sigma_model_confidence_set / denominator

            # construct the model confidence set by iterating through all model pairs and comparing their test
            #    statistics
            converged = False
            model_confidence_set_pvalues = np.ones([M, 1])
            while not converged:

                # if we are on the last pair of models, use the model of worst fit to compute the p-value
                if np.shape(model_confidence_set)[0] == 2:
                    max_test_statistic = rv_test_statistic[model_confidence_set[0], model_confidence_set[1]]
                    if np.sign(max_test_statistic) >= 0:
                        worst_fit = model_confidence_set[0]
                        max_test_statistic = -max_test_statistic
                    else:
                        worst_fit = model_confidence_set[1]
                    model_confidence_set_pvalues[worst_fit] = 2 * norm.cdf(max_test_statistic)
                    converged = True
                else:
                    model_1 = []
                    model_2 = []
                    current_combinations = list(itertools.combinations(model_confidence_set, 2))
                    number_model_combinations = np.shape(current_combinations)[0]
                    sigma_index = np.empty(number_model_combinations, dtype=int)

                    # for each pair of models, find the RV test statistic and the max test statistic among the model
                    #   pairs
                    for model_pair in range(number_model_combinations):
                        model_1.append(current_combinations[model_pair][0])
                        model_2.append(current_combinations[model_pair][1])
                        sigma_index[model_pair] = all_model_combinations.index(current_combinations[model_pair])
                    test_statistic_model_confidence_set = rv_test_statistic[model_1, model_2]
                    index = np.argmax(abs(test_statistic_model_confidence_set))
                    max_test_statistic = test_statistic_model_confidence_set[index]

                    # find the model with the worst fit and remove it from the comparison set
                    if np.sign(max_test_statistic) >= 0:
                        worst_fit = model_1[index]
                    else:
                        worst_fit = model_2[index]
                        max_test_statistic = -max_test_statistic
                    mean = np.zeros([np.shape(current_combinations)[0]])
                    cov = sigma_model_confidence_set[sigma_index[:, None], sigma_index]
                    simulated_test_statistics = np.random.multivariate_normal(mean, cov, options.ndraws)
                    max_simulated_statistic = np.amax(abs(simulated_test_statistics), 1)
                    model_confidence_set_pvalues[worst_fit] = np.mean(max_simulated_statistic > max_test_statistic)
                    model_confidence_set = np.delete(model_confidence_set, np.where(model_confidence_set == worst_fit))

            # update the output list
            g_list[instrument] = g
            Q_list[instrument] = Q
            RV_numerator_list[instrument] = test_statistic_numerator
            RV_denominator_list[instrument] = test_statistic_denominator
            test_statistic_RV_list[instrument] = rv_test_statistic
            F_statistic_list[instrument] = F
            unscaled_F_statistic_list[instrument] = unscaled_F
            MCS_p_values_list[instrument] = model_confidence_set_pvalues
            rho_list[instrument] = rho
            F_cv_size_list[instrument] = F_cv_size
            F_cv_power_list[instrument] = F_cv_power
            symbols_size_list[instrument] = symbols_size
            symbols_power_list[instrument] = symbols_power

        # return results
        results = ProblemResults(Progress(
            self, markups, markups_downstream, markups_upstream, markups_orthogonal, marginal_cost,
            tau_list, g_list, Q_list, RV_numerator_list, RV_denominator_list,
            test_statistic_RV_list, F_statistic_list, MCS_p_values_list, rho_list, unscaled_F_statistic_list,
            F_cv_size_list, F_cv_power_list, symbols_size_list, symbols_power_list
        ))
        step_end_time = time.time()
        total_time = step_end_time - step_start_time
        print('Total Time is ... ' + str(total_time))
        output("")
        output(results)
        return results

    def _compute_first_difference_markups(self, markups_u, markups_l, epsilon, theta_index, gradient_markups):
        """Compute first differences and return the gradient."""
        for m in range(self.M):
            diff_markups = (markups_u[m] - markups_l[m]) / epsilon
            if self._absorb_cost_ids is not None:
                diff_markups, me = self._absorb_cost_ids(diff_markups)
                if self.products.w.any():
                    ols_result = sm.OLS(diff_markups, self.products.w).fit()
                    gradient_markups[m][:, theta_index] = ols_result.resid
                else:
                    gradient_markups[m][:, theta_index] = np.squeeze(diff_markups)
        return gradient_markups

    def _compute_perturbation(self, i, j, perturbation):
        """Perturb pi and recompute markups."""
        self.demand_results._pi[i, j] = perturbation
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            delta_new = self.demand_results.compute_delta()
        self.demand_results._delta = delta_new
        return build_markups(
            self.products, self.demand_results, self.models["models_downstream"], self.models["ownership_downstream"],
            self.models["models_upstream"], self.models["ownership_upstream"], self.models["vertical_integration"],
            self.models["custom_model_specification"], self.models["user_supplied_markups"]
        )

    def _compute_variance_covariance(self, m, i, N, clustering_adjustment, var):
        """Compute the variance covariance matrix."""
        variance_covariance = 1 / N * np.array([
            var[i].T @ var[i], var[m].T @ var[m], var[i].T @ var[m]
        ])
        if clustering_adjustment:
            cluster_ids = np.unique(self.products.clustering_ids)
            for j in cluster_ids:
                index = np.where(self.products.clustering_ids == j)[0]
                var1_l = var[i][index, :]
                var2_l = var[m][index, :]
                var1_c = var1_l
                var2_c = var2_l

                # update the matrix
                for k in range(len(index) - 1):
                    var1_c = np.roll(var1_c, 1, axis=0)
                    var2_c = np.roll(var2_c, 1, axis=0)
                    update = 1 / N * np.array([
                        var1_l.T @ var1_c, var2_l.T @ var2_c, var1_l.T @ var2_c
                    ])
                    variance_covariance = variance_covariance + update
        return variance_covariance


class Problem(ProblemEconomy):
    r"""A firm conduct testing-type problem.

    This class is initialized using the relevant data and formulations, and solved with :meth:`Problem.solve`.

    Parameters
    ----------
    cost_formulation: `Formulation`
        :class:`Formulation` is a list of the variables for observed product characteristics. All observed cost shifters
        included in this formulation must be variables in the `product_data`. To use a constant, one would replace `0`
        with `1`. To absorb fixed effects, specify `absorb = 'C(variable)'`, where the `variable` must also be in the
        `product_data`. Including this option implements fixed effects absorption using
        [PYHDFE](https://github.com/jeffgortmaker/pyhdfe), a companion package to PyBLP.

    instrument_formulation: `Formulation or sequence of Formulation`
        :class:`Formulation` is list of the variables used as excluded instruments for testing. For each instrument
        formulation, there should never be a constant. The user can specify as many instrument formulations as desired.
        All instruments must be variables in `product_data`.

        .. note::
            **Our instrument naming conventions differ from PyBLP**. With PyBLP, one specifies the excluded instruments
            for demand estimation via a naming convention in the product_data: each excluded instrument for demand
            estimation begins with `"demand_instrument"` followed by a number ( i.e., `demand_instrument0`). In
            pyRVtest, you specify directly the names of the variables in the `product_data` that you want to use as
            excluded instruments for testing (i.e., if you want to test with one instrument using the variable in the
            `product_data` named, "transportation_cost" one could specify
            `pyRVtest.Formulation('0 + transportation_cost')`.

    model_formulations: `sequence of ModelFormulation`
        :class:`ModelFormulation` defines the models that the researcher wants to test. There must be at least two
        instances of `ModelFormulation` specified to run the firm conduct testing procedure.

    product_data: `structured array-like`
        This is the data containing product and market observable characteristics, as well as instruments.

    pyblp_results`: `structured array-like`
        The results object returned by `pyblp.solve`.

   """

    def __init__(
            self, cost_formulation: Formulation, instrument_formulation: Sequence[Formulation],
            product_data: Mapping, demand_results: Mapping, model_formulations: Sequence[ModelFormulation] = None,
            markup_data: Optional[RecArray] = None) -> None:
        """Initialize the underlying economy with product and agent data before absorbing fixed effects."""

        # keep track of long it takes to initialize the problem
        output("Initializing the problem ...")
        start_time = time.time()

        # check if there is markup data to specify number of models
        if markup_data is None:
            M = len(model_formulations)
        else:
            M = np.shape(markup_data)[0]

        # check if there are instruments and if so count how many
        if hasattr(instrument_formulation, '__len__'):
            L = len(instrument_formulation)
        else:
            L = 1

        # validate and normalize cost formulation
        if not isinstance(cost_formulation, Formulation):
            raise TypeError("cost_formulation must be a single Formulation instance.")

        # validate instrument formulation
        if L == 1:
            if not isinstance(instrument_formulation, Formulation):
                raise TypeError("instrument_formulation must be a single Formulation instance.")
        elif L > 1:
            if not all(isinstance(f, Formulation) for f in instrument_formulation):
                raise TypeError("Each formulation in instrument_formulation must be a Formulation.")

        # initialize the underlying economy with structured product and cost data
        products = Products(
            cost_formulation=cost_formulation, instrument_formulation=instrument_formulation, product_data=product_data
        )
        if markup_data is None:
            models = Models(model_formulations=model_formulations, product_data=product_data)
            markups = [None] * M
        else:
            models = None
            markups = markup_data

        super().__init__(
            cost_formulation, instrument_formulation, model_formulations, products, models, demand_results, markups
        )

        # check cost shifters for collinearity
        if max(options.collinear_atol, options.collinear_rtol) > 0:
            cost_shifters = self.products.w
            common_message = "To disable collinearity checks, set options.collinear_atol = options.collinear_rtol = 0."
            collinear, successful = precisely_identify_collinearity(cost_shifters)
            if not successful:
                raise ValueError(
                    f"Failed to compute the QR decomposition of w while checking for collinearity issues. "
                    f"{common_message}"
                )
            if collinear.any():
                raise ValueError(
                    f"Detected collinearity issues with w. "
                    f"{common_message}"
                )
            for instrument in range(self.L):
                cost_shifters = self.products.w
                cost_shifters = np.append(cost_shifters, self.products["Z{0}".format(instrument)], axis=1)
                collinear, successful = precisely_identify_collinearity(cost_shifters)
                if not successful:
                    raise ValueError(
                        f"Failed to compute the QR decomposition of [w,z" + str(instrument) + "] while checking for "
                        f"collinearity issues. "
                        f"{common_message}"
                    )
                if collinear.any():
                    raise ValueError(
                        f"Detected collinearity issues with [w,z" + str(instrument) + "]."
                        f"{common_message}"
                    )

            # output information about the initialized problem
            output(f"Initialized the problem after {format_seconds(time.time() - start_time)}.")
            output("")
            output(self)


class Progress(object):
    """Structured information about estimation progress."""

    problem: ProblemEconomy
    markups: Array
    markups_downstream: Array
    markups_upstream: Array
    markups_orthogonal: Array
    marginal_cost: Array
    tau_list: Array
    mc: Array
    g: Array
    Q: Array
    RV_numerator: Array
    RV_denominator: Array
    test_statistic_RV: Array
    F: Array
    MCS_pvalues: Array
    rho: Array
    unscaled_F: Array
    F_cv_size_list: Array
    F_cv_power_list: Array
    symbols_size_list: Array
    symbols_power_list: Array

    def __init__(
            self, problem: ProblemEconomy, markups: Array, markups_downstream: Array, markups_upstream: Array,
            markups_orthogonal: Array, marginal_cost: Array, taus: Array,
            g: Array, Q: Array, RV_numerator: Array, RV_denom: Array, test_statistic_RV: Array,
            F: Array, MCS_pvalues: Array, rho: Array, unscaled_F: Array, F_cv_size_list: Array,
            F_cv_power_list: Array, symbols_size_list: Array, symbols_power_list: Array) -> None:
        """Store progress information, compute the projected gradient and its norm, and compute the reduced Hessian."""

        self.problem = problem
        self.markups = markups
        self.markups_downstream = markups_downstream
        self.markups_upstream = markups_upstream
        self.markups_orthogonal = markups_orthogonal
        self.marginal_cost = marginal_cost
        self.tau_list = taus
        self.g = g
        self.Q = Q
        self.RV_numerator = RV_numerator
        self.RV_denominator = RV_denom
        self.test_statistic_RV = test_statistic_RV
        self.F = F
        self.MCS_pvalues = MCS_pvalues
        self.rho = rho
        self.unscaled_F = unscaled_F
        self.F_cv_size_list = F_cv_size_list
        self.F_cv_power_list = F_cv_power_list
        self.symbols_size_list = symbols_size_list
        self.symbols_power_list = symbols_power_list
