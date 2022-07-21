"""Economy-level BLP problem functionality."""

import abc
import contextlib
import os
import time
import statsmodels.api as sm 
from typing import Mapping, Optional, Sequence
import math
import numpy as np
import itertools
from scipy.linalg import inv, fractional_matrix_power
from scipy.stats import norm
from .economy import Economy
from .. import options
from ..configurations.formulation import Formulation, ModelFormulation
from ..primitives import Models, Products
from ..results.problem_results import ProblemResults
from ..utilities.algebra import precisely_identify_collinearity
from ..utilities.basics import Array, RecArray, format_seconds, output
from ..construction import build_markups_all


class ProblemEconomy(Economy):
    """An abstract BLP problem."""

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
            self, method: str = 'both', demand_adjustment: bool = False, se_type: str = 'unadjusted') -> ProblemResults:
        r"""Solve the problem.

        # TODO: add general overview

        Parameters
        ----------
        method: `str`
            Configuration that allows user to select method? # TODO: what is unused argument both?
        demand_adjustment: `bool'
            Configuration that specifies whether or not to compute a two-step demand adjustment?
        se_type: `str'
            Configuration that specifies what kind of errors to compute... Is this optional?

        """

        # keep track of how long it takes to solve the problem
        output("Solving the problem ...")
        step_start_time = time.time()

        # validate settings
        # TODO: change demand adjustment to bool?
        if not isinstance(demand_adjustment, bool):
            raise TypeError("demand_adjustment must be a boolean.")
        if se_type not in {'robust', 'unadjusted', 'clustered'}:
            raise ValueError("se_type must be 'robust', 'unadjusted', or 'clustered'.")
        if se_type == 'clustered' and np.shape(self.products.clustering_ids)[1] != 1:
            raise ValueError("product_data.clustering_ids must be specified with se_type 'clustered'.")
 
        # initialize constants and precomputed values
        M = self.M
        N = self.N
        L = self.L
        Dict_K = self.Dict_K  # TODO: why is this assigned but not used? what is it?
        markups = self.markups

        # initialize variables to be computed
        # TODO: convert everything to arrays, get rid of looping over models where possible
        markups_upstream = np.zeros(M)
        markups_downstream = np.zeros(M)
        marginal_cost = np.zeros(M)  # seems like now this doesn't need to be initialized anymore
        markups_orthogonal = [None] * M
        marginal_cost_orthogonal = [None] * M
        tau_list = [None] * M
        markups_errors = [None] * M
        marginal_cost_errors = [None] * M

        # if there are no markups, compute them
        if markups[0] is None:
            print('Computing Markups ... ')
            markups, markups_downstream, markups_upstream = build_markups_all(
                self.products, self.demand_results, self.models.models_downstream, self.models.ownership_downstream,
                self.models.models_upstream, self.models.ownership_upstream, self.models.VI,
                self.models.custom_model_specification
            )

        # for each model, use computed markups to compute the marginal costs
        marginal_cost = self.products.prices - markups

        # if demand_adjustment is True, get finite differences approx to the derivative of markups wrt theta
        # TODO: add this feature? or is this comment for the block below

        # absorb any cost fixed effects from prices, markups, and instruments
        # TODO: are the errors used anywhere?
        if self._absorb_cost_ids is not None:
            output("Absorbing cost-side fixed effects ...")
            self.products.w, w_errors = self._absorb_cost_ids(self.products.w)
            prices_orthogonal, prices_errors = self._absorb_cost_ids(self.products.prices)
            for m in range(M):
                markups_orthogonal[m], markups_errors[m] = self._absorb_cost_ids(markups[m])
                marginal_cost_orthogonal[m], marginal_cost_errors[m] = self._absorb_cost_ids(marginal_cost[m])

        # residualize prices, markups, and instruments w.r.t cost shifters w and recover the tau parameters in cost
        #   regression on w
        # TODO: if above if condition is false, then prices_orthogonal not assigned...
        results = sm.OLS(prices_orthogonal, self.products.w).fit()
        prices_orthogonal = np.reshape(results.resid, [N, 1])
        for m in range(M):
            results = sm.OLS(markups_orthogonal[m], self.products.w).fit()
            markups_orthogonal[m] = np.reshape(results.resid, [N, 1])
            results = sm.OLS(marginal_cost_orthogonal[m], self.products.w).fit()
            tau_list[m] = results.params

        # if user specifies demand adjustment,
        if demand_adjustment:

            ZD = self.demand_results.problem.products.ZD
            for i in range(len(self.demand_results.problem.products.dtype.fields['X1'][2])):
                if self.demand_results.problem.products.dtype.fields['X1'][2][i] == 'prices':
                    price_col = i

            # initialize variables for two-step standard error adjustment and other demand results
            XD = np.delete(self.demand_results.problem.products.X1, price_col, 1)
            WD = self.demand_results.updated_W
            h = self.demand_results.moments
            h_i = ZD * self.demand_results.xi
            K2 = self.demand_results.problem.K2  # size of demand side nonlinear characteristics
            D = self.demand_results.problem.D    # size of agent demographics

            # comment here
            # TODO: warning since shouldn't call this method outside of class
            partial_y_theta = np.append(
                self.demand_results.xi_by_theta_jacobian, -self.demand_results.problem.products.prices, 1
            )
            partial_y_theta = self.demand_results.problem._absorb_demand_ids(partial_y_theta)
            partial_y_theta = np.reshape(partial_y_theta[0], [N, len(self.demand_results.theta) + 1])
            if np.shape(XD)[1] == 0:
                partial_xi_theta = partial_y_theta
            else:
                product = XD @ inv(XD.T @ ZD @ WD @ ZD.T @ XD) @ (XD.T @ ZD @ WD @ ZD.T @ partial_y_theta)
                partial_xi_theta = partial_y_theta - product

            # compute the gradient of the GMM moment function
            H = 1 / N * (np.transpose(ZD) @ partial_xi_theta)
            H_prime = np.transpose(H)
            H_prime_wd = H_prime @ WD

            # build adjustment to psi for each model
            epsilon = options.finite_differences_epsilon
            G_m = [None] * M
            gradient_markups = [None] * M

            # get numerical derivatives of markups wrt theta, alpha
            for m in range(M):
                gradient_markups[m] = np.zeros((N, len(self.demand_results.theta) + 1))

            # compute sigma
            theta_index = 0
            delta_estimate = self.demand_results.delta
            for i in range(K2):
                for j in range(K2):
                    if not self.demand_results.sigma[i, j] == 0:
                        sigma_initial = self.demand_results.sigma[i, j]

                        # reduce sigma by small increment
                        self.demand_results.sigma[i, j] = sigma_initial - epsilon / 2

                        # update delta
                        with contextlib.redirect_stdout(open(os.devnull, 'w')):
                            delta_new = self.demand_results.compute_delta()
                        self.demand_results.delta = delta_new

                        # recompute markups
                        markups_l, md, ml = build_markups_all(
                            self.products, self.demand_results, self.models.models_downstream,
                            self.models.ownership_downstream, self.models.models_upstream,
                            self.models.ownership_upstream, self.models.VI, self.models.custom_model_specification
                        )

                        # increase sigma by small increment
                        self.demand_results.sigma[i, j] = sigma_initial + epsilon / 2

                        # update delta
                        with contextlib.redirect_stdout(open(os.devnull, 'w')):
                            delta_new = self.demand_results.compute_delta()
                        self.demand_results.delta = delta_new

                        # recompute markups
                        markups_u, mu, mu = build_markups_all(
                            self.products, self.demand_results, self.models.models_downstream,
                            self.models.ownership_downstream, self.models.models_upstream,
                            self.models.ownership_upstream, self.models.VI, self.models.custom_model_specification
                        )

                        # compute first difference approximation of derivative of markups
                        gradient_markups = self._compute_first_difference_markups(
                            markups_u, markups_l, epsilon, theta_index, gradient_markups
                        )

                        self.demand_results.sigma[i, j] = sigma_initial
                        theta_index = theta_index+1

            # loop over nonlinear demand characteristics and demographics, and compute pi
            for i in range(K2):
                for j in range(D):

                    # if results for pi are not zero
                    if not self.demand_results.pi[i, j] == 0:

                        # recompute markups with perturbatins
                        pi_initial = self.demand_results.pi[i, j]
                        perturbations = [pi_initial - epsilon / 2, pi_initial + epsilon / 2]
                        markups_l, md, ml = self._compute_perturbation(i, j, perturbations[0])
                        markups_u, mu, mu = self._compute_perturbation(i, j, perturbations[1])

                        # compute first differences for the markups
                        gradient_markups = self._compute_first_difference_markups(
                            markups_u, markups_l, epsilon, theta_index, gradient_markups
                        )
                        self.demand_results.pi[i, j] = pi_initial
                        theta_index = theta_index + 1
            self.demand_results.delta = delta_estimate
                
            # compute alpha
            for i in range(len(self.demand_results.beta)):
                if self.demand_results.beta_labels[i] == 'prices':
                    alpha_initial = self.demand_results.beta[i]

                    # perturb alpha in the negative direction and recompute markups
                    self.demand_results.beta[i] = alpha_initial - epsilon / 2
                    markups_l, md, ml = build_markups_all(
                        self.products, self.demand_results, self.models.models_downstream,
                        self.models.ownership_downstream, self.models.models_upstream, self.models.ownership_upstream,
                        self.models.VI, self.models.custom_model_specification
                    )

                    # perturb alpha in the positive direction and recompute markups
                    self.demand_results.beta[i] = alpha_initial + epsilon / 2
                    markups_u, mu, mu = build_markups_all(
                        self.products, self.demand_results, self.models.models_downstream,
                        self.models.ownership_downstream, self.models.models_upstream, self.models.ownership_upstream,
                        self.models.VI, self.models.custom_model_specification
                    )

                    gradient_markups = self._compute_first_difference_markups(
                        markups_u, markups_l, epsilon, theta_index, gradient_markups
                    )
                    self.demand_results.beta[i] = alpha_initial
                    theta_index = theta_index+1

        # initialize empty lists to store statistic related values for each model
        g_list = [None] * L
        Q_list = [None] * L
        RV_numerator_list = [None] * L
        RV_denominator_list = [None] * L
        test_statistic_RV_list = [None] * L
        F_statistic_list = [None] * L
        MCS_p_values_list = [None] * L

        # for each instrument
        for instrument in range(L):
            instruments = self.products["Z{0}".format(instrument)]
            K = np.shape(instruments)[1]

            # absorb any cost fixed effects from prices, markups, and instruments
            if self._absorb_cost_ids is not None:
                # TODO: can absorb_cost_ids be None? bc in that case, it seems like the initial Z_orthogonal is never
                #  assigned
                Z_orthogonal, Z_errors = self._absorb_cost_ids(instruments)
            Z_residual = sm.OLS(Z_orthogonal, self.products.w).fit().resid
            Z_orthogonal = np.reshape(Z_residual, [N, K])

            # initialize variables to store GMM measure of fit Q_m for each model
            # TODO: can these be converted to arrays?
            g = [None] * M
            Q = [None] * M

            # compute the weight matrix
            W_inverse = 1 / N * (Z_orthogonal.T @ Z_orthogonal)
            W_inverse = np.reshape(W_inverse, [K, K])
            weight_matrix = inv(W_inverse)  # TODO: commented out Jeff's precisely invert before - why not use it?

            # for each model compute GMM measure of fit
            for m in range(M):
                g[m] = 1 / N * (Z_orthogonal.T @ (prices_orthogonal - markups_orthogonal[m]))
                g[m] = np.reshape(g[m], [K, 1])
                Q[m] = g[m].T @ weight_matrix @ g[m]

            # compute the pairwise RV numerator
            RV_numerator = np.zeros((M, M))
            for m in range(M):
                for i in range(m):
                    if i < m:
                        RV_numerator[i, m] = math.sqrt(N) * (Q[i] - Q[m])

            # compute the pairwise RV denominator
            RV_denominator = np.zeros((M, M))
            covariance_mc = np.zeros((M, M))

            # take powers of the weighting matrix
            W_12 = fractional_matrix_power(weight_matrix, 0.5)
            W_34 = fractional_matrix_power(weight_matrix, 0.75)

            # compute psi to be used in the variance covariance estimator
            psi = np.zeros([M, N, K])
            dmd_adj = [None] * M
            for m in range(M):
                psi_bar = W_12 @ g[m] - .5 * W_34 @ W_inverse @ W_34 @ g[m]
                WM34Z = Z_orthogonal @ W_34
                WM34Zg = WM34Z @ g[m]
                psi_i = ((prices_orthogonal - markups_orthogonal[m]) * Z_orthogonal) @ W_12 - 0.5 * WM34Zg * (Z_orthogonal @ W_34.T)
                psi[m] = psi_i - np.transpose(psi_bar)

                # make a demand adjustment
                if demand_adjustment:
                    G_k = -1 / N * np.transpose(Z_orthogonal) @ gradient_markups[m]
                    G_m[m] = G_k
                    dmd_adj[m] = W_12 @ G_m[m] @ inv(H_prime_wd @ H) @ H_prime_wd
                    psi[m] = psi[m] - (h_i - np.transpose(h)) @ np.transpose(dmd_adj[m])

            # initialize 
            M_MCS = np.array(range(M))
            all_combinations = list(itertools.combinations(M_MCS, 2))
            number_combinations = np.shape(all_combinations)[0]
            Var_MCS = np.zeros([number_combinations, 1])

            # compute vii = 0 - whatever that means?
            # TODO: is this actually the variance covariance matrix?
            for m in range(M):
                for i in range(m):
                    if i < m:
                        # fill the initial variance covariance matrix
                        variance_covariance = 1 / N * np.array([
                            psi[i].T @ psi[i], psi[m].T @ psi[m], psi[i].T @ psi[m]
                        ])
                        if se_type == 'clustered':
                            cluster_ids = np.unique(self.products.clustering_ids)
                            for j in cluster_ids:
                                ind_kk = np.where(self.products.clustering_ids == j)[0]
                                psi1_l = psi[i][ind_kk, :]
                                psi2_l = psi[m][ind_kk, :]

                                # TODO: not sure what the point of these two is? initializing?
                                psi1_c = psi1_l
                                psi2_c = psi2_l

                                # update the variance covariance matrix
                                for ii in range(len(ind_kk)-1):
                                    psi1_c = np.roll(psi1_c, 1, axis=0)
                                    psi2_c = np.roll(psi2_c, 1, axis=0)
                                    update = 1 / N * np.array([
                                        psi1_l.T @ psi1_c, psi2_l.T @ psi2_c, psi1_l.T @ psi2_c
                                    ])
                                    variance_covariance = variance_covariance + update

                        # compute the sigma
                        weighted_variance = W_12 @ variance_covariance @ W_12
                        sigma2 = 4 * (g[i].T @ weighted_variance[0] @ g[i] + g[m].T @ weighted_variance[1] @ g[m] - 2 * g[i].T @ weighted_variance[2] @ g[m])

                        # compute the covariance matrix for marginal costs
                        covariance_mc[i, m] = g[i].T @ weighted_variance[2] @ g[m]
                        covariance_mc[m, i] = covariance_mc[i, m]
                        covariance_mc[m, m] = g[m].T @ weighted_variance[1] @ g[m]
                        covariance_mc[i, i] = g[i].T @ weighted_variance[0] @ g[i]

                        RV_denominator[i, m] = math.sqrt(sigma2)

            # TODO: what is this block?
            Sigma_MCS = np.zeros([number_combinations, number_combinations])

            # for ii in range(number_combinations):
            #     tmp3 = all_combinations[ii][0]
            #     tmp4 = all_combinations[ii][1]
            #     Var_MCS[ii] = RV_denominator[tmp3, tmp4]/2
            #     for i in range(number_combinations):
            #         tmp1 = all_combinations[i][0]
            #         tmp2 = all_combinations[i][1]
            #         Sigma_MCS[i, ii] = covariance_mc[tmp1, tmp3] - covariance_mc[tmp2, tmp3] - covariance_mc[tmp1, tmp4] + covariance_mc[tmp2, tmp4]
            # Sigma_MCS = Sigma_MCS / (Var_MCS@Var_MCS.T)

            # TODO: check that this works
            for index1, value1 in enumerate(all_combinations):
                Var_MCS[index1] = RV_denominator[value1[0], value1[1]] / 2
                for index2, value2 in enumerate(all_combinations):
                    term1 = covariance_mc[value1[0], value2[0]] - covariance_mc[value1[1], value2[0]]
                    term2 = covariance_mc[value1[0], value2[1]] - covariance_mc[value1[1], value2[1]]
                    Sigma_MCS[index2, index1] = term1 - term2
            Sigma_MCS = Sigma_MCS / (Var_MCS @ Var_MCS.T)

            # compute the pairwise RV test statistic
            test_statistic_RV = np.zeros((M, M))
            for (m, i) in itertools.product(range(M), range(M)):
                if i < m:
                    test_statistic_RV[i, m] = RV_numerator[i, m] / RV_denominator[i, m]
                else:
                    test_statistic_RV[i, m] = "NaN"

            # compute the pairwise F-statistic for each model
            F = np.zeros((M, M))
            pi = np.zeros((K, M))
            phi = np.zeros([M, N, K])
            for m in range(M):
                ols_results = sm.OLS(prices_orthogonal - markups_orthogonal[m], Z_orthogonal).fit()
                pi[:, m] = ols_results.params
                e = np.reshape(ols_results.resid, [N, 1])
                phi[m] = (e * Z_orthogonal) @ weight_matrix
                if demand_adjustment:
                    phi[m] = phi[m] - (h_i - np.transpose(h)) @ np.transpose(W_12 @ dmd_adj[m])

            # TODO: parts of this block are repeated but with phi instead of psi - can I throw this into a function?
            # TODO: what is phi and what is psi?
            for (m, i) in itertools.product(range(M), range(M)):
                if i < m:
                    VAR_11 = 1 / N * phi[i].T @ phi[i]
                    VAR_22 = 1 / N * phi[m].T @ phi[m]
                    VAR_12 = 1 / N * phi[i].T @ phi[m]
                    if se_type == 'clustered':
                        cluster_ids = np.unique(self.products.clustering_ids)
                        for j in cluster_ids:
                            ind_kk = np.where(self.products.clustering_ids == j)[0]

                            phi1_l = phi[i][ind_kk, :]
                            phi2_l = phi[m][ind_kk, :]
                            phi1_c = phi1_l
                            phi2_c = phi2_l

                            for ii in range(len(ind_kk) - 1):
                                phi1_c = np.roll(phi1_c, 1, axis=0)
                                phi2_c = np.roll(phi2_c, 1, axis=0)

                                VAR_11 = VAR_11 + 1 / N * phi1_l.T @ phi1_c
                                VAR_22 = VAR_22 + 1 / N * phi2_l.T @ phi2_c
                                VAR_12 = VAR_12 + 1 / N * phi1_l.T @ phi2_c

                    # TODO: everything below this is also in the loop
                    sigma = 1 / K * np.array([
                        np.trace(VAR_11 @ W_inverse), np.trace(VAR_22 @ W_inverse), np.trace(VAR_12 @ W_inverse)
                    ])
                    numerator = (sigma[0] - sigma[1]) * (sigma[0] - sigma[1])
                    denominator = ((sigma[0] + sigma[1]) * (sigma[0] + sigma[1]) - 4 * sigma[2]**2)
                    rho2 = numerator / denominator

                    # TODO: make this numerator into a vector multiplication
                    F_numerator = (sigma[1] * g[i].T @ weight_matrix @ g[i] + sigma[0] * g[m].T @ weight_matrix @ g[m] - 2 * sigma[2] * g[i].T @ weight_matrix @ g[m])
                    F_denominator = (sigma[0] * sigma[1] - sigma[2]**2)
                    F[i, m] = (1 - rho2) * N / (2 * K) * F_numerator / F_denominator
                if i >= m:
                    F[i, m] = "NaN"

            MCS_pvalues = np.ones([M, 1])

            # set a random seed
            # TODO: why?
            np.random.seed(123)
                
            converged = 0
            while converged == 0:
                if np.shape(M_MCS)[0] == 2:
                    TRV_max = test_statistic_RV[M_MCS[0], M_MCS[1]]
                    if np.sign(TRV_max) >= 0:
                        em = M_MCS[0]
                        TRV_max = -TRV_max
                    else:
                        em = M_MCS[1]
                    MCS_pvalues[em] = 2 * norm.cdf(TRV_max)

                    converged = 1
                else:
                    combos = list(itertools.combinations(M_MCS, 2))
                    tmp1 = []
                    tmp2 = []
                    
                    number_combinations = np.shape(combos)[0]
                    Sig_idx = np.empty(number_combinations, dtype=int)
                    for ii in range(number_combinations):
                        tmp1.append(combos[ii][0])
                        tmp2.append(combos[ii][1])

                        Sig_idx[ii] = all_combinations.index(combos[ii])

                    TRV_MCS = test_statistic_RV[tmp1, tmp2]
                    index = np.argmax(abs(TRV_MCS))
                    TRV_max = TRV_MCS[index]

                    if np.sign(TRV_max) >= 0:
                        em = tmp1[index]
                    else:
                        em = tmp2[index]
                        TRV_max = -TRV_max
                       
                    mean = np.zeros([np.shape(combos)[0]]) 
                    cov = Sigma_MCS[Sig_idx[:, None], Sig_idx]
                    n_draws = 99999  # TODO: why is this set here?
                    simTRV = np.random.multivariate_normal(mean, cov, n_draws)
                    maxsimTRV = np.amax(abs(simTRV), 1)
                    MCS_pvalues[em] = np.mean(maxsimTRV > TRV_max)
                    M_MCS = np.delete(M_MCS, np.where(M_MCS == em))

            g_list[instrument] = g
            Q_list[instrument] = Q
            RV_numerator_list[instrument] = RV_numerator
            RV_denominator_list[instrument] = RV_denominator
            test_statistic_RV_list[instrument] = test_statistic_RV
            F_statistic_list[instrument] = F
            MCS_p_values_list[instrument] = MCS_pvalues

        # return results
        results = ProblemResults(Progress(
            self, markups, markups_downstream, markups_upstream, marginal_cost, tau_list, g_list, Q_list,
            RV_numerator_list, RV_denominator_list, test_statistic_RV_list, F_statistic_list, MCS_p_values_list
        ))
        step_end_time = time.time()
        total_time = step_end_time-step_start_time
        print('Total Time is ... ' + str(total_time))
        output("")
        output(results)
        return results

    def _compute_first_difference_markups(self, markups_u, markups_l, epsilon, theta_index, gradient_markups):
        for m in range(self.M):
            diff_markups = (markups_u[m] - markups_l[m]) / epsilon
            diff_markups, me = self._absorb_cost_ids(diff_markups)
            ols_result = sm.OLS(diff_markups, self.products.w).fit()
            gradient_markups[m][:, theta_index] = ols_result.resid
        return gradient_markups

    def _compute_perturbation(self, i, j, perturbation):
        """Perturb pi and recompute markups."""
        self.demand_results.pi[i, j] = perturbation
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            delta_new = self.demand_results.compute_delta()
        self.demand_results.delta = delta_new
        return build_markups_all(
            self.products, self.demand_results, self.models.models_downstream,
            self.models.ownership_downstream, self.models.models_upstream,
            self.models.ownership_upstream, self.models.VI, self.models.custom_model_specification
        )

    def _not_sure(self, M, N, se_type, var):

        # do the loop!
        for m in range(M):
            for i in range(m):
                if i < m:
                    # fill the initial variance covariance matrix
                    variance_covariance = 1 / N * np.array([
                        var[i].T @ var[i], var[m].T @ var[m], var[i].T @ var[m]
                    ])
                    if se_type == 'clustered':
                        cluster_ids = np.unique(self.products.clustering_ids)
                        for j in cluster_ids:
                            ind_kk = np.where(self.products.clustering_ids == j)[0]
                            var1_l = var[i][ind_kk, :]
                            var2_l = var[m][ind_kk, :]

                            # TODO: not sure what the point of these two is? initializing?
                            var1_c = var1_l
                            var2_c = var2_l

                            # update the variance covariance matrix
                            for ii in range(len(ind_kk) - 1):
                                var1_c = np.roll(var1_c, 1, axis=0)
                                var2_c = np.roll(var2_c, 1, axis=0)
                                update = 1 / N * np.array([
                                    var1_l.T @ var1_c, var2_l.T @ var2_c, var1_l.T @ var2_c
                                ])
                                variance_covariance = variance_covariance + update


class Problem(ProblemEconomy):
    r"""A BLP-type problem.

    
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
            markups = [None]*M
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
                        f"Failed to compute the QR decomposition of [w,z"+str(instrument)+"] while checking for "
                        f"collinearity issues. "
                        f"{common_message}"
                    )
                if collinear.any():
                    raise ValueError(
                        f"Detected collinearity issues with [w,z"+str(instrument)+"]. "
                        f"{common_message}"
                    )    

            # output information about the initialized problem
            output(f"Initialized the problem after {format_seconds(time.time() - start_time)}.")
            output("")
            output(self)


class InitialProgress(object):
    """Structured information about initial estimation progress."""

    problem: ProblemEconomy

    def __init__(
            self, problem: ProblemEconomy 
            ) -> None:
        """Store initial progress information, computing the projected gradient and the reduced Hessian."""
        self.problem = problem
        

class Progress(InitialProgress):
    """Structured information about estimation progress."""

    markups: Array
    markups_downstream: Array
    markups_upstream: Array
    tau_list: Array
    mc: Array
    g: Array
    Q: Array
    RV_numerator: Array
    RV_denominator: Array
    test_statistic_RV: Array
    F: Array
    MCS_p_values: Array

    def __init__(
            self, problem: ProblemEconomy, markups: Array, markups_downstream: Array, markups_upstream: Array,
            mc: Array, taus: Array, g: Array, Q: Array, RV_numerator: Array, RV_denom: Array, test_statistic_RV: Array,
            F: Array, MCS_pvalues: Array
            ) -> None:
        """Store progress information, compute the projected gradient and its norm, and compute the reduced Hessian."""
        super().__init__(
            problem
        )
        self.markups = markups
        self.markups_downstream = markups_downstream
        self.markups_upstream = markups_upstream
        self.tau_list = taus
        self.mc = mc
        self.g = g
        self.Q = Q
        self.RV_numerator = RV_numerator
        self.RV_denominator = RV_denom
        self.test_statistic_RV = test_statistic_RV
        self.F = F
        self.MCS_p_values = MCS_pvalues
