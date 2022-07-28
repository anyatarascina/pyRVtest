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
            Configuration that allows user to specify whether or not to compute a two-step demand adjustment.
        se_type: `str'
            Configuration that specifies what kind of errors to compute.

        """

        # keep track of how long it takes to solve the problem
        output("Solving the problem ...")
        step_start_time = time.time()

        # validate settings
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
        marginal_cost = np.zeros(M)  # TODO: does this need to be initialized anymore?
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

        # if demand_adjustment is yes, get finite differences approx to the derivative of markups wrt theta
        # TODO: this was marked as something to be added, does this feature not exist yet?

        # absorb any cost fixed effects from prices, markups, and instruments
        # TODO: are the errors used anywhere? what should happen if this condition is false?
        if self._absorb_cost_ids is not None:
            output("Absorbing cost-side fixed effects ...")
            self.products.w, w_errors = self._absorb_cost_ids(self.products.w)
            prices_orthogonal, prices_errors = self._absorb_cost_ids(self.products.prices)
            for m in range(M):
                markups_orthogonal[m], markups_errors[m] = self._absorb_cost_ids(markups[m])
                marginal_cost_orthogonal[m], marginal_cost_errors[m] = self._absorb_cost_ids(marginal_cost[m])

        # residualize prices, markups, and instruments w.r.t cost shifters w and recover the tau parameters in cost
        #   regression on w
        # TODO: if above if condition is false, then prices_orthogonal not assigned!
        results = sm.OLS(prices_orthogonal, self.products.w).fit()
        prices_orthogonal = np.reshape(results.resid, [N, 1])
        for m in range(M):
            results = sm.OLS(markups_orthogonal[m], self.products.w).fit()
            markups_orthogonal[m] = np.reshape(results.resid, [N, 1])
            results = sm.OLS(marginal_cost_orthogonal[m], self.products.w).fit()
            tau_list[m] = results.params

        # if user specifies demand adjustment, account for two-step estimation in the standard errors by computing the
        #   finite difference approximation to the derivative of markups with respect to theta
        # TODO: price_column is not initialized? also is it overwritten in the loop (and why?)
        if demand_adjustment:
            ZD = self.demand_results.problem.products.ZD
            for i in range(len(self.demand_results.problem.products.dtype.fields['X1'][2])):
                if self.demand_results.problem.products.dtype.fields['X1'][2][i] == 'prices':
                    price_column = i

            # initialize variables for two-step standard error adjustment and other demand results
            XD = np.delete(self.demand_results.problem.products.X1, price_column, 1)
            WD = self.demand_results.updated_W
            h = self.demand_results.moments
            h_i = ZD * self.demand_results.xi
            K2 = self.demand_results.problem.K2  # size of demand side nonlinear characteristics
            D = self.demand_results.problem.D    # size of agent demographics

            # compute the gradient of the GMM moment function
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

            # TODO: for which of the following can I use the compute perturbations function?
            # compute sigma
            theta_index = 0
            delta_estimate = self.demand_results.delta
            for (i, j) in itertools.product(range(K2), range(K2)):
                if not self.demand_results.sigma[i, j] == 0:
                    sigma_initial = self.demand_results.sigma[i, j]

                    # reduce sigma by small increment, update delta, and recompute markups
                    self.demand_results.sigma[i, j] = sigma_initial - epsilon / 2
                    with contextlib.redirect_stdout(open(os.devnull, 'w')):
                        delta_new = self.demand_results.compute_delta()
                    self.demand_results.delta = delta_new
                    markups_l, md, ml = build_markups_all(
                        self.products, self.demand_results, self.models.models_downstream,
                        self.models.ownership_downstream, self.models.models_upstream,
                        self.models.ownership_upstream, self.models.VI, self.models.custom_model_specification
                    )

                    # increase sigma by small increment, update delta, and recompute markups
                    self.demand_results.sigma[i, j] = sigma_initial + epsilon / 2
                    with contextlib.redirect_stdout(open(os.devnull, 'w')):
                        delta_new = self.demand_results.compute_delta()
                    self.demand_results.delta = delta_new
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
                    theta_index = theta_index + 1

            # loop over nonlinear demand characteristics and demographics, and recompute markups with perturbations if
            #   the demand results for pi are not zero
            for (i, j) in itertools.product(range(K2), range(D)):
                if not self.demand_results.pi[i, j] == 0:
                    pi_initial = self.demand_results.pi[i, j]
                    perturbations = [pi_initial - epsilon / 2, pi_initial + epsilon / 2]
                    markups_l, md, ml = self._compute_perturbation(i, j, perturbations[0])
                    markups_u, mu, mu = self._compute_perturbation(i, j, perturbations[1])
                    gradient_markups = self._compute_first_difference_markups(
                        markups_u, markups_l, epsilon, theta_index, gradient_markups
                    )
                    self.demand_results.pi[i, j] = pi_initial
                    theta_index = theta_index + 1
            self.demand_results.delta = delta_estimate
                
            # if __, perturb alpha in negative (positive) direction and recompute markups
            for i in range(len(self.demand_results.beta)):
                if self.demand_results.beta_labels[i] == 'prices':
                    alpha_initial = self.demand_results.beta[i]
                    self.demand_results.beta[i] = alpha_initial - epsilon / 2
                    markups_l, md, ml = build_markups_all(
                        self.products, self.demand_results, self.models.models_downstream,
                        self.models.ownership_downstream, self.models.models_upstream, self.models.ownership_upstream,
                        self.models.VI, self.models.custom_model_specification
                    )
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

        # initialize empty lists to store statistic related values for each model _np.float64
        g_list = [None] * L   # TODO: possibly update to g_list = np.zeros((M, L), dtype=options.dtype)
        Q_list = [None] * L
        RV_numerator_list = [None] * L
        RV_denominator_list = [None] * L
        test_statistic_RV_list = [None] * L
        F_statistic_list = [None] * L
        MCS_p_values_list = [None] * L

        # for each instrument,
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
            psi = np.zeros([M, N, K])
            adjustment_value = [None] * M
            for m in range(M):
                psi_bar = W_12 @ g[m] - .5 * W_34 @ W_inverse @ W_34 @ g[m]
                WM34Zg = Z_orthogonal @ W_34 @ g[m]
                marginal_cost_orthogonal = (prices_orthogonal - markups_orthogonal[m])
                psi_i = (marginal_cost_orthogonal * Z_orthogonal) @ W_12 - 0.5 * WM34Zg * (Z_orthogonal @ W_34.T)
                psi[m] = psi_i - np.transpose(psi_bar)

                # make a demand adjustment
                if demand_adjustment:
                    G_k = -1 / N * np.transpose(Z_orthogonal) @ gradient_markups[m]
                    G_m[m] = G_k
                    adjustment_value[m] = W_12 @ G_m[m] @ inv(H_prime_wd @ H) @ H_prime_wd
                    psi[m] = psi[m] - (h_i - np.transpose(h)) @ np.transpose(adjustment_value[m])

            # initialize
            model_confidence_set = np.array(range(M))
            all_model_combinations = list(itertools.combinations(model_confidence_set, 2))
            number_model_combinations = np.shape(all_model_combinations)[0]
            model_confidence_set_variance = np.zeros([number_model_combinations, 1])

            # compute vii = 0 # TODO: add more descriptive comment
            for m in range(M):
                for i in range(m):
                    if i < m:
                        # TODO: check terminology - is this the variance covariance matrix?
                        variance_covariance = self._compute_variance_covariance(m, i, N, se_type, psi)
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

            # TODO: comment here
            sigma_model_confidence_set = np.zeros([number_model_combinations, number_model_combinations])
            for index_i, model_i in enumerate(all_model_combinations):
                model_confidence_set_variance[index_i] = test_statistic_denominator[model_i[0], model_i[1]] / 2
                for index_j, model_j in enumerate(all_model_combinations):
                    term1 = covariance_mc[model_i[0], model_j[0]] - covariance_mc[model_i[1], model_j[0]]
                    term2 = covariance_mc[model_i[0], model_j[1]] - covariance_mc[model_i[1], model_j[1]]
                    sigma_model_confidence_set[index_j, index_i] = term1 - term2
            denominator = model_confidence_set_variance @ model_confidence_set_variance.T
            sigma_model_confidence_set = sigma_model_confidence_set / denominator  # TODO: should be multiplied by 4?

            # compute the pairwise RV test statistic
            test_statistic_RV = np.zeros((M, M))
            for (m, i) in itertools.product(range(M), range(M)):
                if i < m:
                    test_statistic_RV[i, m] = test_statistic_numerator[i, m] / test_statistic_denominator[i, m]
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
                    phi[m] = phi[m] - (h_i - np.transpose(h)) @ np.transpose(W_12 @ adjustment_value[m])

            # TODO: what is phi and what is psi?
            for (m, i) in itertools.product(range(M), range(M)):
                if i < m:
                    variance = self._compute_variance_covariance(m, i, N, se_type, phi)
                    sigma = 1 / K * np.array([
                        np.trace(variance[0] @ W_inverse), np.trace(variance[1] @ W_inverse),
                        np.trace(variance[2] @ W_inverse)
                    ])
                    numerator = (sigma[0] - sigma[1]) * (sigma[0] - sigma[1])
                    denominator = ((sigma[0] + sigma[1]) * (sigma[0] + sigma[1]) - 4 * sigma[2] ** 2)
                    rho2 = numerator / denominator

                    # construct F statistic
                    operations = np.array([sigma[1], sigma[0], -2 * sigma[2]])
                    moments = np.array([
                        g[i].T @ weight_matrix @ g[i],
                        g[m].T @ weight_matrix @ g[m],
                        g[i].T @ weight_matrix @ g[m]
                    ]).flatten()
                    F_numerator = operations @ moments
                    F_denominator = (sigma[0] * sigma[1] - sigma[2] ** 2)
                    F[i, m] = (1 - rho2) * N / (2 * K) * F_numerator / F_denominator
                if i >= m:
                    F[i, m] = "NaN"

            # set a random seed
            np.random.seed(123)  # TODO: is this just for testing?

            # construct the model confidence set by iterating through all model pairs and comparing their test
            #    statistics
            # TODO: variables here need to be renamed
            converged = False
            model_confidence_set_pvalues = np.ones([M, 1])
            while not converged:
                # if we are on the last pair of models, use the model of worst fit to compute the p-value
                if np.shape(model_confidence_set)[0] == 2:
                    TRV_max = test_statistic_RV[model_confidence_set[0], model_confidence_set[1]]
                    if np.sign(TRV_max) >= 0:
                        em = model_confidence_set[0]  # TODO: what is em?
                        TRV_max = -TRV_max
                    else:
                        em = model_confidence_set[1]
                    model_confidence_set_pvalues[em] = 2 * norm.cdf(TRV_max)
                    converged = True
                else:
                    combos = list(itertools.combinations(model_confidence_set, 2))
                    model_1 = []  # TODO: get rid of these temp variables?
                    model_2 = []
                    number_model_combinations = np.shape(combos)[0]
                    sigma_index = np.empty(number_model_combinations, dtype=int)
                    for model_pair in range(number_model_combinations):
                        model_1.append(combos[model_pair][0])
                        model_2.append(combos[model_pair][1])
                        sigma_index[model_pair] = all_model_combinations.index(combos[model_pair])
                    TRV_MCS = test_statistic_RV[model_1, model_2]
                    index = np.argmax(abs(TRV_MCS))
                    TRV_max = TRV_MCS[index]

                    if np.sign(TRV_max) >= 0:
                        em = model_1[index]
                    else:
                        em = model_2[index]
                        TRV_max = -TRV_max
                    mean = np.zeros([np.shape(combos)[0]])
                    cov = sigma_model_confidence_set[sigma_index[:, None], sigma_index]
                    simTRV = np.random.multivariate_normal(mean, cov, options.ndraws)
                    maxsimTRV = np.amax(abs(simTRV), 1)
                    model_confidence_set_pvalues[em] = np.mean(maxsimTRV > TRV_max)
                    model_confidence_set = np.delete(model_confidence_set, np.where(model_confidence_set == em))

            g_list[instrument] = g
            Q_list[instrument] = Q
            RV_numerator_list[instrument] = test_statistic_numerator
            RV_denominator_list[instrument] = test_statistic_denominator
            test_statistic_RV_list[instrument] = test_statistic_RV
            F_statistic_list[instrument] = F
            MCS_p_values_list[instrument] = model_confidence_set_pvalues

        # return results
        results = ProblemResults(Progress(
            self, markups, markups_downstream, markups_upstream, marginal_cost, tau_list, g_list, Q_list,
            RV_numerator_list, RV_denominator_list, test_statistic_RV_list, F_statistic_list, MCS_p_values_list
        ))
        # TODO: should time outputs be in a function?
        step_end_time = time.time()
        total_time = step_end_time-step_start_time
        print('Total Time is ... ' + str(total_time))
        output("")
        output(results)
        return results

    def _compute_first_difference_markups(self, markups_u, markups_l, epsilon, theta_index, gradient_markups):
        """Compute first differences and return the gradient."""
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

    def _compute_variance_covariance(self, m, i, N, se_type, var):
        """Compute the variance covariance matrix."""
        variance_covariance = 1 / N * np.array([
            var[i].T @ var[i], var[m].T @ var[m], var[i].T @ var[m]
        ])
        if se_type == 'clustered':
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
