"""Economy-level BLP problem functionality."""

import abc
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
from ..utilities.basics import Array, RecArray, format_seconds, output, HideOutput
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
            self, method: str = 'both', demand_adjustment: str = 'no', se_type: str = 'unadjusted') -> ProblemResults:
        r"""Solve the problem.

        # TODO: add general overview

        Parameters
        ----------
        method: `str`
            Configuration that allows user to select method? # TODO: what is unused argument both?
        demand_adjustment: `str'
            Configuration that specifies whether or not to compute a two-step demand adjustment?
        se_type: `str'
            Configuration that specifies what kind of errors to compute... Is this optional?

        """

        # keep track of how long it takes to solve the problem
        output("Solving the problem ...")
        step_start_time = time.time()

        # validate settings
        if demand_adjustment not in {'yes', 'no'}:
            raise TypeError("demand_adjustment must be 'yes' or 'no'.")
        if se_type not in {'robust', 'unadjusted', 'clustered'}:
            raise ValueError("se_type must be 'robust', 'unadjusted', or 'clustered'.")
        if se_type == 'clustered' and np.shape(self.products.clustering_ids)[1] !=1:
            raise ValueError("product_data.clustering_ids must be specified with se_type 'clustered'.")
 
        # initialize constants and precomputed values
        M = self.M
        N = self.N
        L = self.L
        Dict_K = self.Dict_K  # TODO: why is this assigned but not used?
        markups = self.markups

        # initialize variables to be computed as all zeroes
        markups_upstream = [None] * M
        markups_downstream = [None] * M
        marginal_cost = [None] * M
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
        for m in range(M):
            marginal_cost[m] = self.products.prices - markups[m]

        # if demand_adjustment is yes, get finite differences approx to the derivative of markups wrt theta
        # TODO: add this feature?

        # absorb any cost fixed effects from prices, markups, and instruments
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
        if demand_adjustment == 'yes':
            ZD = self.demand_results.problem.products.ZD
            for i in range(len(self.demand_results.problem.products.dtype.fields['X1'][2])):
                if self.demand_results.problem.products.dtype.fields['X1'][2][i] == 'prices':
                    price_col = i

            # initialize variables for two-step standard error adjustment
            XD = np.delete(self.demand_results.problem.products.X1, price_col, 1)
            WD = self.demand_results.updated_W
            h = self.demand_results.moments

            # comment here
            # TODO: rename these variables?
            dy_dtheta = np.append(
                self.demand_results.xi_by_theta_jacobian, -self.demand_results.problem.products.prices, 1
            )
            dy_dtheta = self.demand_results.problem._absorb_demand_ids(dy_dtheta)
            dy_dtheta = np.reshape(dy_dtheta[0], [N, len(self.demand_results.theta) + 1])
            if np.shape(XD)[1] == 0:
                dxi_dtheta = dy_dtheta
            else:    
                dxi_dtheta = dy_dtheta - XD @ inv(XD.T @ ZD @ WD @ ZD.T @ XD) @ (XD.T @ ZD @ WD @ ZD.T @ dy_dtheta)

            # compute the gradient of the GMM moment function
            H = 1 / N * (np.transpose(ZD) @ dxi_dtheta)
            H_prime = np.transpose(H)
            H_primeWD = H_prime @ WD

            # something else
            h_i = ZD * self.demand_results.xi
            K2 = self.demand_results.problem.K2
            D = self.demand_results.problem.D

            # build adjustment to psi for each model
            epsilon = options.finite_differences_epsilon
            G_m = [None]*M
            gradient_markups = [None]*M

            # get numerical derivatives of markups wrt theta, alpha
            for m in range(M):
                gradient_markups[m] = np.zeros((N, len(self.demand_results.theta)+1))

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
                        with HideOutput():
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
                        with HideOutput():
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

            # compute pi
            for i in range(K2):
                for j in range(D):

                    # if results for pi are not zero
                    if not self.demand_results.pi[i, j] == 0:
                        pi_initial = self.demand_results.pi[i, j]

                        self.demand_results.pi[i, j] = pi_initial - epsilon / 2
                        with HideOutput():
                            delta_new = self.demand_results.compute_delta()
                        self.demand_results.delta = delta_new

                        # recompute markups
                        markups_l, md, ml = build_markups_all(
                            self.products, self.demand_results, self.models.models_downstream,
                            self.models.ownership_downstream, self.models.models_upstream,
                            self.models.ownership_upstream, self.models.VI, self.models.custom_model_specification
                        )

                        # TODO: this block of code is the same as above
                        self.demand_results.pi[i, j] = pi_initial + epsilon / 2
                        with HideOutput():
                            delta_new = self.demand_results.compute_delta()
                        self.demand_results.delta = delta_new

                        markups_u, mu, mu = build_markups_all(
                            self.products, self.demand_results, self.models.models_downstream,
                            self.models.ownership_downstream, self.models.models_upstream,
                            self.models.ownership_upstream, self.models.VI, self.models.custom_model_specification
                        )

                        # compute first differences for the markups
                        gradient_markups = self._compute_first_difference_markups(
                            markups_u, markups_l, epsilon,theta_index, gradient_markups
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

        # for each ? what is L?
        for zz in range(L):
            Z = self.products["Z{0}".format(zz)]
            K = np.shape(Z)[1]

            # absorb any cost fixed effects from prices, markups, and instruments
            if self._absorb_cost_ids is not None:
                Z_orth, Z_errors = self._absorb_cost_ids(Z)
            tmp = sm.OLS(Z_orth, self.products.w).fit()
            Z_orth = tmp.resid     
            Z_orth = np.reshape(Z_orth, [N, K])

            # initialize variables to store GMM measure of fit Q_m for each model
            g = [None] * M
            Q = [None] * M

            # compute the weight matrix
            WMinv = 1 / N * (Z_orth.T @ Z_orth)
            WMinv = np.reshape(WMinv, [K, K])
            WM = inv(WMinv)
            # TODO: what is this and why is it commented?
            # WM = precisely_invert(WMinv)
            # WM = WM[0]

            # for each model compute GMM measure of fit
            for m in range(M):
                g[m] = 1 / N * (Z_orth.T @ (prices_orthogonal-markups_orthogonal[m]))
                g[m] = np.reshape(g[m], [K, 1])
                Q[m] = g[m].T @ WM @ g[m]

            # compute the pairwise RV numerator
            RV_num = np.zeros((M, M))
            for m in range(M):
                for i in range(m):
                    if i < m:
                        RV_num[i, m] = math.sqrt(N) * (Q[i] - Q[m])
            
            # compute the pairwise RV denominator
            RV_denom = np.zeros((M, M))
            COV_MCS = np.zeros((M, M))
            WM14 = fractional_matrix_power(WM, 0.25)  # TODO: why is this not used?
            WM12 = fractional_matrix_power(WM, 0.5)
            WM34 = fractional_matrix_power(WM, 0.75)

            psi = [None] * M
            dmd_adj = [None] * M
            for m in range(M):
                psi[m] = np.zeros([N, K])
                psi_bar = WM12 @ g[m] - .5 * WM34 @ (WMinv) @ WM34 @ g[m]
                WM34Z = Z_orth @ WM34
                WM34Zg = WM34Z @ g[m]
                psi_i = ((prices_orthogonal - markups_orthogonal[m]) * Z_orth) @ WM12 - 0.5 * WM34Zg * (Z_orth @ WM34.T)
                psi[m] = psi_i - np.transpose(psi_bar)
                if demand_adjustment == 'yes':
                    G_k = -1 / N * np.transpose(Z_orth) @ gradient_markups[m]
                    G_m[m] = G_k
                    dmd_adj[m] = WM12 @ G_m[m] @ inv(H_primeWD @ H) @ H_primeWD
                    psi[m] = psi[m] - (h_i - np.transpose(h)) @ np.transpose(dmd_adj[m])

            M_MCS = np.array(range(M))
            all_combos = list(itertools.combinations(M_MCS, 2))
            Var_MCS = np.zeros([np.shape(all_combos)[0], 1])

            # vii = 0
            for m in range(M):
                for i in range(m):
                    if i < m:
                        VRV_11 = 1 / N * psi[i].T @ psi[i]
                        VRV_22 = 1 / N * psi[m].T @ psi[m]
                        VRV_12 = 1 / N * psi[i].T @ psi[m]
                        if se_type == 'clustered':
                            cluster_ids = np.unique(self.products.clustering_ids)
                            for j in cluster_ids:
                                ind_kk = np.where(self.products.clustering_ids == j)[0]
                                psi1_l = psi[i][ind_kk, :]
                                psi2_l = psi[m][ind_kk, :]
                                psi1_c = psi1_l
                                psi2_c = psi2_l
                                
                                for ii in range(len(ind_kk)-1):
                                    psi1_c = np.roll(psi1_c, 1, axis=0)
                                    psi2_c = np.roll(psi2_c, 1, axis=0)
                                 
                                    VRV_11 = VRV_11 + 1 / N * psi1_l.T @ psi1_c
                                    VRV_22 = VRV_22 + 1 / N * psi2_l.T @ psi2_c
                                    VRV_12 = VRV_12 + 1 / N * psi1_l.T @ psi2_c
                        sigma2 = 4 * (g[i].T @ WM12 @ VRV_11 @ WM12 @ g[i] + g[m].T @ WM12 @ VRV_22 @ WM12 @ g[m] - 2 * g[i].T @ WM12 @ VRV_12 @ WM12 @ g[m])

                        COV_MCS[i, m] = g[i].T @ WM12 @ VRV_12 @ WM12 @ g[m]
                        COV_MCS[m, i] = COV_MCS[i, m]
                        COV_MCS[m, m] = g[m].T @ WM12 @ VRV_22 @ WM12 @ g[m]
                        COV_MCS[i, i] = g[i].T @ WM12 @ VRV_11 @ WM12 @ g[i]

                        RV_denom[i, m] = math.sqrt(sigma2)

            # TODO: what is this block?
            N_combos = np.shape(all_combos)[0]
            Sigma_MCS = np.zeros([N_combos, N_combos])
            for ii in range(N_combos):
                tmp3 = all_combos[ii][0]
                tmp4 = all_combos[ii][1]
                Var_MCS[ii] = RV_denom[tmp3, tmp4]/2
                for i in range(N_combos):
                    tmp1 = all_combos[i][0]
                    tmp2 = all_combos[i][1]
                    Sigma_MCS[i, ii] = COV_MCS[tmp1, tmp3] - COV_MCS[tmp2, tmp3] - COV_MCS[tmp1, tmp4] + COV_MCS[tmp2, tmp4]
            Sigma_MCS = Sigma_MCS / (Var_MCS@Var_MCS.T)
                
            # compute the pairwise RV test statistic
            test_statistic_RV = np.zeros((M, M))
            for m in range(M):
                for i in range(M):
                    if i < m:
                        test_statistic_RV[i, m] = RV_num[i, m] / RV_denom[i, m]
                    if i >= m:
                        test_statistic_RV[i, m] = "NaN"

            # compute the pairwise F-statistic for each model
            F = np.zeros((M, M))
            pi = np.zeros((K, M))
            phi = [None] * M
            for m in range(M):
                ols_results = sm.OLS(prices_orthogonal - markups_orthogonal[m], Z_orth).fit()
                pi[:, m] = ols_results.params
                e = np.reshape(ols_results.resid, [N, 1])
                phi[m] = (e * Z_orth) @ WM
                
                if demand_adjustment == 'yes':
                    phi[m] = phi[m] - (h_i-np.transpose(h)) @ np.transpose(WM12 @ dmd_adj[m])

            # TODO: parts of this block are repeated but with phi instead of psi - can I throw this into a function?
            # TODO: what is phi and what is psi?
            for m in range(M):
                for i in range(M):
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

                        sig2_1 = 1 / K * np.trace(VAR_11 @ WMinv)
                        sig2_2 = 1 / K * np.trace(VAR_22 @ WMinv)
                        sig_12 = 1 / K * np.trace(VAR_12 @ WMinv)
                        sig2_12 = sig_12**2

                        numerator = (sig2_1 - sig2_2) * (sig2_1 - sig2_2)
                        denominator = ((sig2_1 + sig2_2) * (sig2_1 + sig2_2) - 4 * sig2_12)
                        rho2 = numerator / denominator
                        F[i, m] = (1 - rho2) * N / (2 * K) * (sig2_2 * g[i].T @ WM @ g[i] + sig2_1 * g[m].T @ WM @ g[m] - 2 * sig_12 * g[i].T @ WM @ g[m]) / (sig2_1 * sig2_2 - sig2_12)
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
                    
                    N_combos = np.shape(combos)[0]
                    Sig_idx = np.empty(N_combos, dtype=int)
                    for ii in range(N_combos):
                        tmp1.append(combos[ii][0])
                        tmp2.append(combos[ii][1])

                        Sig_idx[ii] = all_combos.index(combos[ii])

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
                    Ndraws = 99999  # TODO: why is this set here?
                    simTRV = np.random.multivariate_normal(mean, cov, (Ndraws))
                    maxsimTRV = np.amax(abs(simTRV), 1)

                    MCS_pvalues[em] = np.mean(maxsimTRV > TRV_max)
                    M_MCS = np.delete(M_MCS, np.where(M_MCS == em))

            g_list[zz] = g
            Q_list[zz] = Q
            RV_numerator_list[zz] = RV_num
            RV_denominator_list[zz] = RV_denom
            test_statistic_RV_list[zz] = test_statistic_RV
            F_statistic_list[zz] = F
            MCS_p_values_list[zz] = MCS_pvalues

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


class Problem(ProblemEconomy):
    r"""A BLP-type problem.

    
    """

    def __init__(
            self, cost_formulation: Formulation, instrument_formulation: Sequence[Formulation], 
            product_data: Mapping, demand_results: Mapping, model_formulations: Sequence[ModelFormulation] = None,
            markup_data: Optional[RecArray] = None) -> None:

        #super().__init__(demand_results)
        """Initialize the underlying economy with product and agent data before absorbing fixed effects."""

        # keep track of long it takes to initialize the problem
        output("Initializing the problem ...")
        start_time = time.time()
        if markup_data is None:
            M = len(model_formulations)
        else:
            M = np.shape(markup_data)[0]

        if hasattr(instrument_formulation, '__len__'):
            L = len(instrument_formulation)
        else:
            L = 1

        # validate and normalize cost formulation
        if not isinstance(cost_formulation, Formulation):
            raise TypeError("cost_formulation must be a single Formulation instance.")

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
            models = Models(model_formulations = model_formulations, product_data = product_data)
            markups = [None]*M
        else:
            models = None
            markups = markup_data

        super().__init__(
            cost_formulation, instrument_formulation, model_formulations, products, models, demand_results, markups
        )

        if max(options.collinear_atol, options.collinear_rtol) > 0:

            tmp = self.products.w

            common_message = "To disable collinearity checks, set options.collinear_atol = options.collinear_rtol = 0."
            collinear, successful = precisely_identify_collinearity(tmp)
            if not successful:
                raise ValueError(
                    f"Failed to compute the QR decomposition of w while checking for collinearity issues. "
                    f"{common_message}"
                )
            if collinear.any():
                raise ValueError(
                    f"Detected collinearity issues with w . "
                    f"{common_message}"
                )
            for zz in range(self.L): 
                tmp = self.products.w
                tmp = np.append(tmp,self.products["Z{0}".format(zz)], axis=1)   
                collinear, successful = precisely_identify_collinearity(tmp)
                if not successful:
                    raise ValueError(
                        f"Failed to compute the QR decomposition of [w,z"+str(zz)+"] while checking for collinearity "
                        f"issues. "
                        f"{common_message}"
                    )
                if collinear.any():
                    raise ValueError(
                        f"Detected collinearity issues with [w,z"+str(zz)+"]. "
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
    MCS_pvalues: Array

    def __init__(
            self, problem: ProblemEconomy, markups: Array, markups_downstream: Array, markups_upstream: Array,
            mc: Array, taus: Array, g: Array, Q: Array, RV_num: Array, RV_denom: Array, test_statistic_RV: Array,
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
        self.RV_numerator = RV_num
        self.RV_denominator = RV_denom
        self.test_statistic_RV = test_statistic_RV
        self.F = F
        self.MCS_pvalues = MCS_pvalues
