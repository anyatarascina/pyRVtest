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

    # TODO: there must be a way to parse this out into multiple functions. very long and messy right now
    # TODO: needs to be commented
    def solve(
            self, method: str = 'both', demand_adjustment: str = 'no', se_type: str = 'unadjusted') -> ProblemResults:
        r"""Solve the problem.

        # TODO: add docstring

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
 
        # compute the markups
        M = self.M
        N = self.N
        L = self.L
        Dict_K = self.Dict_K 
        markups = self.markups
        # TODO: not sure why these vars are initialized like this
        markups_upstream = [None]*M
        markups_downstream = [None]*M
        mc = [None]*M
        markups_orth = [None]*M
        mc_orth = [None]*M
        taus = [None]*M
        markups_errors = [None]*M
        mc_errors = [None]*M

        if markups[0] is None:
            print('Computing Markups ... ')
            markups, markups_downstream, markups_upstream = build_markups_all(
                self.products, self.demand_results, self.models.models_downstream, self.models.ownership_downstream,
                self.models.models_upstream, self.models.ownership_upstream, self.models.VI
            )

        for kk in range(M):
            mc[kk] = self.products.prices - markups[kk]

        # if demand_adjustment is yes, get finite differences approx to the derivative of markups wrt theta
        # TODO: add this feature

        # absorb any cost fixed effects from prices, markups, and instruments
        if self._absorb_cost_ids is not None:
            output("Absorbing cost-side fixed effects ...")
            self.products.w, w_errors = self._absorb_cost_ids(self.products.w)
            prices_orth, prices_errors = self._absorb_cost_ids(self.products.prices)
            
            for kk in range(M):
                markups_orth[kk], markups_errors[kk] = self._absorb_cost_ids(markups[kk])
                mc_orth[kk], mc_errors[kk] = self._absorb_cost_ids(mc[kk])

        # residualize prices, markups, and instruments w.r.t cost shifters w and recover the tau parameters in cost
        #   regression on w
        tmp = sm.OLS(prices_orth, self.products.w).fit()
        prices_orth = tmp.resid
        prices_orth = np.reshape(prices_orth, [N, 1])
        
        for kk in range(M):
            tmp = sm.OLS(markups_orth[kk], self.products.w).fit()
            markups_orth[kk] = tmp.resid
            markups_orth[kk] = np.reshape(markups_orth[kk], [N, 1])
            tmp = sm.OLS(mc_orth[kk], self.products.w).fit()
            taus[kk] = tmp.params

        if demand_adjustment == 'yes':
            ZD = self.demand_results.problem.products.ZD
            for jj in range(len(self.demand_results.problem.products.dtype.fields['X1'][2])):
                if self.demand_results.problem.products.dtype.fields['X1'][2][jj] == 'prices':
                    price_col = jj

            XD = np.delete(self.demand_results.problem.products.X1, price_col, 1)
            WD = self.demand_results.updated_W
            h = self.demand_results.moments
            
            dy_dtheta = np.append(
                self.demand_results.xi_by_theta_jacobian, -self.demand_results.problem.products.prices, 1
            )
            dy_dtheta = self.demand_results.problem._absorb_demand_ids(dy_dtheta)
            dy_dtheta = np.reshape(dy_dtheta[0], [N, len(self.demand_results.theta)+1])

            if np.shape(XD)[1] == 0:
                dxi_dtheta = dy_dtheta
            else:    
                dxi_dtheta = dy_dtheta - XD@inv(XD.T@ZD@WD@ZD.T@XD)@(XD.T@ZD@WD@ZD.T@dy_dtheta)
            
            H = 1/N*(np.transpose(ZD)@dxi_dtheta)
            
            H_prime = np.transpose(H)
            H_primeWD = H_prime@WD
            hi = ZD*self.demand_results.xi
            K2 = self.demand_results.problem.K2
            D = self.demand_results.problem.D

            # build adjustment to psi for each model
            eps = options.finite_differences_epsilon
            Gm = [None]*M
            grad_markups = [None]*M

            # get numerical derivatives of markups wrt theta, alpha
            for kk in range(M):
                grad_markups[kk] = np.zeros((N, len(self.demand_results.theta)+1))

            # sigma
            ind_theta = 0
            delta_est = self.demand_results.delta
            for jj in range(K2):
                for ll in range(K2):
                    if not self.demand_results.sigma[jj, ll] == 0:
                        tmp_sig = self.demand_results.sigma[jj, ll]

                        # reduce sigma by small increment
                        self.demand_results.sigma[jj, ll] = tmp_sig - eps/2


                        # update delta
                        with HideOutput():
                            delta_new = self.demand_results.compute_delta()
                        self.demand_results.delta = delta_new
                        
                        # recompute markups
                        markups_l, md, ml = build_markups_all(
                            self.products, self.demand_results, self.models.models_downstream,
                            self.models.ownership_downstream, self.models.models_upstream,
                            self.models.ownership_upstream, self.models.VI
                        )

                        # increase sigma by small increment
                        self.demand_results.sigma[jj, ll] = tmp_sig + eps/2

                        # update delta
                        with HideOutput():
                            delta_new = self.demand_results.compute_delta()
                        self.demand_results.delta = delta_new
                        
                        # recompute markups
                        markups_u, mu, mu = build_markups_all(
                            self.products, self.demand_results, self.models.models_downstream,
                            self.models.ownership_downstream, self.models.models_upstream,
                            self.models.ownership_upstream, self.models.VI
                        )
                        
                        # compute first difference approximation of derivative of markups
                        for kk in range(M):
                            diff_markups = (markups_u[kk]-markups_l[kk])/eps
                            diff_markups, me = self._absorb_cost_ids(diff_markups)
                            tmp = sm.OLS(diff_markups, self.products.w).fit()
                            grad_markups[kk][:, ind_theta] = tmp.resid
                        self.demand_results.sigma[jj, ll] = tmp_sig
                        ind_theta = ind_theta+1

            # pi
            for jj in range(K2):
                for ll in range(D):
                    if not self.demand_results.pi[jj, ll] == 0:
                        tmp_pi = self.demand_results.pi[jj, ll]
                        self.demand_results.pi[jj, ll] = tmp_pi - eps/2
                        with HideOutput():
                            delta_new = self.demand_results.compute_delta()
                        self.demand_results.delta = delta_new
                        markups_l, md, ml = build_markups_all(
                            self.products, self.demand_results, self.models.models_downstream,
                            self.models.ownership_downstream, self.models.models_upstream,
                            self.models.ownership_upstream, self.models.VI
                        )
                        self.demand_results.pi[jj, ll] = tmp_pi + eps/2
                        with HideOutput():
                            delta_new = self.demand_results.compute_delta()
                        self.demand_results.delta = delta_new
                        markups_u, mu, mu = build_markups_all(
                            self.products, self.demand_results, self.models.models_downstream,
                            self.models.ownership_downstream, self.models.models_upstream,
                            self.models.ownership_upstream, self.models.VI
                        )
                        for kk in range(M):
                            diff_markups = (markups_u[kk]-markups_l[kk])/eps
                            diff_markups, me = self._absorb_cost_ids(diff_markups)
                            tmp = sm.OLS(diff_markups, self.products.w).fit()
                            grad_markups[kk][:, ind_theta] = tmp.resid
                        self.demand_results.pi[jj, ll] = tmp_pi
                        ind_theta = ind_theta+1                
            
            self.demand_results.delta = delta_est             
                
            # alpha
            for jj in range(len(self.demand_results.beta)):
                if self.demand_results.beta_labels[jj] == 'prices':
                    tmp_alpha = self.demand_results.beta[jj]
                    self.demand_results.beta[jj] = tmp_alpha - eps/2
                    markups_l, md, ml = build_markups_all(
                        self.products, self.demand_results, self.models.models_downstream,
                        self.models.ownership_downstream, self.models.models_upstream, self.models.ownership_upstream,
                        self.models.VI
                    )
                    self.demand_results.beta[jj] = tmp_alpha + eps/2
                    markups_u, mu, mu = build_markups_all(
                        self.products, self.demand_results, self.models.models_downstream,
                        self.models.ownership_downstream, self.models.models_upstream, self.models.ownership_upstream,
                        self.models.VI
                    )
                    for kk in range(M):
                        diff_markups = (markups_u[kk]-markups_l[kk])/eps
                        diff_markups, me = self._absorb_cost_ids(diff_markups)
                        tmp = sm.OLS(diff_markups, self.products.w).fit()
                        grad_markups[kk][:, ind_theta] = tmp.resid
                    self.demand_results.beta[jj] = tmp_alpha
                    ind_theta = ind_theta+1 

        g_ALL = [None]*L
        Q_ALL = [None]*L
        RV_num_ALL = [None]*L
        RV_denom_ALL = [None]*L
        TRV_ALL = [None]*L
        F_ALL = [None]*L
        MCS_pvalues_ALL = [None]*L
    
        for zz in range(L):
            Z = self.products["Z{0}".format(zz)]
            K = np.shape(Z)[1]

            # absorb any cost fixed effects from prices, markups, and instruments
            if self._absorb_cost_ids is not None:
                Z_orth, Z_errors = self._absorb_cost_ids(Z)
                
            tmp = sm.OLS(Z_orth, self.products.w).fit()
            Z_orth = tmp.resid     
            Z_orth = np.reshape(Z_orth, [N, K])

            # get the GMM measure of fit Q_m for each model
            g = [None]*M
            Q = [None]*M
            WMinv = 1/N*(Z_orth.T@Z_orth)
            WMinv = np.reshape(WMinv, [K, K])

            # TODO: what is this and why is it commented?
            WM = inv(WMinv)
            # WM = precisely_invert(WMinv)
            # WM = WM[0]

            for kk in range(M):
                g[kk] = 1/N*(Z_orth.T@(prices_orth-markups_orth[kk])) 
                g[kk] = np.reshape(g[kk], [K, 1])
                Q[kk] = g[kk].T@WM@g[kk] 

            # compute the pairwise RV numerator
            RV_num = np.zeros((M, M))
            for kk in range(M):
                for jj in range(kk):
                    if jj < kk:
                        RV_num[jj, kk] = math.sqrt(N) * (Q[jj] - Q[kk])
            
            # compute the pairwise RV denominator
            RV_denom = np.zeros((M, M))
            COV_MCS = np.zeros((M, M))
            WM14 = fractional_matrix_power(WM, 0.25)
            WM12 = fractional_matrix_power(WM, 0.5)
            WM34 = fractional_matrix_power(WM, 0.75)

            psi = [None]*M
            dmd_adj = [None]*M
            for kk in range(M):
                psi[kk] = np.zeros([N, K])
                psi_bar = WM12@g[kk]-.5*WM34@(WMinv)@WM34@g[kk]   
                WM34Z = Z_orth@WM34
                WM34Zg = WM34Z@g[kk]
                psi_i = ((prices_orth-markups_orth[kk])*Z_orth)@WM12-0.5*WM34Zg*(Z_orth@WM34.T)
                psi[kk] = psi_i-np.transpose(psi_bar)
                if demand_adjustment == 'yes':
                    G_k = -1/N*np.transpose(Z_orth)@grad_markups[kk]
                    Gm[kk] = G_k
                    dmd_adj[kk] = WM12@Gm[kk]@inv(H_primeWD@H)@H_primeWD
                    psi[kk] = psi[kk] - (hi-np.transpose(h))@np.transpose(dmd_adj[kk])

            M_MCS = np.array(range(M))
            all_combos = list(itertools.combinations(M_MCS, 2))
            Var_MCS = np.zeros([np.shape(all_combos)[0], 1])

            # vii = 0
            for kk in range(M):
                for jj in range(kk):
                    if jj < kk:  
                        VRV_11 = 1/N*psi[jj].T@psi[jj]       
                        VRV_22 = 1/N*psi[kk].T@psi[kk] 
                        VRV_12 = 1/N*psi[jj].T@psi[kk] 
                        if se_type == 'clustered':
                            cids = np.unique(self.products.clustering_ids)
                            for ll in cids:
                                ind_kk = np.where(self.products.clustering_ids == ll)[0]
                                psi1_l = psi[jj][ind_kk, :]
                                psi2_l = psi[kk][ind_kk, :]
                                
                                psi1_c = psi1_l
                                psi2_c = psi2_l
                                
                                for ii in range(len(ind_kk)-1):
                                    psi1_c = np.roll(psi1_c, 1, axis=0)
                                    psi2_c = np.roll(psi2_c, 1, axis=0)
                                 
                                    VRV_11 = VRV_11 + 1/N*psi1_l.T@psi1_c
                                    VRV_22 = VRV_22 + 1/N*psi2_l.T@psi2_c
                                    VRV_12 = VRV_12 + 1/N*psi1_l.T@psi2_c    
                        sigma2 = 4*(g[jj].T@WM12@VRV_11@WM12@g[jj]+g[kk].T@WM12@VRV_22@WM12@g[kk]-2*g[jj].T@WM12@VRV_12@WM12@g[kk])
                        
                        COV_MCS[jj, kk] = g[jj].T@WM12@VRV_12@WM12@g[kk]
                        COV_MCS[kk, jj] = COV_MCS[jj, kk]
                        COV_MCS[kk, kk] = g[kk].T@WM12@VRV_22@WM12@g[kk]
                        COV_MCS[jj, jj] = g[jj].T@WM12@VRV_11@WM12@g[jj]
                        
                        RV_denom[jj, kk] = math.sqrt(sigma2)
                
            Ncombos = np.shape(all_combos)[0]
            Sigma_MCS = np.zeros([Ncombos, Ncombos])
            for ii in range(Ncombos):
                tmp3 = all_combos[ii][0]
                tmp4 = all_combos[ii][1]
                
                Var_MCS[ii] = RV_denom[tmp3, tmp4]/2
                for jj in range(Ncombos):
                    tmp1 = all_combos[jj][0]
                    tmp2 = all_combos[jj][1]
                    Sigma_MCS[jj, ii] = COV_MCS[tmp1, tmp3]-COV_MCS[tmp2, tmp3] - COV_MCS[tmp1, tmp4] + COV_MCS[tmp2, tmp4]
            
            Sigma_MCS = Sigma_MCS/(Var_MCS@Var_MCS.T)
                
            # compute the pairwise RV test statistic
            TRV = np.zeros((M,M))
            for kk in range(M):
                for jj in range(M):
                    if jj < kk:  
                        TRV[jj, kk] = RV_num[jj, kk]/RV_denom[jj, kk]
                    if jj >= kk:
                        TRV[jj, kk] = "NaN"

            # compute the pairwise F-statistic
            F = np.zeros((M, M))
            pi = np.zeros((K, M))
            phi = [None]*M

            for kk in range(M):
                tmp = sm.OLS(prices_orth-markups_orth[kk],Z_orth).fit() 
                pi[:,kk] = tmp.params
                e = np.reshape(tmp.resid,[N,1])
                phi[kk] = (e*Z_orth)@WM
                
                if demand_adjustment == 'yes':
                    phi[kk] = phi[kk] - (hi-np.transpose(h))@np.transpose(WM12@dmd_adj[kk])

            for kk in range(M):
                for jj in range(M):
                    if jj < kk:  
                        VAR_11 = 1/N*phi[jj].T@phi[jj]       
                        VAR_22 = 1/N*phi[kk].T@phi[kk] 
                        VAR_12 = 1/N*phi[jj].T@phi[kk] 
                        if se_type == 'clustered':
                            cids = np.unique(self.products.clustering_ids)
                            for ll in cids:
                                ind_kk = np.where(self.products.clustering_ids == ll)[0]
                                phi1_l = phi[jj][ind_kk,:]
                                phi2_l = phi[kk][ind_kk,:]
                                
                                phi1_c = phi1_l
                                phi2_c = phi2_l
                                
                                for ii in range(len(ind_kk)-1):
                                    phi1_c = np.roll(phi1_c,1,axis=0)
                                    phi2_c = np.roll(phi2_c,1,axis=0)
                                 
                                    VAR_11 = VAR_11 + 1/N*phi1_l.T@phi1_c
                                    VAR_22 = VAR_22 + 1/N*phi2_l.T@phi2_c
                                    VAR_12 = VAR_12 + 1/N*phi1_l.T@phi2_c

                        sig2_1 = 1/K*np.trace(VAR_11@WMinv)
                        sig2_2 = 1/K*np.trace(VAR_22@WMinv)      
                        sig_12 = 1/K*np.trace(VAR_12@WMinv)
                        sig2_12 = sig_12**2

                        temp = (sig2_1-sig2_2)*(sig2_1-sig2_2)
                        rho2 = temp/((sig2_1+sig2_2)*(sig2_1+sig2_2)-4*sig2_12)
                        F[jj, kk] = (1-rho2)*N/(2*K)*(sig2_2*g[jj].T@WM@g[jj] + sig2_1*g[kk].T@WM@g[kk] - 2*sig_12*g[jj].T@WM@g[kk])/(sig2_1*sig2_2-sig2_12)
                    if jj >= kk:
                        F[jj, kk] = "NaN"

            MCS_pvalues = np.ones([M, 1])

            np.random.seed(123)
                
            stop = 0
            while stop == 0:
                if np.shape(M_MCS)[0] == 2:
                    TRVmax = TRV[M_MCS[0], M_MCS[1]]

                    if np.sign(TRVmax) >= 0:
                        em = M_MCS[0]
                        TRVmax = -TRVmax
                    else:
                        em = M_MCS[1]

                    MCS_pvalues[em] = 2*norm.cdf(TRVmax)

                    stop = 1
                else:      

                    combos = list(itertools.combinations(M_MCS, 2))
                    tmp1 = []
                    tmp2 = []
                    
                    Ncombos = np.shape(combos)[0]
                    Sig_idx = np.empty(Ncombos, dtype=int)
                    for ii in range(Ncombos):
                        tmp1.append(combos[ii][0])
                        tmp2.append(combos[ii][1])

                        Sig_idx[ii] = all_combos.index(combos[ii])

                    TRV_MCS = TRV[tmp1,tmp2]
                    index = np.argmax(abs(TRV_MCS))
                    TRVmax = TRV_MCS[index]

                    if np.sign(TRVmax) >= 0:
                        em = tmp1[index]
                    else:
                        em = tmp2[index]
                        TRVmax = -TRVmax
                       
                    mean = np.zeros([np.shape(combos)[0]]) 
                    cov = Sigma_MCS[Sig_idx[:, None], Sig_idx]
                    Ndraws = 99999
                    simTRV = np.random.multivariate_normal(mean, cov, (Ndraws))
                    maxsimTRV = np.amax(abs(simTRV), 1)

                    MCS_pvalues[em] = np.mean(maxsimTRV > TRVmax)
                    M_MCS = np.delete(M_MCS,np.where(M_MCS == em))

            g_ALL[zz] = g
            Q_ALL[zz] = Q
            RV_num_ALL[zz] = RV_num
            RV_denom_ALL[zz] = RV_denom
            TRV_ALL[zz] = TRV
            F_ALL[zz] = F
            MCS_pvalues_ALL[zz] = MCS_pvalues

        results = ProblemResults(Progress(
            self, markups, markups_downstream, markups_upstream, mc, taus, g_ALL, Q_ALL, RV_num_ALL, RV_denom_ALL,
            TRV_ALL, F_ALL, MCS_pvalues_ALL)
        )
        step_end_time = time.time()
        tot_time = step_end_time-step_start_time
        print('Total Time is ... ' + str(tot_time))
        output("")
        output(results)
        return results
        

class Problem(ProblemEconomy):
#class Problem(ProblemEconomy):
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
    taus: Array
    mc: Array
    g: Array
    Q: Array
    RV_num: Array
    RV_denom: Array
    TRV: Array
    F: Array
    MCS_pvalues: Array

    def __init__(
            self, problem: ProblemEconomy, markups: Array, markups_downstream: Array, markups_upstream: Array,
            mc: Array, taus: Array, g: Array, Q: Array, RV_num: Array, RV_denom: Array, TRV: Array, F: Array,
            MCS_pvalues: Array
            ) -> None:
        """Store progress information, compute the projected gradient and its norm, and compute the reduced Hessian."""
        super().__init__(
            problem
        )
        self.markups = markups
        self.markups_downstream = markups_downstream
        self.markups_upstream = markups_upstream
        self.taus = taus
        self.mc = mc
        self.g = g
        self.Q = Q
        self.RV_num = RV_num
        self.RV_denom = RV_denom
        self.TRV = TRV
        self.F = F
        self.MCS_pvalues = MCS_pvalues

