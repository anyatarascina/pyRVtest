"""Economy-level BLP problem functionality."""

import abc
import collections.abc
import functools
import time
import statsmodels.api as sm 
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
import math
import numpy as np
import itertools
from scipy.linalg import inv, fractional_matrix_power
from scipy.stats import norm
from .economy import Economy
from .. import exceptions, options
from ..configurations.formulation import Formulation, ModelFormulation
from ..configurations.integration import Integration
from ..configurations.iteration import Iteration
from ..configurations.optimization import ObjectiveResults, Optimization
from ..markets.problem_market import ProblemMarket
from ..moments import Moment, CustomMoment, EconomyMoments
from ..parameters import Parameters
from ..primitives import Models, Products
from ..results.problem_results import ProblemResults
from ..utilities.algebra import precisely_identify_collinearity, precisely_invert
from ..utilities.basics import (
    Array, Bounds, Error, RecArray, SolverStats, format_number, format_seconds, format_table, generate_items, output,
    update_matrices, compute_finite_differences
)
from ..utilities.statistics import IV, compute_gmm_moments_mean, compute_gmm_moments_jacobian_mean
from pyblp.results.problem_results import ProblemResults as pyblpProblemResults
from ..construction import build_markups, build_markups_all


class ProblemEconomy(Economy):
    """An abstract BLP problem."""

    @abc.abstractmethod
    def __init__(
            self, cost_formulation: Formulation, instrument_formulation: Sequence[Formulation], 
            model_formulations: Sequence[ModelFormulation],
            products: RecArray, models: RecArray, demand_results: Mapping, markups: RecArray) -> None:
        """Initialize the underlying economy with product and agent data."""
        super().__init__(
            cost_formulation, instrument_formulation, model_formulations, products, models, demand_results, markups
        )

    def solve(
            self, method: str = 'both', demand_adjustment: str = 'no', se_type: str = 'unadjusted') -> ProblemResults:
        r"""Solve the problem.

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
 
        # Compute the markups
        M = self.M
        N = self.N
        L = self.L
        Dict_K = self.Dict_K 
        markups = self.markups
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
            markups, markups_downstream, markups_upstream = build_markups_all(self.products, self.demand_results,
                self.models.models_downstream, self.models.ownership_downstream,
                self.models.models_upstream, self.models.ownership_upstream,
                self.models.VI)

        
        for kk in range(M):
            mc[kk] = self.products.prices - markups[kk]
        

        # if demand_adjustment is yes, get finite differences approx to the derivative of markups wrt theta
        # TO BE ADDED



        # absorb any cost fixed effects from prices, markups, and instruments
        if self._absorb_cost_ids is not None:
            output("Absorbing cost-side fixed effects ...")
            self.products.w, w_errors = self._absorb_cost_ids(self.products.w)
            prices_orth, prices_errors = self._absorb_cost_ids(self.products.prices)
            
            for kk in range(M):
                markups_orth[kk], markups_errors[kk] = self._absorb_cost_ids(markups[kk])
                mc_orth[kk], mc_errors[kk] = self._absorb_cost_ids(mc[kk])

        # residualize prices, markups, and instruments w.r.t cost shifters w 
        # Also, recover the tau parameters in cost regression on w
        
        tmp = sm.OLS(prices_orth,self.products.w).fit()    
        prices_orth = tmp.resid
        prices_orth = np.reshape(prices_orth,[N,1])
        
        for kk in range(M):
            tmp = sm.OLS(markups_orth[kk],self.products.w).fit()    
            markups_orth[kk] = tmp.resid
            markups_orth[kk] = np.reshape(markups_orth[kk],[N,1]) 
            tmp = sm.OLS(mc_orth[kk],self.products.w).fit() 
            taus[kk] = tmp.params

        if demand_adjustment == 'yes':
            ZD = self.demand_results.problem.products.ZD
            for jj in range(len(self.demand_results.problem.products.dtype.fields['X1'][2])):
                if self.demand_results.problem.products.dtype.fields['X1'][2][jj] == 'prices':
                    price_col = jj

            XD = np.delete(self.demand_results.problem.products.X1,price_col,1)
            WD = self.demand_results.updated_W
            h = self.demand_results.moments
            
            dy_dtheta = np.append(self.demand_results.xi_by_theta_jacobian,-self.demand_results.problem.products.prices,1)
            dy_dtheta = self.demand_results.problem._absorb_demand_ids(dy_dtheta)
            dy_dtheta = np.reshape(dy_dtheta[0],[N,len(self.demand_results.theta)+1])

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

            #Build adjustment to psi for each model
            eps = options.finite_differences_epsilon
            Gm = [None]*M
            grad_markups = [None]*M
            
            

            #Get numerical derivatives of markups wrt theta, alpha
            for kk in range(M):
                grad_markups[kk] = np.zeros((N,len(self.demand_results.theta)+1))

            #sigma
            ind_theta = 0
            delta_est = self.demand_results.delta
            for jj in range(K2):
                for ll in range(K2):
                    if not self.demand_results.sigma[jj,ll] == 0:
                        tmp_sig = self.demand_results.sigma[jj,ll]

                        #reduce sigma by small increment
                        self.demand_results.sigma[jj,ll] = tmp_sig - eps/2

                        #update delta
                        delta_new = self.demand_results.compute_delta()
                        self.demand_results.delta = delta_new
                        
                        #recompute markups
                        markups_l, md, ml = build_markups_all(self.products, self.demand_results,
                            self.models.models_downstream, self.models.ownership_downstream,
                            self.models.models_upstream, self.models.ownership_upstream,
                            self.models.VI)

                        #increase sigma by small increment
                        self.demand_results.sigma[jj,ll] = tmp_sig + eps/2
                        #update delta
                        delta_new = self.demand_results.compute_delta()
                        self.demand_results.delta = delta_new
                        
                        #recompute markups
                        markups_u, mu, mu = build_markups_all(self.products, self.demand_results,
                            self.models.models_downstream, self.models.ownership_downstream,
                            self.models.models_upstream, self.models.ownership_upstream,
                            self.models.VI)
                        
                        # Compute first difference approximation of derivative of markups
                        for kk in range(M):
                            diff_markups = (markups_u[kk]-markups_l[kk])/eps
                            diff_markups, me = self._absorb_cost_ids(diff_markups)
                            tmp = sm.OLS(diff_markups,self.products.w).fit()
                            grad_markups[kk][:,ind_theta] = tmp.resid
                        self.demand_results.sigma[jj,ll] = tmp_sig
                        ind_theta = ind_theta+1
            #pi            
            for jj in range(K2):
                for ll in range(D):
                    if not self.demand_results.pi[jj,ll] == 0:
                        tmp_pi = self.demand_results.pi[jj,ll]
                        self.demand_results.pi[jj,ll] = tmp_pi - eps/2
                        delta_new = self.demand_results.compute_delta()
                        self.demand_results.delta = delta_new
                        markups_l, md, ml = build_markups_all(self.products, self.demand_results,
                            self.models.models_downstream, self.models.ownership_downstream,
                            self.models.models_upstream, self.models.ownership_upstream,
                            self.models.VI)
                        self.demand_results.pi[jj,ll] = tmp_pi + eps/2
                        delta_new = self.demand_results.compute_delta()
                        self.demand_results.delta = delta_new
                        markups_u, mu, mu = build_markups_all(self.products, self.demand_results,
                            self.models.models_downstream, self.models.ownership_downstream,
                            self.models.models_upstream, self.models.ownership_upstream,
                            self.models.VI)
                        for kk in range(M):
                            diff_markups = (markups_u[kk]-markups_l[kk])/eps
                            diff_markups, me = self._absorb_cost_ids(diff_markups)
                            tmp = sm.OLS(diff_markups,self.products.w).fit()    
                            grad_markups[kk][:,ind_theta] = tmp.resid
                        self.demand_results.pi[jj,ll] = tmp_pi
                        ind_theta = ind_theta+1                
            
            self.demand_results.delta = delta_est             
                
            #alpha
            for jj in range(len(self.demand_results.beta)):
                if self.demand_results.beta_labels[jj] == 'prices':
                    tmp_alpha = self.demand_results.beta[jj]
                    self.demand_results.beta[jj] = tmp_alpha - eps/2
                    markups_l, md, ml = build_markups_all(self.products, self.demand_results,
                        self.models.models_downstream, self.models.ownership_downstream,
                        self.models.models_upstream, self.models.ownership_upstream,
                        self.models.VI)
                    self.demand_results.beta[jj] = tmp_alpha + eps/2
                    markups_u, mu, mu = build_markups_all(self.products, self.demand_results,
                        self.models.models_downstream, self.models.ownership_downstream,
                        self.models.models_upstream, self.models.ownership_upstream,
                        self.models.VI)
                    for kk in range(M):
                        diff_markups = (markups_u[kk]-markups_l[kk])/eps
                        diff_markups, me = self._absorb_cost_ids(diff_markups)
                        tmp = sm.OLS(diff_markups,self.products.w).fit()    
                        grad_markups[kk][:,ind_theta] = tmp.resid
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
                
            tmp = sm.OLS(Z_orth,self.products.w).fit()    
            Z_orth = tmp.resid     
            Z_orth = np.reshape(Z_orth,[N,K])
            # Get the GMM measure of fit Q_m for each model
            g = [None]*M
            Q = [None]*M
            WMinv =  1/N*(Z_orth.T@Z_orth)
            WMinv = np.reshape(WMinv,[K,K])
            
            WM = inv(WMinv)

            #WM = precisely_invert(WMinv)
            #WM = WM[0]


            for kk in range(M):
                g[kk] = 1/N*(Z_orth.T@(prices_orth-markups_orth[kk])) 
                g[kk] = np.reshape(g[kk],[K,1])
                Q[kk] = g[kk].T@WM@g[kk] 
            # Compute the pairwise RV numerator
            RV_num = np.zeros((M,M))
            for kk in range(M):
                for jj in range(kk):
                    if jj < kk:
                        RV_num[jj,kk] = math.sqrt(N)*(Q[jj]-Q[kk])     
            
            # Compute the pairwise RV denominator      
            RV_denom = np.zeros((M,M))
            COV_MCS = np.zeros((M,M))
            WM14 = fractional_matrix_power(WM, 0.25)
            WM12 = fractional_matrix_power(WM, 0.5)
            WM34 = fractional_matrix_power(WM, 0.75)

            psi = [None]*M
            dmd_adj = [None]*M
            for kk in range(M):
                psi[kk] = np.zeros([N,K])
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
            Var_MCS = np.zeros([np.shape(all_combos)[0],1])




            #vii = 0
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
                                psi1_l = psi[jj][ind_kk,:]
                                psi2_l = psi[kk][ind_kk,:]
                                
                                psi1_c = psi1_l
                                psi2_c = psi2_l
                                
                                for ii in range(len(ind_kk)-1):
                                    psi1_c = np.roll(psi1_c,1,axis=0)
                                    psi2_c = np.roll(psi2_c,1,axis=0)
                                 
                                    VRV_11 = VRV_11 + 1/N*psi1_l.T@psi1_c
                                    VRV_22 = VRV_22 + 1/N*psi2_l.T@psi2_c
                                    VRV_12 = VRV_12 + 1/N*psi1_l.T@psi2_c    
                        sigma2 = 4*(g[jj].T@WM12@VRV_11@WM12@g[jj]+g[kk].T@WM12@VRV_22@WM12@g[kk]-2*g[jj].T@WM12@VRV_12@WM12@g[kk])
                        
                        COV_MCS[jj,kk] = g[jj].T@WM12@VRV_12@WM12@g[kk]
                        COV_MCS[kk,jj] = COV_MCS[jj,kk]
                        COV_MCS[kk,kk] = g[kk].T@WM12@VRV_22@WM12@g[kk]
                        COV_MCS[jj,jj] = g[jj].T@WM12@VRV_11@WM12@g[jj] 
                        
                        RV_denom[jj,kk] = math.sqrt(sigma2)
                
            Ncombos = np.shape(all_combos)[0]
            Sigma_MCS = np.zeros([Ncombos,Ncombos])
            for ii in range(Ncombos):
                tmp3 = all_combos[ii][0]
                tmp4 = all_combos[ii][1]
                
                Var_MCS[ii] = RV_denom[tmp3,tmp4]/2
                for jj in range(Ncombos):
                    tmp1 = all_combos[jj][0]
                    tmp2 = all_combos[jj][1]
                    Sigma_MCS[jj,ii]  = COV_MCS[tmp1,tmp3]-COV_MCS[tmp2,tmp3]-COV_MCS[tmp1,tmp4]+COV_MCS[tmp2,tmp4]
            
            Sigma_MCS = Sigma_MCS/(Var_MCS@Var_MCS.T)
                
            # Compute the pairwise RV test statistic           
            TRV = np.zeros((M,M))
            for kk in range(M):
                for jj in range(kk):
                    if jj < kk:  
                        TRV[jj,kk] = RV_num[jj,kk]/RV_denom[jj,kk]

            # Compute the pairwise F-statistic
            F = np.zeros((M,M))
            pi = np.zeros((K,M))
            phi = [None]*M

            for kk in range(M):
                tmp = sm.OLS(prices_orth-markups_orth[kk],Z_orth).fit() 
                pi[:,kk] = tmp.params
                e = np.reshape(tmp.resid,[N,1])
                phi[kk] = (e*Z_orth)@WM
                
                if demand_adjustment == 'yes':
                    phi[kk] = phi[kk] - (hi-np.transpose(h))@np.transpose(WM12@dmd_adj[kk])

            for kk in range(M):
                for jj in range(kk):
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

                        temp =  (sig2_1-sig2_2)*(sig2_1-sig2_2)
                        rho2 = temp/((sig2_1+sig2_2)*(sig2_1+sig2_2)-4*sig2_12)
                        F[jj,kk] = (1-rho2)*N/(2*K)*(sig2_2*g[jj].T@WM@g[jj] + sig2_1*g[kk].T@WM@g[kk] - 2*sig_12*g[jj].T@WM@g[kk])/(sig2_1*sig2_2-sig2_12) 
            
            MCS_pvalues = np.ones([M,1])

            
                
            stop = 0
            while stop == 0:
                if np.shape(M_MCS)[0] == 2:
                    TRVmax = TRV[M_MCS[0],M_MCS[1]]

                    if np.sign(TRVmax)>= 0:
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

                    if np.sign(TRVmax)>= 0:
                        em = tmp1[index]
                    else:
                        em = tmp2[index]
                        TRVmax = -TRVmax
                       
                    mean = np.zeros([np.shape(combos)[0]]) 
                    cov = Sigma_MCS[Sig_idx[:, None], Sig_idx]
                    Ndraws = 99999
                    simTRV = np.random.multivariate_normal(mean, cov, (Ndraws))
                    maxsimTRV = np.amax(abs(simTRV),1)

                    MCS_pvalues[em] = np.mean(maxsimTRV > TRVmax)

                    M_MCS = np.delete(M_MCS,np.where(M_MCS == em))


            g_ALL[zz] = g
            Q_ALL[zz] = Q
            RV_num_ALL[zz] = RV_num
            RV_denom_ALL[zz] = RV_denom
            TRV_ALL[zz] = TRV
            F_ALL[zz] = F
            MCS_pvalues_ALL[zz] = MCS_pvalues

        results = ProblemResults(Progress(self,markups,markups_downstream,markups_upstream,mc,taus,g_ALL,Q_ALL,RV_num_ALL,RV_denom_ALL,TRV_ALL,F_ALL,MCS_pvalues_ALL))
        step_end_time = time.time()
        tot_time = step_end_time-step_start_time
        print('Total Time is ... ' + str(tot_time))
        output("")
        output(results)
        return results
        

    def _compute_progress(
            self, parameters: Parameters, moments: EconomyMoments, iv: IV, W: Array, scale_objective: bool,
            error_behavior: str, error_punishment: float, delta_behavior: str, iteration: Iteration, fp_type: str,
            shares_bounds: Bounds, costs_bounds: Bounds, finite_differences: bool, theta: Array,
            progress: 'InitialProgress', compute_gradient: bool, compute_hessian: bool,
            compute_micro_covariances: bool) -> 'Progress':
        """Compute demand- and supply-side contributions before recovering the linear parameters and structural error
        terms. Then, form the GMM objective value and its gradient. Finally, handle any errors that were encountered
        before structuring relevant progress information.
        """
        errors: List[Error] = []

        # expand theta
        sigma, pi, rho, beta, gamma = parameters.expand(theta)

        # compute demand-side contributions
        compute_jacobians = compute_gradient and not finite_differences
        (
            delta, micro, xi_jacobian, micro_jacobian, micro_covariances, micro_values, clipped_shares, iteration_stats,
            demand_errors
        ) = (
            self._compute_demand_contributions(
                parameters, moments, iteration, fp_type, shares_bounds, sigma, pi, rho, progress, compute_jacobians,
                compute_micro_covariances
            )
        )
        errors.extend(demand_errors)

        # compute supply-side contributions
        if self.K3 == 0:
            tilde_costs = np.full((self.N, 0), np.nan, options.dtype)
            omega_jacobian = np.full((self.N, parameters.P), np.nan, options.dtype)
            clipped_costs = np.zeros((self.N, 1), np.bool)
        else:
            compute_jacobian = compute_gradient and not finite_differences
            tilde_costs, omega_jacobian, clipped_costs, supply_errors = self._compute_supply_contributions(
                parameters, costs_bounds, sigma, pi, rho, beta, delta, xi_jacobian, progress, compute_jacobian
            )
            errors.extend(supply_errors)

        # optionally compute Jacobians with central finite differences
        if compute_gradient and finite_differences and parameters.P > 0:
            def compute_perturbed_stack(perturbed_theta: Array) -> Array:
                """Evaluate a stack of xi, micro moments, and omega at a perturbed parameter vector."""
                perturbed_progress = self._compute_progress(
                    parameters, moments, iv, W, scale_objective, error_behavior, error_punishment, delta_behavior,
                    iteration, fp_type, shares_bounds, costs_bounds, finite_differences=False, theta=perturbed_theta,
                    progress=progress, compute_gradient=False, compute_hessian=False, compute_micro_covariances=False
                )
                perturbed_stack = perturbed_progress.iv_delta
                if moments.MM > 0:
                    perturbed_stack = np.r_[perturbed_stack, perturbed_progress.micro]
                if self.K3 > 0:
                    perturbed_stack = np.r_[perturbed_stack, perturbed_progress.iv_tilde_costs]
                return perturbed_stack

            # compute and unstack the Jacobians
            stack_jacobian = compute_finite_differences(compute_perturbed_stack, theta)
            xi_jacobian = stack_jacobian[:self.N]
            if moments.MM > 0:
                micro_jacobian = stack_jacobian[self.N:self.N + moments.MM]
            if self.K3 > 0:
                omega_jacobian = stack_jacobian[-self.N:]

        # subtract contributions of linear parameters in theta
        iv_delta = delta.copy()
        iv_tilde_costs = tilde_costs.copy()
        if not parameters.eliminated_beta_index.all():
            theta_beta = np.c_[beta[~parameters.eliminated_beta_index]]
            iv_delta -= self._compute_true_X1(index=~parameters.eliminated_beta_index.flatten()) @ theta_beta
        if not parameters.eliminated_gamma_index.all():
            theta_gamma = np.c_[gamma[~parameters.eliminated_gamma_index]]
            iv_tilde_costs -= self._compute_true_X3(index=~parameters.eliminated_gamma_index.flatten()) @ theta_gamma

        # absorb any fixed effects
        if self._absorb_demand_ids is not None:
            iv_delta, demand_absorption_errors = self._absorb_demand_ids(iv_delta)
            errors.extend(demand_absorption_errors)
        if self._absorb_supply_ids is not None:
            iv_tilde_costs, supply_absorption_errors = self._absorb_supply_ids(iv_tilde_costs)
            errors.extend(supply_absorption_errors)

        # collect inputs into GMM estimation
        X_list = [self.products.X1[:, parameters.eliminated_beta_index.flat]]
        Z_list = [self.products.ZD]
        y_list = [iv_delta]
        jacobian_list = [xi_jacobian]
        if self.K3 > 0:
            X_list.append(self.products.X3[:, parameters.eliminated_gamma_index.flat])
            Z_list.append(self.products.ZS)
            y_list.append(iv_tilde_costs)
            jacobian_list.append(omega_jacobian)

        # recover the linear parameters and structural error terms
        parameters_list, u_list = iv.estimate(X_list, Z_list, W[:self.MD + self.MS, :self.MD + self.MS], y_list)
        beta[parameters.eliminated_beta_index] = parameters_list[0].flat
        xi = u_list[0]
        if self.K3 == 0:
            omega = np.full((self.N, 0), np.nan, options.dtype)
        else:
            gamma[parameters.eliminated_gamma_index] = parameters_list[1].flat
            omega = u_list[1]

        # compute the objective value and replace it with its last value if computation failed
        with np.errstate(all='ignore'):
            mean_g = np.r_[compute_gmm_moments_mean(u_list, Z_list), micro]
            objective = mean_g.T @ W @ mean_g
            if scale_objective:
                objective *= self.N
        if not np.isfinite(np.squeeze(objective)):
            objective = progress.objective
            errors.append(exceptions.ObjectiveReversionError())

        # compute the gradient and replace any invalid elements with their last values (even if we concentrate out
        #   linear parameters, it turns out that one can use orthogonality conditions to show that treating the linear
        #   parameters as fixed is fine, so that we can treat xi and omega Jacobians as equal to delta and transformed
        #   marginal cost Jacobians when computing the gradient)
        gradient = np.full_like(progress.gradient, np.nan)
        if compute_gradient:
            with np.errstate(all='ignore'):
                mean_G = np.r_[compute_gmm_moments_jacobian_mean(jacobian_list, Z_list), micro_jacobian]
                gradient = 2 * (mean_G.T @ W @ mean_g)
                if scale_objective:
                    gradient *= self.N
            bad_gradient_index = ~np.isfinite(gradient)
            if np.any(bad_gradient_index):
                gradient[bad_gradient_index] = progress.gradient[bad_gradient_index]
                errors.append(exceptions.GradientReversionError(bad_gradient_index))

        # handle any errors
        if errors:
            if error_behavior == 'raise':
                raise exceptions.MultipleErrors(errors)
            if error_behavior == 'revert':
                objective *= error_punishment
            else:
                assert error_behavior == 'punish'
                objective = np.array(error_punishment)
                if compute_gradient:
                    gradient = np.zeros_like(progress.gradient)

        # select the delta that will be used in the next objective evaluation
        if delta_behavior == 'last':
            next_delta = delta
        else:
            assert delta_behavior == 'first'
            next_delta = progress.next_delta

        # optionally compute the Hessian with central finite differences
        hessian = np.full_like(progress.hessian, np.nan)
        if compute_hessian:
            def compute_perturbed_gradient(perturbed_theta: Array) -> Array:
                """Evaluate the gradient at a perturbed parameter vector."""
                perturbed_progress = self._compute_progress(
                    parameters, moments, iv, W, scale_objective, error_behavior, error_punishment, delta_behavior,
                    iteration, fp_type, shares_bounds, costs_bounds, finite_differences, perturbed_theta, progress,
                    compute_gradient=True, compute_hessian=False, compute_micro_covariances=False
                )
                return perturbed_progress.gradient

            # compute the Hessian, enforcing shape and symmetry
            hessian = compute_finite_differences(compute_perturbed_gradient, theta)
            hessian = np.c_[hessian + hessian.T] / 2

        # structure progress
        return Progress(
            self, parameters, moments, W, theta, objective, gradient, hessian, next_delta, delta, tilde_costs, micro,
            xi_jacobian, omega_jacobian, micro_jacobian, micro_covariances, micro_values, iv_delta, iv_tilde_costs, xi,
            omega, beta, gamma, iteration_stats, clipped_shares, clipped_costs, errors
        )

    def _compute_demand_contributions(
            self, parameters: Parameters, moments: EconomyMoments, iteration: Iteration, fp_type: str,
            shares_bounds: Bounds, sigma: Array, pi: Array, rho: Array, progress: 'InitialProgress',
            compute_jacobians: bool, compute_micro_covariances: bool) -> (
            Tuple[Array, Array, Array, Array, Array, Array, Array, Dict[Hashable, SolverStats], List[Error]]):
        """Compute delta and the Jacobian (holding beta fixed) of xi (equivalently, of delta) with respect to theta
        market-by-market. If there are any micro moments, compute them (taking the average across relevant markets)
        along with their Jacobian and covariances. Revert any problematic elements to their last values.
        """
        errors: List[Error] = []

        # initialize delta, micro moments, their Jacobians, micro moment covariances, micro moment values, indices of
        #   clipped shares, and fixed point statistics so that they can be filled
        delta = np.zeros((self.N, 1), options.dtype)
        micro = np.zeros((moments.MM, 1), options.dtype)
        xi_jacobian = np.zeros((self.N, parameters.P), options.dtype)
        micro_jacobian = np.zeros((moments.MM, parameters.P), options.dtype)
        micro_covariances = np.zeros((moments.MM, moments.MM), options.dtype)
        micro_values = np.full((self.T, moments.MM), np.nan, options.dtype)
        clipped_shares = np.zeros((self.N, 1), np.bool)
        iteration_stats: Dict[Hashable, SolverStats] = {}

        # when possible and when a gradient isn't needed, compute delta with a closed-form solution
        if self.K2 == 0 and moments.MM == 0 and (parameters.P == 0 or not compute_jacobians):
            delta = self._compute_logit_delta(rho)
        else:
            def market_factory(s: Hashable) -> Tuple[ProblemMarket, Array, Array, Iteration, str, Bounds, bool, bool]:
                """Build a market along with arguments used to compute delta, micro moment values, and Jacobians."""
                market_s = ProblemMarket(self, s, parameters, sigma, pi, rho, moments=moments)
                delta_s = progress.next_delta[self._product_market_indices[s]]
                last_delta_s = progress.delta[self._product_market_indices[s]]
                return (
                    market_s, delta_s, last_delta_s, iteration, fp_type, shares_bounds, compute_jacobians,
                    compute_micro_covariances
                )

            # compute delta, micro moment values, their Jacobians, and micro moment covariances market-by-market
            micro_jacobian_mapping: Dict[Hashable, Array] = {}
            micro_covariances_mapping: Dict[Hashable, Array] = {}
            generator = generate_items(self.unique_market_ids, market_factory, ProblemMarket.solve_demand)
            for t, generated_t in generator:
                (
                    delta_t, micro_values_t, xi_jacobian_t, micro_jacobian_t, micro_covariances_t, clipped_shares_t,
                    iteration_stats_t, errors_t
                ) = generated_t
                delta[self._product_market_indices[t]] = delta_t
                xi_jacobian[self._product_market_indices[t], :parameters.P] = xi_jacobian_t
                micro_values[self._market_indices[t], moments.market_indices[t]] = micro_values_t.flat
                micro_jacobian_mapping[t] = micro_jacobian_t
                micro_covariances_mapping[t] = micro_covariances_t
                clipped_shares[self._product_market_indices[t]] = clipped_shares_t
                iteration_stats[t] = iteration_stats_t
                errors.extend(errors_t)

            # aggregate micro moments, their Jacobian, and their covariances across all markets (this is done after
            #   market-by-market computation to preserve numerical stability with different market orderings)
            if moments.MM > 0:
                with np.errstate(all='ignore'):
                    for t in self.unique_market_ids:
                        indices = moments.market_indices[t]
                        weights = moments.market_weights[t]
                        if indices.size > 0:
                            differences = moments.market_values[t] - micro_values[self._market_indices[t], indices]
                            micro[indices] += weights[:, None] * differences[:, None]
                            micro_jacobian[indices, :parameters.P] -= weights[:, None] * micro_jacobian_mapping[t]
                            if compute_micro_covariances:
                                pairwise_indices = tuple(np.meshgrid(indices, indices))
                                pairwise_weights = weights[:, None] * weights[:, None].T
                                micro_covariances[pairwise_indices] += pairwise_weights * micro_covariances_mapping[t]

                    # enforce shape and symmetry of micro covariances
                    if compute_micro_covariances:
                        micro_covariances = np.c_[micro_covariances + micro_covariances.T] / 2

        # replace invalid elements in delta and the micro moment values with their last values
        bad_delta_index = ~np.isfinite(delta)
        bad_micro_index = ~np.isfinite(micro)
        if np.any(bad_delta_index):
            delta[bad_delta_index] = progress.delta[bad_delta_index]
            errors.append(exceptions.DeltaReversionError(bad_delta_index))
        if np.any(bad_micro_index):
            micro[bad_micro_index] = progress.micro[bad_micro_index]
            errors.append(exceptions.MicroMomentsReversionError(bad_micro_index))

        # replace invalid elements in the Jacobians with their last values
        if compute_jacobians:
            bad_xi_jacobian_index = ~np.isfinite(xi_jacobian)
            bad_micro_jacobian_index = ~np.isfinite(micro_jacobian)
            if np.any(bad_xi_jacobian_index):
                xi_jacobian[bad_xi_jacobian_index] = progress.xi_jacobian[bad_xi_jacobian_index]
                errors.append(exceptions.XiByThetaJacobianReversionError(bad_xi_jacobian_index))
            if np.any(bad_micro_jacobian_index):
                micro_jacobian[bad_micro_jacobian_index] = progress.micro_jacobian[bad_micro_jacobian_index]
                errors.append(exceptions.MicroMomentsByThetaJacobianReversionError(bad_micro_jacobian_index))

        return (
            delta, micro, xi_jacobian, micro_jacobian, micro_covariances, micro_values, clipped_shares, iteration_stats,
            errors
        )

    def _compute_supply_contributions(
            self, parameters: Parameters, costs_bounds: Bounds, sigma: Array, pi: Array, rho: Array, beta: Array,
            delta: Array, xi_jacobian: Array, progress: 'InitialProgress', compute_jacobian: bool) -> (
            Tuple[Array, Array, Array, List[Error]]):
        """Compute transformed marginal costs and the Jacobian (holding gamma fixed) of omega (equivalently, of
        transformed marginal costs) with respect to theta market-by-market. Revert any problematic elements to their
        last values.
        """
        errors: List[Error] = []

        # initialize transformed marginal costs, their Jacobian, and indices of clipped costs so that they can be filled
        tilde_costs = np.zeros((self.N, 1), options.dtype)
        omega_jacobian = np.zeros((self.N, parameters.P), options.dtype)
        clipped_costs = np.zeros((self.N, 1), np.bool)

        def market_factory(s: Hashable) -> Tuple[ProblemMarket, Array, Array, Bounds, bool]:
            """Build a market along with arguments used to compute transformed marginal costs and their Jacobian."""
            market_s = ProblemMarket(self, s, parameters, sigma, pi, rho, beta, delta=delta)
            last_tilde_costs_s = progress.tilde_costs[self._product_market_indices[s]]
            xi_jacobian_s = xi_jacobian[self._product_market_indices[s]]
            return market_s, last_tilde_costs_s, xi_jacobian_s, costs_bounds, compute_jacobian

        # compute transformed marginal costs and their Jacobian market-by-market
        generator = generate_items(self.unique_market_ids, market_factory, ProblemMarket.solve_supply)
        for t, (tilde_costs_t, omega_jacobian_t, clipped_costs_t, errors_t) in generator:
            tilde_costs[self._product_market_indices[t]] = tilde_costs_t
            omega_jacobian[self._product_market_indices[t], :parameters.P] = omega_jacobian_t
            clipped_costs[self._product_market_indices[t]] = clipped_costs_t
            errors.extend(errors_t)

        # replace invalid transformed marginal costs with their last values
        bad_tilde_costs_index = ~np.isfinite(tilde_costs)
        if np.any(bad_tilde_costs_index):
            tilde_costs[bad_tilde_costs_index] = progress.tilde_costs[bad_tilde_costs_index]
            errors.append(exceptions.CostsReversionError(bad_tilde_costs_index))

        # replace invalid elements in their Jacobian with their last values
        if compute_jacobian:
            bad_omega_jacobian_index = ~np.isfinite(omega_jacobian)
            if np.any(bad_omega_jacobian_index):
                omega_jacobian[bad_omega_jacobian_index] = progress.omega_jacobian[bad_omega_jacobian_index]
                errors.append(exceptions.OmegaByThetaJacobianReversionError(bad_omega_jacobian_index))

        return tilde_costs, omega_jacobian, clipped_costs, errors


class Problem(ProblemEconomy):
#class Problem(ProblemEconomy):
    r"""A BLP-type problem.

    
    """

    def __init__(
            self, cost_formulation: Formulation, instrument_formulation: Sequence[Formulation], 
            product_data: Mapping, demand_results: Mapping, model_formulations: Sequence[ModelFormulation] = None, markup_data: Optional[RecArray] = None) -> None:

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

        # validate and normalize instrument formulation
        #if not isinstance(instrument_formulation, Formulation):
        #    raise TypeError("instrument_formulation must be a single Formulation instance.")
        


        # validate and normalize model formulations
        
        #if isinstance(model_formulations, collections.abc.Sequence) and M > 1:
        #    model_formulations = list(model_formulations)
        #    
        #else:
        #    raise TypeError("model_formulations must be a sequence of at least two models.")
       




        # initialize the underlying economy with structured product and cost data
        products = Products(cost_formulation = cost_formulation, instrument_formulation = instrument_formulation, product_data = product_data)
        if markup_data is None:
            models = Models(model_formulations = model_formulations, product_data = product_data)
            markups = [None]*M
        else:
            models = None
            markups = markup_data
    

        super().__init__(
            cost_formulation, instrument_formulation, model_formulations, products, models, demand_results, markups
        )


        # absorb any cost fixed effects
        #if self._absorb_cost_ids is not None:
        #    output("Absorbing cost-side fixed effects ...")
        #    self.products.X1, X1_errors = self._absorb_demand_ids(self.products.X1)
        #    self._handle_errors(X1_errors)
        #    if add_exogenous:
        #        self.products.ZD, ZD_errors = self._absorb_demand_ids(self.products.ZD)
        #        self._handle_errors(ZD_errors)


        # detect any problems with the cost data
        #self._detect_collinearity()
        
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
                        f"Failed to compute the QR decomposition of [w,z"+srt(zz)+"] while checking for collinearity issues. "
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


class OptimalInstrumentProblem(ProblemEconomy):
    """A BLP problem updated with optimal excluded instruments.

    This class can be used exactly like :class:`Problem`.

    """

    def __init__(self, problem: ProblemEconomy, demand_instruments: Array, supply_instruments: Array) -> None:
        """Initialize the underlying economy with updated product data before absorbing fixed effects."""

        # keep track of long it takes to re-create the problem
        output("Re-creating the problem ...")
        start_time = time.time()

        # supplement the excluded demand-side instruments with exogenous characteristics in X1
        X1 = problem._compute_true_X1()
        ZD = demand_instruments
        for index, formulation in enumerate(problem._X1_formulations):
            if 'prices' not in formulation.names:
                ZD = np.c_[ZD, X1[:, [index]]]

        # supplement the excluded supply-side instruments with X3
        X3 = problem._compute_true_X3()
        ZS = np.c_[supply_instruments, X3]

        # update the products array
        updated_products = update_matrices(problem.products, {
            'ZD': (ZD, options.dtype),
            'ZS': (ZS, options.dtype)
        })

        # initialize the underlying economy with structured product and agent data
        super().__init__(
            problem.product_formulations, problem.agent_formulation, updated_products, problem.agents,
            distributions=problem.distributions, epsilon_scale=problem.epsilon_scale, costs_type=problem.costs_type
        )

        # absorb any demand-side fixed effects, which have already been absorbed into X1
        if self._absorb_demand_ids is not None:
            output("Absorbing demand-side fixed effects ...")
            self.products.ZD, ZD_errors = self._absorb_demand_ids(self.products.ZD)
            if ZD_errors:
                raise exceptions.MultipleErrors(ZD_errors)

        # absorb any supply-side fixed effects, which have already been absorbed into X3
        if self._absorb_supply_ids is not None:
            output("Absorbing supply-side fixed effects ...")
            self.products.ZS, ZS_errors = self._absorb_supply_ids(self.products.ZS)
            if ZS_errors:
                raise exceptions.MultipleErrors(ZS_errors)

        # detect any collinearity issues with the updated instruments
        self._detect_collinearity()

        # output information about the re-created problem
        output(f"Re-created the problem after {format_seconds(time.time() - start_time)}.")
        output("")
        output(self)


class ImportanceSamplingProblem(ProblemEconomy):
    """A BLP problem updated after importance sampling.

    This class can be used exactly like :class:`Problem`.

    """

    def __init__(self, problem: ProblemEconomy, sampled_agents: RecArray) -> None:
        """Initialize the underlying economy with updated agent data."""

        # keep track of long it takes to re-create the problem
        output("Re-creating the problem ...")
        start_time = time.time()

        # initialize the underlying economy with structured product and agent data
        super().__init__(
            problem.product_formulations, problem.agent_formulation, problem.products, sampled_agents,
            distributions=problem.distributions, epsilon_scale=problem.epsilon_scale, costs_type=problem.costs_type
        )

        # output information about the re-created problem
        output(f"Re-created the problem after {format_seconds(time.time() - start_time)}.")
        output("")
        output(self)


class InitialProgress(object):
    """Structured information about initial estimation progress."""

    problem: ProblemEconomy
    #parameters: Parameters
    #moments: EconomyMoments
    #W: Array
    #theta: Array
    #objective: Array
    #gradient: Array
    #hessian: Array
    #next_delta: Array
    #delta: Array
    #tilde_costs: Array
    #micro: Array
    #xi_jacobian: Array
    #omega_jacobian: Array
    #micro_jacobian: Array
    

    def __init__(
            self, problem: ProblemEconomy 
            #parameters: Parameters, moments: EconomyMoments, W: Array, theta: Array,
            #objective: Array, gradient: Array, hessian: Array, next_delta: Array, delta: Array, tilde_costs: Array,
            #micro: Array, xi_jacobian: Array, omega_jacobian: Array, micro_jacobian: Array
            ) -> None:
        """Store initial progress information, computing the projected gradient and the reduced Hessian."""
        self.problem = problem
        #self.markups = markups
        #self.parameters = parameters
        #self.moments = moments
        #self.W = W
        #self.theta = theta
        #self.objective = objective
        #self.gradient = gradient
        #self.hessian = hessian
        #self.next_delta = next_delta
        #self.delta = delta
        #self.tilde_costs = tilde_costs
        #self.micro = micro
        #self.xi_jacobian = xi_jacobian
        #self.omega_jacobian = omega_jacobian
        #self.micro_jacobian = micro_jacobian


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
    #micro_covariances: Array
    #micro_values: Array
    #xi: Array
    #omega: Array
    #beta: Array
    #gamma: Array
    #iteration_stats: Dict[Hashable, SolverStats]
    #clipped_shares: Array
    #clipped_costs: Array
    #errors: List[Error]
    #projected_gradient: Array
    #reduced_hessian: Array
    #projected_gradient_norm: Array

    def __init__(
            self, problem: ProblemEconomy, markups: Array, markups_downstream: Array, markups_upstream: Array, mc: Array, taus: Array, g: Array, Q: Array, RV_num: Array, RV_denom: Array, TRV: Array, F: Array, MCS_pvalues: Array
            #parameters: Parameters, moments: EconomyMoments, W: Array, theta: Array,
            #objective: Array, gradient: Array, hessian: Array, next_delta: Array, delta: Array, tilde_costs: Array,
            #micro: Array, xi_jacobian: Array, omega_jacobian: Array, micro_jacobian: Array, micro_covariances: Array,
            #micro_values: Array, iv_delta: Array, iv_tilde_costs: Array, xi: Array, omega: Array, beta: Array,
            #gamma: Array, iteration_stats: Dict[Hashable, SolverStats], clipped_shares: Array, clipped_costs: Array,
            #errors: List[Error]
            ) -> None:
        """Store progress information, compute the projected gradient and its norm, and compute the reduced Hessian."""
        super().__init__(
            problem
            #, parameters, moments, W, theta, objective, gradient, hessian, next_delta, delta, tilde_costs, micro,
            #xi_jacobian, omega_jacobian, micro_jacobian
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
        #self.micro_covariances = micro_covariances
        #self.micro_values = micro_values
        #self.iv_delta = iv_delta
        #self.iv_tilde_costs = iv_tilde_costs
        #self.xi = xi
        #self.omega = omega
        #self.beta = beta
        #self.gamma = gamma
        #self.iteration_stats = iteration_stats or {}
        #self.clipped_shares = clipped_shares
        #self.clipped_costs = clipped_costs
        #self.errors = errors or []

        # compute the projected gradient and the reduced Hessian
        #self.projected_gradient = self.gradient.copy()
        #self.reduced_hessian = self.hessian.copy()
        #for p, (lb, ub) in enumerate(self.parameters.compress_bounds()):
        #    if not lb < theta[p] < ub:
        #        self.reduced_hessian[p] = self.reduced_hessian[:, p] = 0
        #        with np.errstate(invalid='ignore'):
        #            if theta[p] <= lb:
        #                self.projected_gradient[p] = min(0, self.gradient[p])
        #            elif theta[p] >= ub:
        #                self.projected_gradient[p] = max(0, self.gradient[p])

        # compute the norm of the projected gradient
        #self.projected_gradient_norm = np.array(np.nan, options.dtype)
        #if gradient.size > 0:
        #    with np.errstate(invalid='ignore'):
        #        self.projected_gradient_norm = np.abs(self.projected_gradient).max()

    def format(
            self, optimization: Optimization, shares_bounds: Bounds, costs_bounds: Bounds, step: int, iterations: int,
            evaluations: int, smallest_objective: Array) -> str:
        """Format a universal display of optimization progress as a string. The first iteration will include the
        progress table header. If there are any errors, information about them will be formatted as well, regardless of
        whether or not a universal display is to be used. The smallest_objective is the smallest objective value
        encountered so far during optimization.
        """
        lines: List[str] = []

        # include information about any errors
        if self.errors:
            preamble = (
                "At least one error was encountered. As long as the optimization routine does not get stuck at values "
                "of theta that give rise to errors, this is not necessarily a problem. If the errors persist or seem "
                "to be impacting the optimization results, consider setting an error punishment or following any of "
                "the other suggestions below:"
            )
            lines.extend(["", preamble, str(exceptions.MultipleErrors(self.errors)), ""])

        # only output errors if the solver's display is being used
        if not optimization._universal_display:
            return "\n".join(lines)

        # construct the leftmost part of the table that always shows up
        header = [
            ("GMM", "Step"), ("Optimization", "Iterations"), ("Objective", "Evaluations"),
            ("Fixed Point", "Iterations"), ("Contraction", "Evaluations")
        ]
        values = [
            str(step),
            str(iterations),
            str(evaluations),
            str(sum(s.iterations for s in self.iteration_stats.values())),
            str(sum(s.evaluations for s in self.iteration_stats.values()))
        ]

        # add a count of any clipped shares or marginal costs
        if np.isfinite(shares_bounds).any():
            header.append(("Clipped", "Shares"))
            values.append(str(self.clipped_shares.sum()))
        if np.isfinite(costs_bounds).any():
            header.append(("Clipped", "Costs"))
            values.append(str(self.clipped_costs.sum()))

        # add information about the objective
        header.extend([("Objective", "Value"), ("Objective", "Improvement")])
        values.append(format_number(self.objective))
        improvement = smallest_objective - self.objective
        if np.isfinite(improvement) and improvement > 0:
            values.append(format_number(smallest_objective - self.objective))
        else:
            values.append(" " * len(format_number(improvement)))

        # add information about the gradient
        if optimization._compute_gradient:
            header.append(("Projected", "Gradient Norm") if self.parameters.any_bounds else ("Gradient", "Norm"))
            values.append(format_number(self.projected_gradient_norm))

        # add information about theta
        header.append(("", "Theta"))
        values.append(", ".join(format_number(x) for x in self.theta))

        # add information about micro moments
        if self.moments.MM > 0:
            header.append(("Micro", "Moments"))
            values.append(", ".join(format_number(x) for x in self.micro))

        # format the table
        lines.append(format_table(header, values, include_border=False, include_header=evaluations == 1))
        return "\n".join(lines)
