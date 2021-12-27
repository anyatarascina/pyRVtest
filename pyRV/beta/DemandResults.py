class DemandResults(object):
import json
import copy
import os


out_file = "/Users/christophersullivan/Dropbox/Economics/Research/RiversVuong_Endogeneity/pyRV_code/beta/pyblp_results.json"
a = pd.read_json(out_file, typ='series', orient='records')

def save_pyblp(pyblp_results,out_file)
    demand_results = {**pyblp_results.__dict__}
    demand_results['elasticities'] = pyblp_results.compute_elasticities()

    to_drop = ['problem','_parameters','_moments','_iteration','_fp_type','_errors','tilde_costs','clipped_shares',
                'clipped_costs','_scaled_objective','_shares_bounds','_costs_bounds','_se_type',
                'reduced_hessian_eigenvalues','step','total_time','cumulative_total_time','optimization_time', 
                'cumulative_optimization_time','converged','cumulative_converged', 'optimization_iterations', 
                'cumulative_optimization_iterations','objective_evaluations','cumulative_objective_evaluations',
                'fp_converged','cumulative_fp_converged','fp_iterations','cumulative_fp_iterations','contraction_evaluations', 
                'cumulative_contraction_evaluations','last_results','objective','gradient','projected_gradient','projected_gradient_norm',
                'hessian','reduced_hessian','omega','omega_by_theta_jacobian','micro_by_theta_jacobian','micro','micro_values','gamma_labels',
                'gamma_bounds','sigma_bounds','pi_bounds','rho_bounds','beta_bounds','gamma_se','gamma'
                ]
    for kk in to_drop:
        del demand_results[kk]


    pd_demand_results = pd.Series(demand_results)
    out = pd.Series.to_json(pd_demand_results)

    with open(out_file,"w") as f:
      f.write(out)




