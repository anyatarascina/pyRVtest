pyRV_path = '/Users/christophersullivan/Dropbox/Economics/Research/RiversVuong_Endogeneity/pyRV_folder'

import sys
sys.path.append(pyRV_path)

import numpy as np
import pandas as pd
import pyblp
import pyRVtest as pyRV

product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)

#Demand Estimation
pyblp_problem = pyblp.Problem(
    product_formulations = (
        pyblp.Formulation('0 + prices ', absorb = 'C(product_ids)' ),
        pyblp.Formulation('1 + prices + sugar + mushy'),
        ),
        agent_formulation = pyblp.Formulation('0 + income + income_squared + age + child'), 
        product_data = product_data,
        agent_data = pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)
        )


pyblp_results = pyblp_problem.solve(
  #sigma = ([[0.3302, 0.9,0,0],[1.4, 2.4526, 0,.4],[0.01,0, 0.0163, 0],[0,0,0, 0.2441]]),
  sigma = np.diag([0.3302, 2.4526, 0.0163, 0.2441]), 
  pi = [[5.4819,0, 0.2037 ,0 ],[15.8935,-1.2000, 0 ,2.6342 ],[-0.2506,0, 0.0511 ,0 ],[1.2650,0, -0.8091 ,0 ]  ],
  method = '1s', 
  optimization = pyblp.Optimization('bfgs',{'gtol':1e-5})  
  )


testing_problem_new = pyRV.Problem(
    cost_formulation = (
        pyRV.Formulation('1 + sugar', absorb = 'C(firm_ids)' )
        ),
    instrument_formulation = (
        pyRV.Formulation('0 + demand_instruments0 + demand_instruments1'),
        pyRV.Formulation('0 + demand_instruments2 + demand_instruments3 + demand_instruments4'),
        pyRV.Formulation('0 + demand_instruments5')
        ), 
    model_formulations = (
        pyRV.ModelFormulation(model_downstream='monopoly', ownership_downstream='firm_ids'),
        pyRV.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids'),
        pyRV.ModelFormulation(model_downstream='cournot', ownership_downstream='firm_ids'),
        pyRV.ModelFormulation(model_downstream='monopoly', ownership_downstream='firm_ids', model_upstream='bertrand',  ownership_upstream='firm_ids'),
        pyRV.ModelFormulation(model_downstream='monopoly', ownership_downstream='firm_ids', model_upstream='monopoly',  ownership_upstream='firm_ids')
        ),       
    product_data = product_data,
    demand_results = pyblp_results
    )

testing_results_new = testing_problem_new.solve(
    demand_adjustment = 'yes', 
    se_type = 'clustered'
    )

assert testing_results_new.F[0][0]-0.04<1e-3
