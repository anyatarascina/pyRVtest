pyRV_path = '/Users/christophersullivan/Dropbox/Economics/Research/RiversVuong_Endogeneity/pyRV_folder'

import sys
sys.path.append(pyRV_path)

import numpy as np
import pandas as pd
import pyblp
import pyRV_multipleinstruments as pyRV

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


product_data["clustering_ids"] = product_data.market_ids
product_data["vi_ids"] = product_data.brand_ids


# RV TESTING
testing_problem = pyRV.Problem(
    cost_formulation = (
            pyRV.Formulation('1 + demand_instruments2 + demand_instruments3', absorb = 'C(product_ids)' )
        ),
    instrument_formulation = (
            pyRV.Formulation('0 + demand_instruments0 + demand_instruments1')
        ), 
    model_formulations = (
            pyRV.ModelFormulation(model_downstream='monopoly', ownership_downstream='firm_ids'),
            pyRV.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids'),
            pyRV.ModelFormulation(model_downstream='monopoly', ownership_downstream='firm_ids', model_upstream='bertrand',  ownership_upstream='firm_ids'),
            pyRV.ModelFormulation(model_downstream='monopoly', ownership_downstream='firm_ids', model_upstream='monopoly',  ownership_upstream='firm_ids'),
            pyRV.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids', model_upstream='bertrand',  ownership_upstream='firm_ids', vertical_integration='vi_ids'),
            #pyRV.ModelFormulation(model_downstream='bertrand', ownership_downstream='retailer_id', kappa_specification_downstream = None, model_upstream='monopoly'),
       ),       
    product_data = product_data,
    demand_results = pyblp_results
        )


testing_results = testing_problem.solve(
    method = 'both',  #method: MCS, pairwise, both (default)
    demand_adjustment = 'no',  #demand adjustment: yes (default), no
    #se_type = 'clustered' #clustering: 'robust', 'unadjusted' (default), or 'clustered'
    se_type = 'unadjusted'
    )



testing_results_none = testing_problem.solve(
    method = 'both',  #method: MCS, pairwise, both (default)
    demand_adjustment = 'no',  #demand adjustment: yes (default), no
    #se_type = 'clustered' #clustering: 'robust', 'unadjusted' (default), or 'clustered'
    se_type = 'unadjusted'
    )


testing_results_da = testing_problem.solve(
    method = 'both',  #method: MCS, pairwise, both (default)
    demand_adjustment = 'yes',  #demand adjustment: yes (default), no
    #se_type = 'clustered' #clustering: 'robust', 'unadjusted' (default), or 'clustered'
    se_type = 'unadjusted'
    )


testing_results_c = testing_problem.solve(
    method = 'both',  #method: MCS, pairwise, both (default)
    demand_adjustment = 'no',  #demand adjustment: yes (default), no
    #se_type = 'clustered' #clustering: 'robust', 'unadjusted' (default), or 'clustered'
    se_type = 'clustered'
    )


testing_results_both = testing_problem.solve(
    method = 'both',  #method: MCS, pairwise, both (default)
    demand_adjustment = 'yes',  #demand adjustment: yes (default), no
    #se_type = 'clustered' #clustering: 'robust', 'unadjusted' (default), or 'clustered'
    se_type = 'clustered'
    )


testing_results_none
testing_results_da
testing_results_c
testing_results_both

testing_results.Fstat
testing_results.RVstats
testing_results.MCS
testing_results.markups
testing_results.predicted_markups
testing_results.cost_coefficients


#Saving pyblp results -- maybe need pickling for preserving class functionality when doing demand adjustment
#demand_results = "/Users/christophersullivan/Dropbox/Economics/Research/RiversVuong_Endogeneity/pyRV_code/beta/pyblp_results.json"
#save_pyblp(pyblp_results,out_file)
#   demand_results = pd.read_json(out_file, typ='series', orient='records')

pyRV_path = '/Users/christophersullivan/Dropbox/Economics/Research/RiversVuong_Endogeneity/pyRV_folder'

import sys
sys.path.append(pyRV_path)

import numpy as np
import pandas as pd
import pyblp
import pyRV

product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)

x = pyRV.ModelFormulation(model_downstream='bertrand', ownership_downstream='retailer_id', model_upstream = 'cournot', ownership_upstream='retailer_id')
x._build_matrix(product_data)





pyRV_path = '/Users/christophersullivan/Dropbox/Economics/Research/RiversVuong_Endogeneity/pyRV_folder'

import sys
sys.path.append(pyRV_path)

import numpy as np
import pandas as pd
import pyblp
import pyRV

product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)

from pyRV.primitives import Products
from pyRV.primitives import Models

y = Products(cost_formulation = (
            pyRV.Formulation('1 + sugar + demand_instruments2', absorb = 'C(product_ids)' )
        ),
    instrument_formulation = (
            pyRV.Formulation('0 + demand_instruments0 + demand_instruments1')
        ),        
    product_data = product_data
)

y1 = Models(model_formulations = (
            pyRV.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids'),
            pyRV.ModelFormulation(model_downstream='bertrand', ownership_downstream='brand_ids'),
            #pyRV.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_id', model_upstream='bertrand',  ownwership_upstream='product_id'),
            #pyRV.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_id', model_upstream='bertrand',  ownwership_upstream='product_id'),
            #pyRV.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_id', kappa_specification_downstream = None, model_upstream='monopoly'),
       ),product_data = product_data)

a = pyblp.Formulation('1 + sugar + demand_instruments2', absorb = 'C(product_ids)' )