# Monte Carlo Example

import pyblp
import sys
import numpy as np
sys.path.append('/home/md/Dropbox/Projects/pyRVtest')
import pyRVtest
import pandas as pd

pyblp.options.digits = 2
pyblp.options.verbose = False

# %% DGP

rng = np.random.default_rng(seed=0)

# Market structure
F = 5           # number of firms (constant across markets)
J_min = 20      # minimum products per market
J_max = 40      # maximum products per market
target_obs = 10000
T = target_obs // ((J_min + J_max) // 2)
J_per_market = rng.integers(J_min, J_max + 1, size=T)
market_ids = np.repeat(np.arange(T), J_per_market)
firm_ids = np.concatenate([np.arange(J_t) % F for J_t in J_per_market])
id_data = pd.DataFrame({'market_ids': market_ids, 'firm_ids': firm_ids})
integration = pyblp.Integration('product', 9)
X1 = pyblp.Formulation('1 + prices + x1 + x2')
X2 = pyblp.Formulation('0 + x1')
X3 = pyblp.Formulation('1  + z + shares')

# Simulation
simulation = pyblp.Simulation(
    product_formulations=(X1, X2, X3),
    beta=[0, -2, 1, 1],
    sigma=1,
    xi_variance=0.2,
    omega_variance=0.2,
    correlation=0,
    gamma=[1, 4, 0.5],
    product_data=id_data,
    integration=integration,
    seed=0
)
simulation_results = simulation.replace_endogenous(constant_costs=False)
data = pd.DataFrame(pyblp.data_to_dict(simulation_results.product_data))

# Demand instruments
local_instruments = pyblp.build_differentiation_instruments(
    pyblp.Formulation('0+ x1+x2+z'),
    data, version='local'
)
for i, column in enumerate(local_instruments.T):
    data[f'demand_instruments{i}'] = column
data[f'demand_instruments{i+1}'] = data['z']

# Test instruments
test_instruments = pyblp.build_blp_instruments(
    pyblp.Formulation('0+ z+x1+x2'),
    data
)
for i, column in enumerate(test_instruments.T):
    data[f'test_instruments{i}'] = column
instr_form='+'.join(x for x in data.columns if x.startswith('test_instruments'))
    
data['clustering_ids'] = data['market_ids']

# Mix flag: firms 0, 1, 2 compete in prices (Bertrand); firms 3, 4 compete in quantities (Cournot)
data['mix_flag'] = (data['firm_ids'] < 3)

# %% Demand estimation
problem = pyblp.Problem(
    (X1, X2),
    product_data=data, integration=integration
)
pyblp_results = problem.solve(sigma=0.5, method='1s')
print(pyblp_results)


# %% Contruct markups
models=(
    pyRVtest.ModelFormulation(model_downstream='bertrand',           ownership_downstream='firm_ids'),
    pyRVtest.ModelFormulation(model_downstream='cournot',            ownership_downstream='firm_ids'),
    pyRVtest.ModelFormulation(model_downstream='monopoly',           ownership_downstream='firm_ids'),
    pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
    pyRVtest.ModelFormulation(model_downstream='monopoly',           ownership_downstream='firm_ids',
                              model_upstream='bertrand',             ownership_upstream='firm_ids'),
    pyRVtest.ModelFormulation(model_downstream='mix_cournot_bertrand', ownership_downstream='firm_ids',
                              mix_flag='mix_flag')
    )
Markups, markups_down, markups_up = pyRVtest.build_markups(models, data, pyblp_results)

# %% Conduct test with economies of scale correction
# Setting endogenous_cost_component='shares' tells pyRVtest to:
#   1. Run a per-model 2SLS to estimate the coefficient on shares and correct marginal costs.
#   2. Residualize the test instruments jointly on exogenous cost-shifters and the first-stage
#      fitted values of shares, so the instruments are valid despite shares entering the cost.

testing_problem = pyRVtest.Problem(
    cost_formulation=(
        pyRVtest.Formulation('1+z+shares')
    ),
    instrument_formulation=(
        pyRVtest.Formulation('0+x1+x2+'+instr_form)
    ),
    model_formulations=(
        pyRVtest.ModelFormulation(model_downstream='bertrand',           ownership_downstream='firm_ids'),
        pyRVtest.ModelFormulation(model_downstream='cournot',            ownership_downstream='firm_ids'),
        pyRVtest.ModelFormulation(model_downstream='monopoly',           ownership_downstream='firm_ids'),
        pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
        pyRVtest.ModelFormulation(model_downstream='monopoly',           ownership_downstream='firm_ids',
                                  model_upstream='bertrand',             ownership_upstream='firm_ids'),
        pyRVtest.ModelFormulation(model_downstream='mix_cournot_bertrand', ownership_downstream='firm_ids',
                                  mix_flag='mix_flag'),
    ),
    product_data=data,
    demand_results=pyblp_results,
    endogenous_cost_component='shares'
)

testing_results = testing_problem.solve(
    demand_adjustment=True,
    clustering_adjustment=True,
)

print(testing_results)
    

    
    
    