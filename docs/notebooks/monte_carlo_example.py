# Monte Carlo Example

import pyblp
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
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

# True DGP: Bertrand competition with cost formulation 1 + z + shares (non-constant marginal cost).
# gamma = [1, 4, 0.5]: intercept, z coefficient, shares coefficient.
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

# Test instruments (set 0): BLP instruments
test_instruments = pyblp.build_blp_instruments(
    pyblp.Formulation('0+z+x1+x2'),
    data
)
for i, column in enumerate(test_instruments.T):
    data[f'test_instruments{i}'] = column
instr_form = '+'.join(x for x in data.columns if x.startswith('test_instruments'))

# Test instruments (set 1): local differentiation instruments
test_instruments2 = pyblp.build_differentiation_instruments(
    pyblp.Formulation('0+z+x1+x2'),
    data, version='local'
)
for i, column in enumerate(test_instruments2.T):
    data[f'test_instruments2_{i}'] = column
instr_form2 = '+'.join(x for x in data.columns if x.startswith('test_instruments2_'))

data['clustering_ids'] = data['market_ids']

# Mix flag: firms 0, 1, 2 compete in prices (Bertrand); firms 3, 4 compete in quantities (Cournot)
data['mix_flag'] = (data['firm_ids'] < 3)

# Cost scaling (lambda=0.5) for rule-of-thumb and scaled-cost models
data['cost_scaling_col'] = 0.5

# Vertical integration indicator: products of firm 0 are vertically integrated
data['vi_id'] = (data['firm_ids'] == 0).astype(int)

# Partial collusion: firms put weight 0.5 on own-portfolio profits (intermediate between Bertrand and monopoly)
def kappa_partial_collusion(f, g):
    return 0.5 if f == g else 0.0

# Non-profit conduct: welfare weight lambda=0.7 implies kappa = 1/lambda for own-firm product pairs
def kappa_nonprofit(f, g):
    return 1 / 0.7 if f == g else 0.0

# %% Demand estimation
problem = pyblp.Problem(
    (X1, X2),
    product_data=data, integration=integration
)
pyblp_results = problem.solve(sigma=0.5, method='1s')
print(pyblp_results)

# %% Full library of model formulations
model_formulations = (
    # --- Standard models ---
    pyRVtest.ModelFormulation(model_downstream='bertrand',          ownership_downstream='firm_ids'),
    pyRVtest.ModelFormulation(model_downstream='cournot',           ownership_downstream='firm_ids'),
    pyRVtest.ModelFormulation(model_downstream='monopoly'),
    pyRVtest.ModelFormulation(model_downstream='perfect_competition'),

    # --- Profit-weight models (Bertrand/Cournot with partial collusion, kappa=0.5) ---
    pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids',
                              kappa_specification_downstream=kappa_partial_collusion),
    pyRVtest.ModelFormulation(model_downstream='cournot',  ownership_downstream='firm_ids',
                              kappa_specification_downstream=kappa_partial_collusion),

    # --- Non-profit conduct (Bertrand, welfare weight lambda=0.7, kappa=1/0.7) ---
    pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids',
                              kappa_specification_downstream=kappa_nonprofit),

    # --- Rule-of-thumb (markups = lambda * cost, lambda=0.5) ---
    pyRVtest.ModelFormulation(model_downstream='perfect_competition', cost_scaling='cost_scaling_col'),

    # --- Bertrand with scaled costs (lambda=0.5) ---
    pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids',
                              cost_scaling='cost_scaling_col'),

    # --- Vertical: linear pricing (Bertrand downstream + Bertrand upstream) ---
    pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids',
                              model_upstream='bertrand',   ownership_upstream='firm_ids'),

    # --- Vertical with partial VI (firm 0 products are vertically integrated) ---
    pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids',
                              model_upstream='bertrand',   ownership_upstream='firm_ids',
                              vertical_integration='vi_id'),

    # --- Mixed Cournot-Bertrand ---
    pyRVtest.ModelFormulation(model_downstream='mix_cournot_bertrand', ownership_downstream='firm_ids',
                              mix_flag='mix_flag'),

    # --- Custom model: single-product Bertrand (each product ignores cross-portfolio effects) ---
    # markup_j = -s_j / (ds_j/dp_j), using only own-price derivatives; differs from multi-product Bertrand
    pyRVtest.ModelFormulation(model_downstream='other', ownership_downstream='firm_ids',
                              custom_model_specification={
                                  'single_product_bertrand': lambda O, D, s: -s / np.diag(D).reshape(-1, 1)
                              }),
)

# %% Standalone markup construction and comparison vs. Bertrand
markups, markups_down, markups_up = pyRVtest.build_markups(model_formulations, data, pyblp_results)

model_names = [
    'Bertrand',
    'Cournot',
    'Monopoly',
    'Perfect competition',
    'Bertrand (kappa=0.5)',
    'Cournot (kappa=0.5)',
    'Non-profit Bertrand (lambda=0.7)',
    'Rule-of-thumb (lambda=0.5)',
    'Bertrand scaled costs (lambda=0.5)',
    'Vertical Bertrand+Bertrand',
    'Vertical Bertrand+Bertrand with VI',
    'Mixed Cournot-Bertrand',
    'Single-product Bertrand (custom)',
]

bertrand_markups = markups[0].flatten()
rows = []
for name, mu in zip(model_names, markups):
    diff = mu.flatten() - bertrand_markups
    rows.append({
        'Model': name,
        'Mean': diff.mean(),
        'Median': np.median(diff),
        'Std': diff.std(),
        'Min': diff.min(),
        'Max': diff.max(),
    })

markup_table = pd.DataFrame(rows).set_index('Model')
print("\nMarkup differences relative to Bertrand (model - Bertrand):")
print(markup_table.to_string(float_format='{:.4f}'.format))

# %% Conduct test (full model library, two instrument sets, endogenous cost component)
# For each instrument set independently:
#   1. A per-model 2SLS uses that set's instruments to estimate the coefficient on shares
#      and correct marginal costs.
#   2. That set's instruments are residualized on exogenous cost-shifters and its own
#      first-stage fitted values of shares, so the instruments remain valid.

testing_problem = pyRVtest.Problem(
    cost_formulation=pyRVtest.Formulation('1+z+shares'),
    instrument_formulation=[
        pyRVtest.Formulation('0+x1+x2+' + instr_form),
        pyRVtest.Formulation('0+x1+x2+' + instr_form2),
    ],
    model_formulations=model_formulations,
    product_data=data,
    demand_results=pyblp_results,
    endogenous_cost_component='shares'
)

testing_results = testing_problem.solve(
    demand_adjustment=True,
    clustering_adjustment=True,
)
print(testing_results)

results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'monte_carlo_results.txt')
with open(results_path, 'w') as f:
    f.write(str(testing_results))
print(f"\nResults written to {results_path}")
