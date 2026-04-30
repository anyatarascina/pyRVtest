import pyblp
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import pyRVtest
from pyRVtest.instruments import blp_instruments, differentiation_ivs
import pandas as pd

pyblp.options.digits = 2
pyblp.options.verbose = False

# %% DGP

rng = np.random.default_rng(seed=0)

# Market structure
F = 5           # number of firms (constant across markets)
J_min = 20      # minimum products per market
J_max = 40      # maximum products per market
target_obs = 3000
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

# %% Instruments (using pyRVtest.instruments helpers instead of raw pyblp)

# Demand instruments: local differentiation IVs on x1, x2, z
data['demand_instruments0'] = data['z']
count=0
for char in ['x1', 'x2', 'z']:
    divs = differentiation_ivs(data, char)
    count=count+1
    data[f'demand_instruments{count}'] = divs['sum_squared_diff_rival']
    count=count+1
    data[f'demand_instruments{count}'] = divs['sum_squared_diff_same_firm']


# Test instruments (set 0): BLP instruments on z, x1, x2
blp_ivs = blp_instruments(data, columns=['z', 'x1', 'x2'])
for key, arr in blp_ivs.items():
    data[f'test_iv0_{key}'] = arr

# Test instruments (set 1): local differentiation instruments on z, x1, x2
for char in ['z', 'x1', 'x2']:
    divs = differentiation_ivs(data, char)
    data[f'test_iv1_{char}_rival'] = divs['sum_squared_diff_rival']
    data[f'test_iv1_{char}_same'] = divs['sum_squared_diff_same_firm']

instr_form0 = '+'.join(c for c in data.columns if c.startswith('test_iv0_'))
instr_form1 = '+'.join(c for c in data.columns if c.startswith('test_iv1_'))

data['clustering_ids'] = data['market_ids']

problem = pyblp.Problem(
    (X1, X2),
    product_data=data, integration=integration
)
pyblp_results = problem.solve(sigma=0.5, method='1s')

# %% Model library — v0.4 class-based API

# Core 13-model set (analogous to the original v0.3 library): used for the
# main RV conduct test where numerical stability of the MCS is well-established.
model_core = [
    # --- Standard models ---
    pyRVtest.Bertrand(ownership='firm_ids'),
    pyRVtest.Cournot(ownership='firm_ids')]
    
testing_problem = pyRVtest.Problem(
    cost_formulation=pyRVtest.Formulation('1+z+shares'),
    instrument_formulation=[
        pyRVtest.Formulation('0+x1+x2+' + instr_form0),
        pyRVtest.Formulation('0+x1+x2+' + instr_form1),
    ],
    models=model_core,
    product_data=data,
    demand_results=pyblp_results,
    endogenous_cost_component='shares'
)

_t0 = time.perf_counter()
testing_results = testing_problem.solve(
    demand_adjustment=True,
    clustering_adjustment=True,
)
_t1 = time.perf_counter()
print(f"\n[speed_test] Problem.solve(demand_adjustment=True) wall-time: {_t1 - _t0:.2f}s\n")
