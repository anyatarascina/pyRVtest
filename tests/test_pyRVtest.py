"""Testing the package."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyblp
import pyRVtest

# set project directory
PROJECT_DIR = Path(__file__).absolute().parents[1]
sys.path.append(PROJECT_DIR)


def test_nevo_method1():

    # load product data
    product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)

    # use pyblp package for demand estimation
    pyblp_problem = pyblp.Problem(
        product_formulations=(
            pyblp.Formulation('0 + prices ', absorb='C(product_ids)'),
            pyblp.Formulation('1 + prices + sugar + mushy'),
            ),
        agent_formulation=pyblp.Formulation('0 + income + income_squared + age + child'),
        product_data=product_data,
        agent_data=pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)
        )

    # return pyblp results
    pyblp_results = pyblp_problem.solve(
      sigma=np.diag([0.3302, 2.4526, 0.0163, 0.2441]),
      pi=[
          [5.4819,   0.0000,  0.2037, 0.0000],
          [15.8935, -1.2000,  0.0000, 2.6342],
          [-0.2506,  0.0000,  0.0511, 0.0000],
          [1.2650,   0.0000, -0.8091, 0.0000]
      ],
      method='1s',
      optimization=pyblp.Optimization('bfgs', {'gtol': 1e-5})
      )

    # update product data with market ids
    product_data["clustering_ids"] = product_data.market_ids

    # additional variables for tax testing
    product_data["unit_tax"] = .5 * np.ones((product_data.shape[0], 1))
    product_data["advalorem_tax"] = .5 * np.ones((product_data.shape[0], 1))
    product_data["lambda"] = 1 * np.ones((product_data.shape[0], 1))

    # new testing problem
    # TODO: allow separate custom formulas for upstream and downstream
    testing_problem_new = pyRVtest.Problem(
        cost_formulation=(
            pyRVtest.Formulation('1 + sugar', absorb='C(firm_ids)')
            ),
        instrument_formulation=(
            pyRVtest.Formulation('0 + demand_instruments0 + demand_instruments1'),
            pyRVtest.Formulation('0 + demand_instruments2 + demand_instruments3 + demand_instruments4'),
            pyRVtest.Formulation('0 + demand_instruments5')
            ),
        model_formulations=(
            pyRVtest.ModelFormulation(model_downstream='monopoly', ownership_downstream='firm_ids'),
            pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids'),
            pyRVtest.ModelFormulation(model_downstream='cournot', ownership_downstream='firm_ids'),
            pyRVtest.ModelFormulation(
                model_downstream='monopoly',
                ownership_downstream='firm_ids',
                model_upstream='bertrand',
                ownership_upstream='firm_ids'
            ),
            pyRVtest.ModelFormulation(
                model_downstream='monopoly',
                ownership_downstream='firm_ids',
                model_upstream='monopoly',
                ownership_upstream='firm_ids')
            ),
        product_data=product_data,
        demand_results=pyblp_results
        )
    testing_results_new = testing_problem_new.solve(
        demand_adjustment=True,
        se_type='clustered'
        )

    testing_problem = pyRVtest.Problem(
        cost_formulation=(
            pyRVtest.Formulation('0 + sugar', absorb='C(firm_ids)')
        ),
        instrument_formulation=(
            pyRVtest.Formulation('0 + demand_instruments0 + demand_instruments1')
        ),
        model_formulations=(
            pyRVtest.ModelFormulation(
                model_downstream='bertrand',
                ownership_downstream='firm_ids',
                cost_scaling='lambda',
                unit_tax='unit_tax',
                advalorem_tax='advalorem_tax',
                advalorem_payer='consumer'),
            pyRVtest.ModelFormulation(
                model_downstream='bertrand',
                ownership_downstream='firm_ids'
            ),
            pyRVtest.ModelFormulation(
                model_downstream='perfect_competition',
                ownership_downstream='firm_ids'
            ),
        ),
        product_data=product_data,
        demand_results=pyblp_results
    )

    # output taxation model results
    testing_results = testing_problem.solve(
        demand_adjustment=False,
        se_type='unadjusted'
    )


if __name__ == '__main__':
    test_nevo_method1()
