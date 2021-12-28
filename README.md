# pyRV

This code was developed by Christopher Sullivan and Lorenzo Magnolfi.  It largely adapts the pyblp source code (copyright Jeff Gortmaker and Christopher Conlon) to perform the procedure for testing firm conduct developed in "Testing Firm Conduct" by Marco Duarte, Lorenzo Magnolfi, Mikkel Solvsten, and Christopher Sullivan.


## Install
First, you will need to download and install python, which you can do from this [link](https://www.python.org/).

Then you will need to install four python packages: 
* numpy
* pandas
* statsmodels
* pyblp

These can be installed by running the pip3 install command in either terminal (Mac) or Command Prompt (Windows).  For example, to install numpy, run the following:

    pip3 install numpy

Finally, you should download the folder pyRV_folder.  


## Running the code
First open python3 (pyRV has been developed for releases of python 3.6 and later) in either terminal or command prompt

### Imposting the packages including pyRV
Then you import the necessary packages:

        import numpy as np
        import pandas as pd
        import pyblp
        import sys

To import pyRV, you specify the path on your computer for the pyRV_folder you downloaded

        pyRV_path = '<user specified path>/pyRV_folder'
        sys.path.append(pyRV_path)
        import pyRV

### Load the main dataset
Load the main dataset, which we refer to as `product_data`:

        product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)

### Estimate demand with pyblp
Next, you extimate demand using pyblp.  An excellent step-by-step tutorial for doing so can be found [here](https://pyblp.readthedocs.io/en/stable/index.html). Both pyblp and pyRV come with the Nevo (2000) fake cereal dataset.  For example, to estimate demand with pyblp, one would run the following code:

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
            sigma = np.diag([0.3302, 2.4526, 0.0163, 0.2441]), 
            pi = [[5.4819,0, 0.2037 ,0 ],[15.8935,-1.2000, 0 ,2.6342 ],[-0.2506,0, 0.0511 ,0 ],[1.2650,0, -0.8091 ,0 ]  ],
            method = '1s', 
            optimization = pyblp.Optimization('bfgs',{'gtol':1e-5})  
            )
