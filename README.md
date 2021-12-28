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

````
pip3 install numpy
````

Finally, you should download the folder pyRV_folder.  


## Running the code
First open python3 (pyRV has been developed for releases of python 3.6 and later) in either terminal or command prompt

### Import python packages
Then you import the necessary packages:

````
import numpy as np
import pandas as pd
import pyblp
import sys
````



To import pyRV, you specify the path on your computer for the pyRV_folder you downloaded

````
pyRV_path = '<user specified path>/pyRV_folder'
sys.path.append(pyRV_path)
import pyRV
````

### Load the main dataset
In this tutorial, we are going to use the Nevo (2000) fake cereal data which is provided in both the pyblp and pyRV packages.  pyblp has excellent [documentation](https://pyblp.readthedocs.io/en/stable/index.html) including a thorough tutorial for estimating demand on this dataset which can be found [here](https://pyblp.readthedocs.io/en/stable/_notebooks/tutorial/nevo.html).   

First you load the main dataset, which we refer to as `product_data`:

````
product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
````

### Estimate demand with pyblp
Next, you extimate demand using pyblp.  

````
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
````

In this example the linear characteristics in consumer preferences are price and product fixed effects.  Nonlinear characteristics with random coefficients include a constant, `price`, `sugar`, and `mushy`.  For these variables, we are going to estimate both the variance of the random coefficient as well as demographic interactions for `income`, `income_squared`, `age`, and `child`. `price`, `sugar`, and `mushy` are variables in the `product_data`.  Draws of `income`, `income_squared`, `age`, and `child` are in the agent data.  More info can be found [here](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.html)

The first 9 lines of code set up the demand estimation problem.  Running them yeilds output which summarize the dimensions of the problem (see [here](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.html)) for description of each variable.  Also reported are the Formulations, i.e., the linear characteristics, non-linear characteristics for which we are estimating either variances or demographic interactions, and a list of the demographics being used.

````
Dimensions:
=================================================
 T    N     F    I     K1    K2    D    MD    ED 
---  ----  ---  ----  ----  ----  ---  ----  ----
94   2256   5   1880   1     4     4    20    1  
=================================================

Formulations:
===================================================================
       Column Indices:           0           1           2      3  
-----------------------------  ------  --------------  -----  -----
 X1: Linear Characteristics    prices                              
X2: Nonlinear Characteristics    1         prices      sugar  mushy
       d: Demographics         income  income_squared   age   child
===================================================================
````

The second block of code actually runs the estimation. It output includes information on computation as well as tables witrh the parameter estimates. A full list of post-estimation output which can be queried is found [here](https://pyblp.readthedocs.io/en/stable/_api/pyblp.ProblemResults.html).

````
    Problem Results Summary:
================================================================================================================
GMM     Objective      Gradient         Hessian         Hessian     Clipped  Weighting Matrix  Covariance Matrix
Step      Value          Norm       Min Eigenvalue  Max Eigenvalue  Shares   Condition Number  Condition Number 
----  -------------  -------------  --------------  --------------  -------  ----------------  -----------------
 1    +4.561514E+00  +6.914932E-06  +3.442537E-05   +1.649684E+04      0      +6.927228E+07      +8.395896E+08  
================================================================================================================

Cumulative Statistics:
===========================================================================
Computation  Optimizer  Optimization   Objective   Fixed Point  Contraction
   Time      Converged   Iterations   Evaluations  Iterations   Evaluations
-----------  ---------  ------------  -----------  -----------  -----------
 00:00:21       Yes          51           57          45479       141178   
===========================================================================

Nonlinear Coefficient Estimates (Robust SEs in Parentheses):
=========================================================================================================================================================
Sigma:         1             prices            sugar            mushy       |   Pi:        income       income_squared         age             child     
------  ---------------  ---------------  ---------------  ---------------  |  ------  ---------------  ---------------  ---------------  ---------------
  1      +5.580936E-01                                                      |    1      +2.291971E+00    +0.000000E+00    +1.284432E+00    +0.000000E+00 
        (+1.625326E-01)                                                     |          (+1.208569E+00)                   (+6.312149E-01)                 
                                                                            |                                                                            
prices   +0.000000E+00    +3.312489E+00                                     |  prices   +5.883251E+02    -3.019201E+01    +0.000000E+00    +1.105463E+01 
                         (+1.340183E+00)                                    |          (+2.704410E+02)  (+1.410123E+01)                   (+4.122564E+00)
                                                                            |                                                                            
sugar    +0.000000E+00    +0.000000E+00    -5.783552E-03                    |  sugar    -3.849541E-01    +0.000000E+00    +5.223427E-02    +0.000000E+00 
                                          (+1.350452E-02)                   |          (+1.214584E-01)                   (+2.598529E-02)                 
                                                                            |                                                                            
mushy    +0.000000E+00    +0.000000E+00    +0.000000E+00    +9.341447E-02   |  mushy    +7.483723E-01    +0.000000E+00    -1.353393E+00    +0.000000E+00 
                                                           (+1.854333E-01)  |          (+8.021081E-01)                   (+6.671086E-01)                 
=========================================================================================================================================================

Beta Estimates (Robust SEs in Parentheses):
===============
    prices     
---------------
 -6.272990E+01 
(+1.480321E+01)
===============
````

### Test Models of Conduct with pyRV
pyRV follows a similar structure to pyblp.  First, you set up the testing problem, then you run the test.  

#### Setting Up Testing Problem
Here is an example of the code to set up the testing problem for a simple example where we will test two models: (1) manufacturers set retail prices according to bertrand vs (2) maunfactureres set retail prices according to monopoly (i.e., perfect collusion).  We set up the testing problem with `pyRV.problem` and we store this as a variable `testing_problem`:
````
testing_problem = pyRV.Problem(
    cost_formulation = (
            pyRV.Formulation('0 + sugar', absorb = 'C(firm_ids)' )
        ),
    instrument_formulation = (
            pyRV.Formulation('0 + demand_instruments0 + demand_instruments1')
        ), 
    model_formulations = (
            pyRV.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids'),
            pyRV.ModelFormulation(model_downstream='monopoly', ownership_downstream='firm_ids')
       ),       
    product_data = product_data,
    demand_results = pyblp_results
        )
````
pyRV.problem takes the following imputs:
* `cost_formulation`: list of the variables for observed product characteristics.  In this example, we have defined the cost formulation as:`pyRV.Formulation('0 + sugar', absorb = 'C(firm_ids)' )`.  Here, `0` means no constant.  To use a constant, one would replace `0` with `1`.  We are also including the variable `sugar` as an observed cost shifter and this variable must be in the `product_data`.  Finally `absorb = 'C(firm_ids)'` specifies that we are including firm fixed effects which will be absorbed using [PYHDFE](https://github.com/jeffgortmaker/pyhdfe), a companion package to pyblp developed by Jeff Gortmaker and Anya Tarascina.  The variable `firm_ids` must also be in the `product_data`.
* `instrument_formulation`: list of the variables used as excluded instruments for testing.  In this example, we have defined the cost formulation as:`pyRV.Formulation('0 + demand_instruments0 + demand_instruments1')`.  Here, `0` means no constant and this should always be specfied as a 0.  Here, we have an important difference with pyblp.  With pyblp, one specifies the excluded instruments for demand estimation via a naming convention in the product_data: each excluded instrument for demand estimation begins with demand_instrument followed by a number ( i.e., `demand_instrument0`).  In pyRV, you specify directly the names of the variables in the `product_data` that you want to use as excluded instruments for testing. 
* `model_formulations`: Here the researcher specifies the models that she wants to test.  There is a built in library of models that the researcher can choose from discussed below.  Here, we have specified two ModelFormulations and therefore two models to test. the first model is specified as:
`pyRV.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids')` `model_downstream = 'bertrand'` indicates that retail prices are set according to bertrand.  `ownership_downstream='firm_ids'` specifies that the ownership matrix in each market should be built from the variable `firm_id` in the `product_data`.  Here, we have another difference with pyblp.  In pyblp, if one wants to build an ownership matrix, there must be a variable called `firm_id` in the `product_data`.  With pyRV, the researcher can pass any variable in the `product_data` as `ownership_downstream` and from this, the ownership martix in each market will be built
   (Future Release Note: We are working on adding additional models to this library as well as options for the researcher to specify their own markup function)
* `product_data`: same as in pyblp
* `demand_results`: set to `pyblp_results` which is the name of the variable defining pyblp.solve in the pyblp code above



Running the `pyRV.problem` block of code yeilds the following output:
````
Dimensions:
=============================
 T    N     L    M    EC   K0
---  ----  ---  ---  ----  --
94   2256   1    2    1    2 
=============================

Formulations:
==========================================================
Column Indices:            0                    1         
----------------  -------------------  -------------------
w: Marginal Cost         sugar                            
z0: Instruments   demand_instruments0  demand_instruments1
==========================================================

Models:
========================================
                         0         1    
--------------------  --------  --------
 Model - Downstream   monopoly  bertrand
  Model - Upstream      None      None  
Firm id - Downstream  monopoly  firm_ids
 Firm id - Upstream     None      None  
       VI ind           None      None  
========================================

````
The first table `Dimensions` reports the following statistics:
* T = number of markets
* N = number of observations
* L = number of instrument sets (each specified by an instrument formulation)
* M = number of models (each specified by a model formulation)
* EC = ??
* K0 = number of instruments in the first instrument set (with more than one instrument formulation, additional columns K1, K2, .. K(L-1) would be reported)

The second table `Formulations` reports the variables specified as observed cost shifers and excluded instruments. The first row indicates that sugar is the only included observed cost shifter (ignoring the fixed effects).  The second row indicates that `demand_instruments0` and `demand_instruments1` are the excluded instrunments for testing each model.

The third table `Models` specifies the models being tested where each model is a column in the table
* Model-Downsream reports the name of the model governing how retail prices are set (current options are bertrand, cournot, monopoly)
* Model-Upstream reports the name of the model governing how wholesale prices are set (current options are bertrand, cournot, monopoly).   In this example, we are ignoring upstream behavior and assuming manufacturers set retail prices directly as in Nevo (2001). 
* Firm id - Downstream: the variable in `product_data` used to make the ownership martix for setting retail conduct (prices or quantities).  If monopoly is specified as Model-Downstream, then Firm id - Downstream will default to monopoly and the ownership martrix in each market will be a matrix of ones.  
* Firm id - Upstream: same as Firm id - Downstream but for wholesale price or quantity behavior
* VI id = name of dummy variable indicating whether retailer and manufacturer are vertically integrated.

#### Running the Testing Procedure
Now that the problem is set up, we can run the test, which we do with the following code
````
testing_results = testing_problem.solve(
    demand_adjustment = 'no',
    se_type = 'unadjusted'
    )
````

Given that we define the variable `testing_problem` as pyRV.problem, we must write `testing_problem.solve` in the first line.  There are two user specified options in running the test:
* demand_adjustment: 'no' indicates that the user does not want to adjust standard errors to account for two-step estimation with demand.  'yes' indicates standard errors should be adjusted to account for demand estimation.
* se_type: 'unadjusted' means no clustering. 'clustered' indicated that all standard errors should be clustered.  In this case, a variable called `clustering_ids` which indicates the cluster to which each group belongs needs to appear in the `product_data`. See example below.

Both of these adjustments are implemented according to the formulas in Appendix C of Duarte, Magnolfi, Solvsten, and Sullivan (2021).

This block of code to solve the testing problem returns the following output:

````
Testing Results - Instruments z0:
=======================================================================
  TRV:                |   F-stats:              |   MCS:               
--------  ---  -----  |  ----------  ---  ----  |  ------  ------------
 models    0     1    |    models     0    1    |  models  MCS p-values
--------  ---  -----  |  ----------  ---  ----  |  ------  ------------
   0      0.0  1.144  |      0       0.0  13.3  |    0        0.252    
   1      0.0   0.0   |      1       0.0  0.0   |    1         1.0     
=======================================================================
F-stat critical values...                                                             
... for worst-case size:                                                             
......07.5% worst-case size:  0.0                                                        
......10.0% worst-case size:  0.0                                                        
......12.5% worst-case size:  0.0                                                        
... for maximal power:                                                             
......95% max power:  18.9                                                        
......75% max power:  13.2                                                        
......50% max power:  10.4                                                        
=====================================================================
````

This table first reports the pairwise RV test statistic given the specified adjustments to the standard errors.  Then the pairwise F-statistics are reported, again with the specified adjustments to the standard errors.  Finally, the p-values associated with the model confidence set are reported.  Details on the model confidence set procedure are found in Section 6 of Duarte, Magnolfi, Solvsten, and Sullivan (2021) which adapts the procedure in Hansen, Lunde, and Naison () to the setting of testing firm conduct.  Beneath the table are the appropriate critical values from Table 1 of DMSS given the number of instruments the researcher is using.  The researcher can compare her pariwise F-statistics to these critical values.  Here, we are using two instruments, so there are no size distortions above 2.5%.  However, for a target maximal power of 95%, the F-statistic of 13.3 is less than the critical value of 18.9, so the instruments are weak for power.


The testing procedure also stores additional output which the user can access:

Library of Models
