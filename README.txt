# pyRVtest

This code was written to perform the procedure for testing firm conduct developed in "Testing Firm Conduct" by Marco Duarte, Lorenzo Magnolfi, Mikkel Solvsten, and Christopher Sullivan.  It largely adapts the PyBLP source code (copyright Jeff Gortmaker and Christopher Conlon) to do so.

The code implements the following features:
* Computes [Rivers and Vuong (2002)](#rv) (RV) test statistics to test a menu of two or more models of firm conduct - see `Models` in [tutorial.md](https://github.com/chrissullivanecon/pyRV/blob/main/TUTORIAL.md) for the current supported models
* Implements the RV test using the variance estimator of [Duarte, Magnolfi, Solvsten, and Sullivan (2021)](#dmss), including options to adjust for demand estimation error and clustering
* Computes the effective F-statistic proposed in [Duarte, Magnolfi, Solvsten, and Sullivan (2021)](#dmss) to diagnose instrument strength with respect to worst-case size and maximal power of the test, and reports appropriate critical values 
* Reports [Hansen, Lunde, and Nason (2011)](#hln) MCS p-values for testing more than two models

For a full list of references, see the references in [Duarte, Magnolfi, Solvsten, and Sullivan (2021)](#dmss).

## Install
First, you will need to download and install python, which you can do from this [link](https://www.python.org/).

Then you will need to install the pyRVtest python package: 

````
pip install pyRVtest
````

This should automatically install the python packages on which pyRVtest depends: numpy, pandas, statsmodels, pyblp

These can also be installed by running the pip install command in either terminal (Mac) or Command Prompt (Windows). 
## Import python packages
First open python3 (pyRV has been developed for releases of python 3.6 and later) in either terminal or command prompt.  Then you import the necessary packages:

````
import numpy as np
import pandas as pd
import pyblp
import pyRVtest
````

## Running the code

For a detailed tutorial about how to set up and run the code, see [tutorial.md](https://github.com/chrissullivanecon/pyRV/blob/main/TUTORIAL.md)
