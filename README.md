# pyRV

This code was written to perform the procedure for testing firm conduct developed in "Testing Firm Conduct" by Marco Duarte, Lorenzo Magnolfi, Mikkel Solvsten, and Christopher Sullivan.  It largely adapts the PyBLP source code (copyright Jeff Gortmaker and Christopher Conlon) to do so.

The code implements the following features:
* Computes RV test statistics to test a menu of two or more models of firm conduct - see `Models` below for the current supported models
* Implements the RV test using the variance estimator of [Duarte, Magnolfi, Solvsten, and Sullivan (2021)](#dmss), including options to adjust for demand estimation error and clustering
* Computes the effective F-statistic proposed in [Duarte, Magnolfi, Solvsten, and Sullivan (2021)](#dmss) to diagnose instrument strength with respect to worst-case size and maximal power of the test, and reports appropriate critical values 
* Reports [Hansen, Lunde, and Nason (2011)](#hln) MCS p-values for testing more than two models

For a full list of references, see the references in [Duarte, Magnolfi, Solvsten, and Sullivan (2021)](#dmss).

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

Finally, you should download the pyRV code.  To do so, click on the green Code button above.  Then click "Download ZIP".      


## Import python packages
First open python3 (pyRV has been developed for releases of python 3.6 and later) in either terminal or command prompt.  Then you import the necessary packages:

````
import numpy as np
import pandas as pd
import pyblp
import sys
````



To import pyRV, you specify the path on your computer for the pyRV_folder you downloaded

````
pyRV_path = '<user specified path>/pyRV-Main/pyRV_folder'
sys.path.append(pyRV_path)
import pyRV
````
## Running the code

For a detailed tutorial about how to set up and run the code, see [tutorial.md](https://github.com/chrissullivanecon/pyRV/blob/main/TUTORIAL.md)
