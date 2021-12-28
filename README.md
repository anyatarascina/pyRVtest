# pyRV

This code was written to perform the procedure for testing firm conduct developed in "Testing Firm Conduct" by Marco Duarte, Lorenzo Magnolfi, Mikkel Solvsten, and Christopher Sullivan.  It largely adapts the pyblp source code (copyright Jeff Gortmaker and Christopher Conlon) to do so.


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
pyRV_path = '<user specified path>/pyRV_folder'
sys.path.append(pyRV_path)
import pyRV
````
## Running the code

For a detailed tutorial about how to set up and run the code, see [tutorial.md](https://github.com/chrissullivanecon/pyRV/blob/main/TUTORIAL.md)
