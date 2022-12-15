r"""Locations of example data that are included in the package for convenience.

Attributes
----------
F_CRITICAL_VALUES_POWER : `str`
    Location of a CSV file containing # TODO: finish description
F_CRITICAL_VALUES_SIZE : `str`
    Location of a CSV file containing # TODO: finish description

Examples
--------
.. toctree::

   /_notebooks/api/data.ipynb

"""

from pathlib import Path


_DATA_PATH = Path(__file__).resolve().parent
F_CRITICAL_VALUES_POWER = str(_DATA_PATH / 'f_critical_values_power.csv')
F_CRITICAL_VALUES_SIZE = str(_DATA_PATH / 'f_critical_values_size.csv')
F_CRITICAL_VALUES_POWER_RHO = str(_DATA_PATH / 'f_critical_values_power_rho.csv')
F_CRITICAL_VALUES_SIZE_RHO = str(_DATA_PATH / 'f_critical_values_size_rho.csv')
