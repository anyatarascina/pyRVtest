r"""Locations of critival value tables that are used to evaluate whether the instruments being tested are weak for size
or power.

Attributes
----------
F_CRITICAL_VALUES_POWER_RHO : `str`
    Location of a CSV file containing critical values for power for each combination of :math:`\rho` and number of
    instruments.
F_CRITICAL_VALUES_SIZE_RHO : `str`
    Location of a CSV file containing critical values for size for each combination of :math:`\rho` and number of
    instruments.

"""

from pathlib import Path


_DATA_PATH = Path(__file__).resolve().parent
F_CRITICAL_VALUES_POWER_RHO = str(_DATA_PATH / 'f_critical_values_power_rho.csv')
F_CRITICAL_VALUES_SIZE_RHO = str(_DATA_PATH / 'f_critical_values_size_rho.csv')
