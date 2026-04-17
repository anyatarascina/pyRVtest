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
from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


_DATA_PATH = Path(__file__).resolve().parent
F_CRITICAL_VALUES_POWER_RHO = str(_DATA_PATH / 'f_critical_values_power_rho.csv')
F_CRITICAL_VALUES_SIZE_RHO = str(_DATA_PATH / 'f_critical_values_size_rho.csv')

_critical_values_cache: Optional[Tuple[NDArray[Any], NDArray[Any]]] = None


def read_critical_values_tables() -> Tuple[NDArray[Any], NDArray[Any]]:
    """Read in the critical values for size and power from the corresponding csv file. These will be used to evaluate
    the strength of the instruments. Results are cached after the first read."""
    global _critical_values_cache
    if _critical_values_cache is not None:
        return _critical_values_cache

    # read in data for critical values for size as a structured array
    critical_values_size = np.genfromtxt(
        F_CRITICAL_VALUES_SIZE_RHO,
        delimiter=',',
        skip_header=1,
        dtype=[('K', 'i4'), ('rho', 'f8'), ('r_075', 'f8'), ('r_10', 'f8'), ('r_125', 'f8')]
    )

    # read in data for critical values for power as a structured array
    critical_values_power = np.genfromtxt(
        F_CRITICAL_VALUES_POWER_RHO,
        delimiter=',',
        skip_header=1,
        dtype=[('K', 'i4'), ('rho', 'f8'), ('r_50', 'f8'), ('r_75', 'f8'), ('r_95', 'f8')]
    )

    _critical_values_cache = (critical_values_power, critical_values_size)
    return _critical_values_cache
