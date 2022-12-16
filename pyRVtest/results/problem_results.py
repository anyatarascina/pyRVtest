"""Economy-level structuring of BLP problem results."""

from typing import List, TYPE_CHECKING

from pyblp.utilities.basics import (Array, format_table)

from .results import Results
from ..utilities.basics import format_table_notes


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from ..economies.problem import Progress


class ProblemResults(Results):
    r"""Results of a testing procedures.

   
    Attributes
    ----------
    

    """

    Delta: Array
    problem: Array
    markups: Array
    markups_downstream: Array
    markups_upstream: Array
    taus: Array
    mc: Array
    g: Array
    Q: Array
    RV_num: Array
    RV_denom: Array
    TRV: Array
    F: Array
    MCS_pvalues: Array
    rho: Array
    unscaled_F: Array
    AR_variance: Array
    F_cv_size_list: Array
    F_cv_power_list: Array
    symbols_size_list: Array
    symbols_power_list: Array

    def __init__(self, progress: 'Progress') -> None:
        self.problem = progress.problem
        self.markups = progress.markups
        self.markups_downstream = progress.markups_downstream
        self.markups_upstream = progress.markups_upstream
        self.taus = progress.tau_list
        self.mc = progress.mc
        self.g = progress.g
        self.Q = progress.Q
        self.RV_num = progress.RV_numerator
        self.RV_denom = progress.RV_denominator
        self.TRV = progress.test_statistic_RV
        self.F = progress.F
        self.MCS_pvalues = progress.MCS_p_values
        self.rho = progress.rho
        self.unscaled_F = progress.unscaled_F
        self.AR_variance = progress.AR_variance
        self.F_cv_size_list = progress.F_cv_size_list
        self.F_cv_power_list = progress.F_cv_power_list
        self.symbols_size_list = progress.symbols_size_list
        self.symbols_power_list = progress.symbols_power_list

    def __str__(self) -> str:
        """Format economy information as a string."""
        out = ""
        for zz in range(len(self.TRV)):
            tmp = "\n\n".join([self._format_Fstats_notes(zz)])
            out = "\n\n".join([out, tmp])
        return out

    def _format_rv_stats(self, zz: int) -> str:
        """Formation information about the formulations of the economy as a string."""

        # construct the data
        data: List[List[str]] = []
        for kk in range(len(self.markups)):
            data.append([str(kk)] + [round(self.TRV[zz][kk, i], 3) for i in range(len(self.markups))])
    
        # construct the header
        header = [" "] + [f" {i} " for i in range(len(self.markups))]

        return format_table(header, *data, title="RV test statistics - Instruments {0}".format(zz))    

    def _format_Fstats_notes(self, zz: int) -> str:
        """Formation information about the formulations of the economy as a string."""

        # format output
        cvs: List[List[str]] = []
        cvs.append(['Significance of size and power diagnostic reported below each F-stat'])
        cvs.append(['*, **, or *** indicate that F > cv for a target size of 0.125, 0.10, and 0.075 given d_z and rho'])
        cvs.append(['^, ^^, or ^^ indicate that F > cv for a maximal power of 0.50, 0.75, and 0.95 given d_z and rho'])
        cvs.append(['appropriate critical values for size are stored in the variable F_cv_size_list of the pyRVtest results class'])
        cvs.append(['appropriate critical values for power are stored in the variable F_cv_power_list of the pyRVtest results class'])

        # construct the data
        data: List[List[str]] = []
        for kk in range(len(self.markups)):
            # TODO: check pep8 for these next 2 blocks
            data.append(
                [str(kk)] + [round(self.TRV[zz][kk, i], 3) for i in range(len(self.markups))] + [str(kk)]
                + [round(self.F[zz][kk, i], 1) for i in range(len(self.markups))] + [str(kk)]
                + [str(round(self.MCS_pvalues[zz][kk][0], 3))]
            )
            data.append(
                [""] + ["" for i in range(len(self.markups))] + [""] +
                [self.symbols_size_list[zz][kk, i] + " " + self.symbols_power_list[zz][kk, i] for i in range(len(self.markups))] + [""] + [""]
            )
        
        # construct the header
        header = [" TRV: "] + [f"  " for i in range(len(self.markups))] + [" F-stats: "] \
            + [f"  " for i in range(len(self.markups))] + [" MCS: "] + [" "]
        subheader = [" models "] + [f" {i} " for i in range(len(self.markups))] + [" models "] \
            + [f" {i} " for i in range(len(self.markups))] + ["models"] + ["MCS p-values"]
        return format_table_notes(
            header, subheader, *data, title="Testing Results - Instruments z{0}".format(zz), notes=cvs,
            line_indices=[len(self.markups), 2 * len(self.markups) + 1]
        )
