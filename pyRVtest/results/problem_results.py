"""Economy-level structuring of conduct testing problem results."""

from pathlib import Path
import pickle
from typing import List, Union, TYPE_CHECKING

from pyblp.utilities.basics import (Array, format_table)

from .results import Results
from ..utilities.basics import format_table


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from ..economies.problem import Progress


class ProblemResults(Results):
    r"""Results of a testing procedures.

    Attributes
    ----------
        Delta: `ndarray`
            # TODO: add comment
        problem: `ndarray`
            # TODO: add comment
        markups: `ndarray`
            # TODO: add comment
        markups_downstream: `ndarray`
            # TODO: add comment
        markups_upstream: `ndarray`
            # TODO: add comment
        taus: `ndarray`
            # TODO: add comment
        mc: `ndarray`
            # TODO: add comment
        g: `ndarray`
            # TODO: add comment
        Q: `ndarray`
            # TODO: add comment
        RV_numerator: `ndarray`
            # TODO: add comment
        RV_denominator: `ndarray`
            # TODO: add comment
        TRV: `ndarray`
            # TODO: add comment
        F: `ndarray`
            # TODO: add comment
        MCS_pvalues: `ndarray`
            # TODO: add comment
        rho: `ndarray`
            # TODO: add comment
        unscaled_F: `ndarray`
            # TODO: add comment
        AR_variance: `ndarray`
            # TODO: add comment
        F_cv_size_list: `ndarray`
            # TODO: add comment
        F_cv_power_list: `ndarray`
            # TODO: add comment
        symbols_size_list: `ndarray`
            # TODO: add comment
        symbols_power_list: `ndarray`
            # TODO: add comment

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
    RV_numerator: Array
    RV_denominator: Array
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
        self.RV_numerator = progress.RV_numerator
        self.RV_denominator = progress.RV_denominator
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
        """Format results information as a string."""
        out = ""
        for i in range(len(self.TRV)):
            tmp = "\n\n".join([self._format_results_tables(i)])
            out = "\n\n".join([out, tmp])
        return out

    def _format_results_tables(self, j: int) -> str:
        """Formation information about the testing results as a string."""

        # construct the data
        data: List[List[str]] = []
        number_models = len(self.markups)
        for k in range(number_models):
            rv_results = [round(self.TRV[j][k, i], 3) for i in range(number_models)]
            f_stat_results = [round(self.F[j][k, i], 1) for i in range(number_models)]
            pvalues_results = [str(round(self.MCS_pvalues[j][k][0], 3))]
            symbols_results = [
                self.symbols_size_list[j][k, i] + " " + self.symbols_power_list[j][k, i] for i in range(number_models)
            ]
            data.append([str(k)] + rv_results + [str(k)] + f_stat_results + [str(k)] + pvalues_results)
            data.append([""] + ["" for i in range(number_models)] + [""] + symbols_results + [""] + [""])
        
        # construct the header
        header = [" TRV: "] + [f"  " for i in range(number_models)] + [" F-stats: "] \
            + [f"  " for i in range(number_models)] + [" MCS: "] + [" "]
        subheader = [" models "] + [f" {i} " for i in range(number_models)] + [" models "] \
            + [f" {i} " for i in range(number_models)] + ["models"] + ["MCS p-values"]

        # if on the last table, set table notes to true
        last_table = False
        if j == (len(self.TRV) - 1):
            last_table = True

        return format_table(
            header, subheader, *data, title="Testing Results - Instruments z{0}".format(j), include_notes=last_table,
            line_indices=[number_models, 2 * number_models + 1]
        )

    def to_pickle(self, path: Union[str, Path]) -> None:
        """Save these results as a pickle file. This function is copied from PyBLP.

        Parameters
        ----------
        path: `str or Path`
            File path to which these results will be saved.

        """
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)
