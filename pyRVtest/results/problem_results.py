"""Economy-level structuring of conduct testing problem results."""

from pathlib import Path
import pickle
from typing import List, Union, TYPE_CHECKING

from pyblp.utilities.basics import Array

from .results import Results
from ..utilities.basics import format_table


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from ..economies.problem import Progress


class ProblemResults(Results):
    r"""Results of running the firm conduct testing procedures.

    Attributes
    ----------
        problem: `ndarray`
            An instance of the Problem class.
        markups: `ndarray`
            Array of the total markups implied by each model (sum of retail and wholesale markups).
        markups_downstream: `ndarray`
            Array of the retail markups implied by each model.
        markups_upstream: `ndarray`
            Array of the manufacturer markups implied by each model of double marginalization.
        marginal_cost: `ndarray`
            Array of implied marginal costs for each model.
        taus: `ndarray`
            Array of coefficients from regressing implied marginal costs for each model on observed cost shifters.
        g: `ndarray`
            Array of moments for each model and each instrument set of conduct between implied residualized cost
            unobservable and the instruments.
        Q: `ndarray`
            Array of lack of fit given by GMM objective function with 2SLS weight matrix for each set of instruments and
            each model.
        RV_numerator: `ndarray`
            Array of numerators of pairwise RV test statistics for each instrument set and each pair of models.
        RV_denominator: `ndarray`
            Array of denominators of pairwise RV test statistics for each instrument set and each pair of models.
        TRV: `ndarray`
            Array of pairwise RV test statistics for each instrument set and each pair of models.
        F: `ndarray`
            Array of pairwise F-statistics for each instrument set and each pair of models.
        MCS_pvalues: `ndarray`
            Array of MCS p-values for each instrument set and each model.
        rho: `ndarray`
            Scaling parameter for F-statistics.
        unscaled_F: `ndarray`
            Array of pairwise F-statistics without scaling by rho.
        F_cv_size_list: `ndarray`
            Vector of critical values for size for each pairwise F-statistic.
        F_cv_power_list: `ndarray`
            Vector of critical values for power for each pairwise F-statistic.

    """

    problem: Array
    markups: Array
    markups_downstream: Array
    markups_upstream: Array
    marginal_cost: Array
    taus: Array
    g: Array
    Q: Array
    RV_numerator: Array
    RV_denominator: Array
    TRV: Array
    F: Array
    MCS_pvalues: Array
    rho: Array
    unscaled_F: Array
    F_cv_size_list: Array
    F_cv_power_list: Array
    _symbols_size_list: Array
    _symbols_power_list: Array

    def __init__(self, progress: 'Progress') -> None:
        self.problem = progress.problem
        self.markups = progress.markups
        self.markups_downstream = progress.markups_downstream
        self.markups_upstream = progress.markups_upstream
        self.marginal_cost = progress.marginal_cost
        self.taus = progress.tau_list
        self.g = progress.g
        self.Q = progress.Q
        self.RV_numerator = progress.RV_numerator
        self.RV_denominator = progress.RV_denominator
        self.TRV = progress.test_statistic_RV
        self.F = progress.F
        self.MCS_pvalues = progress.MCS_pvalues
        self.rho = progress.rho
        self.unscaled_F = progress.unscaled_F
        self.F_cv_size_list = progress.F_cv_size_list
        self.F_cv_power_list = progress.F_cv_power_list
        self._symbols_size_list = progress.symbols_size_list
        self._symbols_power_list = progress.symbols_power_list

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
                self._symbols_size_list[j][k, i] + " " + self._symbols_power_list[j][k, i] for i in range(number_models)
            ]
            data.append([str(k)] + rv_results + [str(k)] + f_stat_results + [str(k)] + pvalues_results)
            data.append([""] + ["" for i in range(number_models)] + [""] + symbols_results + [""] + [""])

        # construct the header
        blanks = [f"  " for i in range(number_models)]
        numbers = [f" {i} " for i in range(number_models)]
        header = [" TRV: "] + blanks + [" F-stats: "] + blanks + [" MCS: "] + [" "]
        subheader = [" models "] + numbers + [" models "] + numbers + [" models "] + ["MCS p-values"]

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
