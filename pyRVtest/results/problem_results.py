"""Economy-level structuring of BLP problem results."""

import itertools
import time
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, TYPE_CHECKING, Tuple

import numpy as np
import scipy.linalg
import pandas as pd
import pyRVtest.data as DATA
from .results import Results
from .. import exceptions, options

from ..primitives import Models
from ..utilities.algebra import (
    approximately_invert, approximately_solve, compute_condition_number, precisely_compute_eigenvalues, vech_to_full
)
from ..utilities.basics import (
    Array, Bounds, Error, Mapping, RecArray, SolverStats, format_number, format_seconds, format_table, format_table_notes, format_table_notes_sub, generate_items,
    get_indices, output, output_progress, update_matrices
)


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from ..economies.problem import Progress  # noqa


class ProblemResults(Results):
    r"""Results of a testing.

   
    Attributes
    ----------
    

    """

   
    Delta: Array
    

    def __init__(
            self, progress: 'Progress'
            ) -> None:
        self.problem = progress.problem
        self.markups = progress.markups
        self.markups_downstream = progress.markups_downstream
        self.markups_upstream = progress.markups_upstream
        self.taus = progress.taus
        self.mc = progress.mc
        self.g = progress.g
        self.Q = progress.Q
        self.RV_num = progress.RV_num
        self.RV_denom = progress.RV_denom
        self.TRV = progress.TRV
        self.F = progress.F
        self.MCS_pvalues = progress.MCS_pvalues

    def __str__(self) -> str:
        """Format economy information as a string."""
        out = ""
        for zz in range(len(self.TRV)):
            tmp =  "\n\n".join([self._format_Fstats_notes(zz)])
            out = "\n\n".join([out,tmp])    
        return out
    def _format_RVstats(self, zz: int) -> str:
        """Formation information about the formulations of the economy as a string."""

        # construct the data
        
        data: List[List[str]] = []
        for kk in range(len(self.markups)):
            data.append([str(kk)] + [round(self.TRV[zz][kk,i],3) for i in range(len(self.markups))])
    
        # construct the header
        header = [" "] + [f" {i} " for i in range(len(self.markups))]

        return format_table(header, *data, title="RV test statistics - Instruments {0}".format(zz))    

    def _format_Fstats_notes(self, zz: int) -> str:

        """Formation information about the formulations of the economy as a string."""
        # get critical values

    
        CVall_s = pd.read_csv(DATA.F_CRITVALS_SIZE)
        ind = CVall_s['K'] == len(self.g[zz][0])
        tmp = np.array(CVall_s['r_075'])
        cv_s075 = float(tmp[ind])
        tmp = np.array(CVall_s['r_10'])
        cv_s10 = float(tmp[ind])
        tmp = np.array(CVall_s['r_125'])
        cv_s125 = float(tmp[ind])
        
        CVall_p = pd.read_csv(DATA.F_CRITVALS_POWER)
        ind = CVall_p['K'] == len(self.g[zz][0])
        tmp = np.array(CVall_p['r_95'])
        cv_p95 = float(tmp[ind])
        tmp = np.array(CVall_p['r_75'])
        cv_p75 = float(tmp[ind])
        tmp = np.array(CVall_p['r_50'])
        cv_p50 = float(tmp[ind])
        
        cvs: List[List[str]] = []
        cvs.append(['F-stat critical values...'])
        cvs.append(['... for worst-case size:'])
        cvs.append(['......07.5% worst-case size:'] + [str(cv_s075)])                                  
        cvs.append(['......10.0% worst-case size:'] + [str(cv_s10)])   
        cvs.append(['......12.5% worst-case size:'] + [str(cv_s125)])
        cvs.append(['... for maximal power:'])              
        cvs.append(['......95% max power:'] + [str(cv_p95)])                                          
        cvs.append(['......75% max power:'] + [str(cv_p75)]) 
        cvs.append(['......50% max power:'] + [str(cv_p50)])                                          
                                           
                                                      
        # construct the data

        data: List[List[str]] = []
        for kk in range(len(self.markups)):
            data.append([str(kk)] + [round(self.TRV[zz][kk,i],3) for i in range(len(self.markups))] + [str(kk)] +  [round(self.F[zz][kk,i],1) for i in range(len(self.markups))] + [str(kk)] + [str(round(self.MCS_pvalues[zz][kk][0],3))])
        
        # construct the header
        header =  [" TRV: "] + [f"  " for i in range(len(self.markups))]+ [" F-stats: "] + [f"  " for i in range(len(self.markups))] + [" MCS: "] + [" "]
        subheader = [" models "] + [f" {i} " for i in range(len(self.markups))]+ [" models "] + [f" {i} " for i in range(len(self.markups))] + ["models"] +  ["MCS p-values"] 
        return format_table_notes_sub(header, subheader, *data, title="Testing Results - Instruments z{0}".format(zz), notes = cvs, line_indices =[len(self.markups),2*len(self.markups)+1])    

    
    def to_dict(
            self, attributes: Sequence[str] = (
                'Delta'
            )) -> dict:
        """Convert these results into a dictionary that maps attribute names to values.

        Parameters
        ----------
        attributes : `sequence of str, optional`
            Name of attributes that will be added to the dictionary. By default, all :class:`ProblemResults` attributes
            are added except for :attr:`ProblemResults.problem` and :attr:`ProblemResults.last_results`.

        Returns
        -------
        `dict`
            Mapping from attribute names to values.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        return {k: getattr(self, k) for k in attributes}

    

    def _coerce_matrices(self, matrices: Any, market_ids: Array) -> Array:
        """Coerce array-like stacked matrices into a stacked matrix and validate it."""
        matrices = np.c_[np.asarray(matrices, options.dtype)]
        rows = sum(i.size for t, i in self.problem._product_market_indices.items() if t in market_ids)
        columns = max(i.size for t, i in self.problem._product_market_indices.items() if t in market_ids)
        if matrices.shape != (rows, columns):
            raise ValueError(f"matrices must be {rows} by {columns}.")
        return matrices

    
