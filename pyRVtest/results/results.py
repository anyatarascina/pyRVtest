"""Conduct testing results and export helpers.

v0.4 step 9: extracted from the former ``pyRVtest/results.py`` module so
the ``results/`` subpackage can hold additional files (``panel.py``,
``aggregation.py``, ``diagnostics.py``) as later steps populate them.
``pyRVtest/results/__init__.py`` re-exports ``Progress`` and
``ProblemResults`` so existing imports (``from pyRVtest.results import
ProblemResults``) keep working.

Step 9 also adds four additive export methods on ``ProblemResults``:

- :meth:`ProblemResults.to_dataframe` — long-form pandas DataFrame of
  pairwise RV / F / MCS results with one row per (instrument set,
  model_i, model_j) combination.
- :meth:`ProblemResults.summary_df` — compact wide-form DataFrame with
  pass/reject indicators at the 5 percent level, suitable for tables.
- :meth:`ProblemResults.to_latex` — LaTeX tabular rendering of
  ``summary_df``; math symbols pass through unescaped.
- :meth:`ProblemResults.to_markdown` — GitHub-flavored markdown
  rendering of ``summary_df``.

The existing attributes (``TRV``, ``F``, ``MCS_pvalues``, ``markups``,
``marginal_cost``, ``taus``, etc.) are unchanged; the new methods only
read them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, List, Optional, Sequence, TYPE_CHECKING, Union

import numpy as np

from pyblp.utilities.basics import Array, StringRepresentation

from ..output import format_table

if TYPE_CHECKING:
    import pandas as pd
    from ..problem import Problem


@dataclass
class Progress:
    """Structured information passed from Problem.solve to ProblemResults.

    Examples
    --------
    >>> from pyRVtest.results import Progress
    >>> from dataclasses import fields
    >>> field_names = {f.name for f in fields(Progress)}
    >>> {'problem', 'markups', 'F', 'MCS_pvalues'}.issubset(field_names)
    True
    """
    problem: 'Problem'
    markups: Array
    markups_downstream: Array
    markups_upstream: Array
    markups_orthogonal: Array
    marginal_cost: Array
    tau_list: Array
    g: Array
    Q: Array
    RV_numerator: Array
    RV_denominator: Array
    test_statistic_RV: Array
    F: Array
    MCS_pvalues: Array
    rho: Array
    unscaled_F: Array
    F_cv_size_list: Array
    F_cv_power_list: Array
    symbols_size_list: Array
    symbols_power_list: Array
    cost_param: Optional[List[Any]] = None
    tau_list_per_instrument: Optional[List[Any]] = None


class ProblemResults(StringRepresentation):  # type: ignore[misc]
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

    Examples
    --------
    >>> from pyRVtest import ProblemResults  # doctest: +SKIP
    >>> # ProblemResults instances are returned by Problem.solve(); they are
    >>> # not meant to be constructed directly. See docs/tutorial.rst for an
    >>> # end-to-end example. A representative session is:
    >>> results = problem.solve()  # doctest: +SKIP
    >>> print(results)  # doctest: +SKIP
    >>> results.TRV  # doctest: +SKIP
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
    cost_param: Array

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
        self.cost_param = progress.cost_param
        self.tau_list_per_instrument = progress.tau_list_per_instrument

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
        blanks = ["  " for i in range(number_models)]
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

        Examples
        --------
        >>> results.to_pickle('/tmp/rv_results.pkl')  # doctest: +SKIP
        >>> # Reload with pyRVtest.read_pickle:
        >>> import pyRVtest  # doctest: +SKIP
        >>> loaded = pyRVtest.read_pickle('/tmp/rv_results.pkl')  # doctest: +SKIP
        """
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)

    # ------------------------------------------------------------------
    # v0.4 step 9: export helpers
    # ------------------------------------------------------------------

    def _number_of_models(self) -> int:
        """Number of candidate conduct models (M)."""
        return int(len(self.markups))

    def _number_of_instrument_sets(self) -> int:
        """Number of instrument sets (L)."""
        return int(len(self.TRV))

    def _instrument_set_labels(self) -> List[str]:
        """Return a human-readable label for each instrument set.

        Uses the string repr of the underlying ``Formulation`` where
        possible, falling back to ``"z{instrument_set}"`` when no
        labels are available (for example, when ``self.problem`` does
        not expose ``instrument_formulation`` — primarily a defensive
        guard for pickled pre-v0.4 ``ProblemResults``).
        """
        L = self._number_of_instrument_sets()
        instrument_formulation = getattr(self.problem, 'instrument_formulation', None)
        labels: List[str] = []
        if instrument_formulation is None:
            return [f"z{j}" for j in range(L)]
        # The attribute is either a single Formulation or a sequence.
        if hasattr(instrument_formulation, '__len__'):
            seq: Sequence[Any] = instrument_formulation
        else:
            seq = [instrument_formulation]
        for j in range(L):
            if j < len(seq):
                try:
                    labels.append(str(seq[j]))
                except Exception:  # pragma: no cover — defensive fallback
                    labels.append(f"z{j}")
            else:
                labels.append(f"z{j}")
        return labels

    def _model_labels(self) -> List[str]:
        """Return a human-readable label per candidate model.

        Prefer downstream-model names (e.g. ``"bertrand"``,
        ``"perfect_competition"``) when the ``Models`` recarray is
        available on ``self.problem``. Fall back to ``"model_{i}"``
        when ``self.problem.models`` is not present or does not carry
        the ``models_downstream`` column — again a defensive guard for
        older pickles.
        """
        M = self._number_of_models()
        models = getattr(self.problem, 'models', None)
        if models is None:
            return [f"model_{i}" for i in range(M)]
        try:
            downstream = models["models_downstream"]
        except (IndexError, ValueError, KeyError, TypeError):
            return [f"model_{i}" for i in range(M)]
        labels: List[str] = []
        for i in range(M):
            try:
                labels.append(str(downstream[i]))
            except Exception:  # pragma: no cover — defensive fallback
                labels.append(f"model_{i}")
        return labels

    def to_dataframe(self) -> 'pd.DataFrame':
        """Return a long-form pandas DataFrame of pairwise test results.

        One row per (instrument set, model i, model j) ordered pair with
        ``i != j``. Columns:

        - ``instrument_set``: integer index of the instrument set.
        - ``instrument_set_label``: human-readable instrument set label
          (the string repr of the underlying ``Formulation`` where
          available, else ``"z{instrument_set}"``).
        - ``model_i`` / ``model_j``: integer model indices.
        - ``model_i_label`` / ``model_j_label``: downstream model names
          where available, else ``"model_{i}"``.
        - ``TRV``: pairwise Rivers-Vuong test statistic.
        - ``F``: pairwise scaled F-statistic.
        - ``MCS_pvalue``: MCS p-value for model ``i`` in this instrument
          set (a function of ``i`` only; duplicated across ``j`` rows
          to keep the long form self-contained).

        Returns
        -------
        pd.DataFrame
            Long-form frame with ``L * M * (M - 1)`` rows, where ``L``
            is the number of instrument sets and ``M`` is the number of
            candidate models.
        """
        import pandas as pd

        L = self._number_of_instrument_sets()
        M = self._number_of_models()
        iv_labels = self._instrument_set_labels()
        model_labels = self._model_labels()

        records: List[dict[str, Any]] = []
        for j in range(L):
            trv_mat = np.asarray(self.TRV[j])
            f_mat = np.asarray(self.F[j])
            mcs_vec = np.asarray(self.MCS_pvalues[j]).reshape(-1)
            for i in range(M):
                for k in range(M):
                    if i == k:
                        continue
                    records.append({
                        'instrument_set': j,
                        'instrument_set_label': iv_labels[j],
                        'model_i': i,
                        'model_j': k,
                        'model_i_label': model_labels[i],
                        'model_j_label': model_labels[k],
                        'TRV': float(trv_mat[i, k]),
                        'F': float(f_mat[i, k]),
                        'MCS_pvalue': float(mcs_vec[i]),
                    })
        return pd.DataFrame.from_records(records, columns=[
            'instrument_set', 'instrument_set_label',
            'model_i', 'model_j', 'model_i_label', 'model_j_label',
            'TRV', 'F', 'MCS_pvalue',
        ])

    def summary_df(self, alpha: float = 0.05) -> 'pd.DataFrame':
        """Return a compact wide-form summary DataFrame.

        One row per (instrument set, unordered model pair) with the
        signed RV test statistic, F-statistic, MCS p-value for the
        lower-index model, and a stable ``reject`` indicator at the
        two-sided ``alpha`` level (default 5 percent). The pair is
        represented with ``model_i < model_j`` so each unordered pair
        appears exactly once per instrument set.

        The alpha level used for the rejection flag is recorded on the
        returned frame's ``attrs`` dict under the key ``'alpha'``, so
        downstream aggregators can read it without re-deriving the
        column name.

        Parameters
        ----------
        alpha : float, optional
            Two-sided significance level for the rejection flag on the
            RV statistic. Defaults to ``0.05``. Must lie strictly
            between 0 and 1.

        Returns
        -------
        pd.DataFrame
            Wide-form summary frame with ``L * M * (M - 1) / 2`` rows.

        Raises
        ------
        ValueError
            If ``alpha`` is not in the open interval ``(0, 1)``.
        """
        import pandas as pd
        from scipy.stats import norm

        if not (0.0 < alpha < 1.0):
            raise ValueError(
                f"alpha must lie strictly between 0 and 1 (got {alpha!r}). "
                f"Typical values are 0.01, 0.05, 0.10."
            )

        L = self._number_of_instrument_sets()
        M = self._number_of_models()
        iv_labels = self._instrument_set_labels()
        model_labels = self._model_labels()
        # Two-sided critical value.
        crit = float(norm.ppf(1.0 - alpha / 2.0))

        records: List[dict[str, Any]] = []
        for j in range(L):
            trv_mat = np.asarray(self.TRV[j])
            f_mat = np.asarray(self.F[j])
            mcs_vec = np.asarray(self.MCS_pvalues[j]).reshape(-1)
            for i in range(M):
                for k in range(i + 1, M):
                    trv_ik = float(trv_mat[i, k])
                    reject = bool(np.isfinite(trv_ik) and abs(trv_ik) > crit)
                    records.append({
                        'instrument_set': j,
                        'instrument_set_label': iv_labels[j],
                        'model_i': i,
                        'model_j': k,
                        'model_i_label': model_labels[i],
                        'model_j_label': model_labels[k],
                        'TRV': trv_ik,
                        'F': float(f_mat[i, k]),
                        'MCS_pvalue_model_i': float(mcs_vec[i]),
                        'reject': reject,
                    })
        columns = [
            'instrument_set', 'instrument_set_label',
            'model_i', 'model_j', 'model_i_label', 'model_j_label',
            'TRV', 'F', 'MCS_pvalue_model_i', 'reject',
        ]
        frame = pd.DataFrame.from_records(records, columns=columns)
        frame.attrs['alpha'] = alpha
        return frame

    def to_latex(
        self,
        path: Optional[Union[str, Path]] = None,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        alpha: float = 0.05,
    ) -> Optional[str]:
        r"""Render :meth:`summary_df` as a LaTeX ``tabular``.

        Wraps :func:`pandas.DataFrame.to_latex` with ``escape=False`` so
        math symbols in model or instrument labels pass through
        untouched. When ``caption`` or ``label`` is supplied the output
        is wrapped in a floating ``table`` environment.

        Parameters
        ----------
        path : str or Path, optional
            If given, the LaTeX string is written to this path and the
            method returns ``None``. If ``None`` (default), the LaTeX
            string is returned.
        caption : str, optional
            LaTeX caption. Forwarded to
            :func:`pandas.DataFrame.to_latex`.
        label : str, optional
            LaTeX label (e.g. ``"tab:rv_results"``). Forwarded to
            :func:`pandas.DataFrame.to_latex`.
        alpha : float, optional
            Forwarded to :meth:`summary_df`. Defaults to ``0.05``.

        Returns
        -------
        str or None
            The LaTeX string when ``path`` is ``None``; otherwise
            ``None`` (the string was written to ``path``).
        """
        summary = self.summary_df(alpha=alpha)
        tex: str = summary.to_latex(
            index=False,
            escape=False,
            caption=caption,
            label=label,
        )
        if path is not None:
            Path(path).write_text(tex)
            return None
        return tex

    def to_markdown(
        self,
        path: Optional[Union[str, Path]] = None,
        alpha: float = 0.05,
    ) -> Optional[str]:
        """Render :meth:`summary_df` as a GitHub-flavored markdown table.

        Implemented without relying on pandas' optional ``tabulate``
        dependency: we emit pipes and hyphens directly so the output is
        usable in any GitHub-style markdown viewer with no extra
        install.

        Parameters
        ----------
        path : str or Path, optional
            If given, the markdown string is written to this path and
            the method returns ``None``. If ``None`` (default), the
            markdown string is returned.
        alpha : float, optional
            Forwarded to :meth:`summary_df`. Defaults to ``0.05``.

        Returns
        -------
        str or None
            The markdown string when ``path`` is ``None``; otherwise
            ``None``.
        """
        summary = self.summary_df(alpha=alpha)
        md = _dataframe_to_github_markdown(summary)
        if path is not None:
            Path(path).write_text(md)
            return None
        return md


def _dataframe_to_github_markdown(frame: 'pd.DataFrame') -> str:
    """Render a DataFrame as a GitHub-flavored markdown pipe table.

    Local helper used by :meth:`ProblemResults.to_markdown`. Kept
    out of pandas to avoid the optional ``tabulate`` dependency that
    :meth:`pandas.DataFrame.to_markdown` pulls in; GitHub-flavored
    markdown only needs pipes and hyphens with an empty-row guard.

    Parameters
    ----------
    frame : pd.DataFrame
        The frame to render. Values are rendered via ``format(value)``
        with ``:.6g`` for floats and ``str`` otherwise.

    Returns
    -------
    str
        A multi-line markdown string. Trailing newline included.
    """
    columns = [str(c) for c in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body_lines: List[str] = []
    if len(frame) == 0:
        # GitHub renders zero-row tables correctly with just header + sep.
        return "\n".join([header, sep]) + "\n"
    for _, row in frame.iterrows():
        cells: List[str] = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                if np.isnan(val):
                    cells.append("NaN")
                else:
                    cells.append(f"{val:.6g}")
            elif isinstance(val, (bool, np.bool_)):
                cells.append("True" if bool(val) else "False")
            else:
                cells.append(str(val))
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body_lines]) + "\n"


__all__ = ['Progress', 'ProblemResults']
