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
from typing import Any, Dict, Hashable, List, Optional, Sequence, TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from pyblp.utilities.basics import Array, StringRepresentation

from .. import options
from ..exceptions import ValidationError
from ..models.vertical import Vertical
from ..output import format_table
from ..solve.passthrough import build_passthrough
from ._format import _dataframe_to_github_markdown

if TYPE_CHECKING:
    import pandas as pd
    from ..problem import Problem


# ----------------------------------------------------------------------
# F-stat reliability footer helpers (Phase 2 of the F-reliability work).
# ----------------------------------------------------------------------

def _worst_cell_borderline(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Among borderline cells, pick the one with the smallest CI lower bound
    relative to its strongest claim's CV — that's the most-fragile case to
    surface in the footer."""
    return cells[0] if len(cells) == 1 else min(cells, key=lambda c: c['F_ci_low'])


def _borderline_lines(cells: List[Dict[str, Any]]) -> List[List[str]]:
    """Build the borderline footer line(s). For a single cell, show the
    specific F value, CI, and CV. For multiple cells, summarize the worst
    case."""
    n = len(cells)
    suffix = '' if n == 1 else 's'
    if n == 1:
        c = cells[0]
        claim = c['claim_size'] or c['claim_power'] or '(none)'
        # Choose the relevant CV for the strongest claim
        if c['claim_size']:
            # determine which size CV is "strongest"
            cvs = c['cv_size']
            relevant_cv = max(cv for cv in cvs if cv > 0 and c['F'] > cv)
        elif c['claim_power']:
            cvs = c['cv_power']
            relevant_cv = max(cv for cv in cvs if cv > 0 and c['F'] > cv)
        else:
            relevant_cv = float('nan')
        return [
            [
                f"  borderline (1 cell): F = {c['F']:.2f} "
                f"(95% CI lower bound {c['F_ci_low']:.2f}), strongest claim \"{claim}\""
            ],
            [
                f"     CI overlaps the relevant CV = {relevant_cv:.2f}: "
                f"claim is decision-uncertain at the conventional 95% level."
            ],
        ]
    # Multiple cells: summarize.
    worst = _worst_cell_borderline(cells)
    return [
        [
            f"  borderline ({n} cell{suffix}): worst-case F = {worst['F']:.2f} "
            f"(95% CI lower bound {worst['F_ci_low']:.2f})."
        ],
        [
            "     for these cells the asymptotic 95% CI for F overlaps the "
            "CV that supports the strongest claim — the strength claim is "
            "decision-uncertain at conventional levels."
        ],
    ]


def _near_degenerate_lines(cells: List[Dict[str, Any]]) -> List[List[str]]:
    """Build the near-degenerate footer line(s)."""
    n = len(cells)
    suffix = '' if n == 1 else 's'
    min_lambda = min(c['lambda'] for c in cells if np.isfinite(c['lambda']))
    return [
        [
            f"  near-degenerate ({n} cell{suffix}): smallest lambda = {min_lambda:.3f}"
        ],
        [
            "     F's denominator has lost most of its scale to cancellation; "
            "F's value is numerically unreliable here. Treat as flagging weak "
            "model separation rather than reporting a precise number."
        ],
    ]


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
    # F-stat reliability diagnostic (added in feat/f-reliability). Optional so
    # older pickled Progress instances stay constructible. ProblemResults
    # treats missing diagnostic fields as None and falls back to the legacy
    # display path.
    lambda_dmss_list: Optional[List[Any]] = None
    F_se_list: Optional[List[Any]] = None
    F_ci_low_list: Optional[List[Any]] = None
    F_ci_high_list: Optional[List[Any]] = None
    verdict_list: Optional[List[Any]] = None
    strongest_claim_size_list: Optional[List[Any]] = None
    strongest_claim_power_list: Optional[List[Any]] = None


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
        # v0.4 step 14d: propagate market_side from the Problem so __str__
        # can switch to labor-side terminology (markdown / MRP / wage) when
        # the Problem was constructed with market_side='labor'. Falls back
        # to 'product' for older pickles that predate the attribute.
        self._market_side: str = getattr(progress.problem, '_market_side', 'product')
        # F-stat reliability diagnostic (additive). See
        # MEMO_F_reliability_diagnostic_2026-04-28.md.
        self.lambda_dmss = progress.lambda_dmss_list
        self.F_se = progress.F_se_list
        self.F_ci_low = progress.F_ci_low_list
        self.F_ci_high = progress.F_ci_high_list
        self.verdict = progress.verdict_list
        self.strongest_claim_size = progress.strongest_claim_size_list
        self.strongest_claim_power = progress.strongest_claim_power_list

    def __str__(self) -> str:
        """Format results information as a string."""
        out = ""
        for i in range(len(self.TRV)):
            tmp = "\n\n".join([self._format_results_tables(i)])
            out = "\n\n".join([out, tmp])
        return out

    def _format_results_tables(self, j: int) -> str:
        """Formation information about the testing results as a string."""

        # F-stat reliability glyph: cells with verdict != robust get a marker
        # appended to their F value. Loaded once per call; falls back to ""
        # if the diagnostic was not computed (older pickles).
        verdict_arr = self.verdict[j] if getattr(self, 'verdict', None) is not None else None

        def _f_value_with_glyph(k: int, i: int) -> str:
            base = str(round(self.F[j][k, i], 1))
            if verdict_arr is None:
                return base
            v = verdict_arr[k, i]
            if v is None or v == 'robust':
                return base
            # Any non-robust verdict gets the warning glyph.
            return base + '⚠'

        # construct the data
        data: List[List[str]] = []
        number_models = len(self.markups)
        for k in range(number_models):
            rv_results = [round(self.TRV[j][k, i], 3) for i in range(number_models)]
            f_stat_results = [_f_value_with_glyph(k, i) for i in range(number_models)]
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
        # v0.4 step 14d: labor-side title banner. Product-side keeps the
        # pre-v0.4 title byte-for-byte ("Testing Results - Instruments z{0}");
        # labor-side gets a suffix so ``print(results)`` shows the
        # markdown / MRP / wage terminology even though the underlying
        # statistics are identical.
        if self._market_side == 'labor':
            title = (
                "Testing Results (labor: markdown / MRP / wage) - Instruments z{0}"
                .format(j)
            )
        else:
            title = "Testing Results - Instruments z{0}".format(j)

        # F-stat reliability footer (only on the last printed table).
        # Aggregates verdicts across all instrument sets and emits one
        # line per fragility type that fired, plus an "all robust" line
        # if nothing fired. Returns [] for the non-last tables.
        extra_notes: List[List[str]] = []
        if last_table:
            extra_notes = self._build_F_reliability_footer()

        return format_table(
            header, subheader, *data, title=title, include_notes=last_table,
            line_indices=[number_models, 2 * number_models + 1],
            extra_notes=extra_notes,
        )

    def _build_F_reliability_footer(self) -> List[List[str]]:
        """Build the F-stat reliability footer aggregated across all instrument sets.

        Returns a list of rows (each a list of strings, the first is the
        line text) suitable for ``format_table(extra_notes=...)``. Returns
        an empty list when the diagnostic was not computed (older pickles).
        Returns a single one-line "all robust" row when no cell is flagged.
        """
        if getattr(self, 'verdict', None) is None:
            return []

        # Walk all (j, k, i) cells, collect verdicts and per-cell numbers.
        flagged_borderline: List[dict] = []
        flagged_near_deg: List[dict] = []
        flagged_trivially: List[dict] = []
        max_rho2 = 0.0
        min_lambda = 1.0
        any_valid = False
        for j in range(len(self.verdict)):
            verdict_j = self.verdict[j]
            rho_j = np.asarray(self.rho[j])
            lambda_j = (
                np.asarray(self.lambda_dmss[j])
                if getattr(self, 'lambda_dmss', None) is not None
                else None
            )
            F_j = np.asarray(self.F[j])
            ci_low_j = (
                np.asarray(self.F_ci_low[j])
                if getattr(self, 'F_ci_low', None) is not None
                else None
            )
            cv_size_j = self.F_cv_size_list[j]
            cv_power_j = self.F_cv_power_list[j]
            claim_size_j = (
                self.strongest_claim_size[j]
                if getattr(self, 'strongest_claim_size', None) is not None
                else None
            )
            claim_power_j = (
                self.strongest_claim_power[j]
                if getattr(self, 'strongest_claim_power', None) is not None
                else None
            )
            M = verdict_j.shape[0]
            for k in range(M):
                for i in range(M):
                    if k >= i:
                        continue  # only upper triangle is computed
                    v = verdict_j[k, i]
                    if v is None:
                        continue
                    any_valid = True
                    rho_val = rho_j[k, i]
                    if np.isfinite(rho_val):
                        max_rho2 = max(max_rho2, rho_val ** 2)
                    if lambda_j is not None and np.isfinite(lambda_j[k, i]):
                        min_lambda = min(min_lambda, float(lambda_j[k, i]))
                    if v == 'robust':
                        continue
                    cell_info = {
                        'instrument_set': j,
                        'pair': (k, i),
                        'F': float(F_j[k, i]) if np.isfinite(F_j[k, i]) else float('nan'),
                        'F_ci_low': (
                            float(ci_low_j[k, i])
                            if ci_low_j is not None and np.isfinite(ci_low_j[k, i])
                            else float('nan')
                        ),
                        'lambda': (
                            float(lambda_j[k, i])
                            if lambda_j is not None and np.isfinite(lambda_j[k, i])
                            else float('nan')
                        ),
                        'claim_size': (
                            claim_size_j[k, i] if claim_size_j is not None else None
                        ),
                        'claim_power': (
                            claim_power_j[k, i] if claim_power_j is not None else None
                        ),
                        'cv_size': cv_size_j[k, i],
                        'cv_power': cv_power_j[k, i],
                    }
                    if v == 'borderline':
                        flagged_borderline.append(cell_info)
                    elif v == 'near-degenerate':
                        flagged_near_deg.append(cell_info)
                    elif v == 'trivially-degenerate':
                        flagged_trivially.append(cell_info)

        if not any_valid:
            return []

        any_flagged = bool(flagged_borderline or flagged_near_deg or flagged_trivially)
        rows: List[List[str]] = []

        if not any_flagged:
            rows.append([
                "F-stat reliability: all cells robust "
                f"(max rho^2 = {max_rho2:.2f}, min lambda = {min_lambda:.2f})."
            ])
            return rows

        # Header for the reliability section
        rows.append(["F-stat reliability:"])

        if flagged_borderline:
            rows.extend(_borderline_lines(flagged_borderline))
        if flagged_near_deg:
            rows.extend(_near_degenerate_lines(flagged_near_deg))
        if flagged_trivially:
            n = len(flagged_trivially)
            rows.append([
                f"  triv-deg ({n} cell{'s' if n != 1 else ''}): models produce identical markups; F is undefined."
            ])
        rows.append(["  See: results.F_reliability_summary() for per-cell detail."])
        return rows

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
        - ``MCS_pvalue_model_i``: MCS p-value for model ``i`` in this
          instrument set (a function of ``i`` only; duplicated across
          ``j`` rows to keep the long form self-contained).

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
                        'MCS_pvalue_model_i': float(mcs_vec[i]),
                    })
        return pd.DataFrame.from_records(records, columns=[
            'instrument_set', 'instrument_set_label',
            'model_i', 'model_j', 'model_i_label', 'model_j_label',
            'TRV', 'F', 'MCS_pvalue_model_i',
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

        if not (0.0 < alpha < 1.0):
            raise ValueError(
                f"Expected alpha to lie strictly in the open interval (0, 1). "
                f"Received alpha={alpha!r}. "
                f"Fix: pass a two-sided significance level such as 0.01, 0.05, "
                f"or 0.10."
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

    def F_reliability_summary(self) -> 'pd.DataFrame':
        """Return per-cell F-stat reliability diagnostics.

        One row per (instrument set, model_i, model_j) with model_i < model_j.
        Columns:

        - ``F``, ``rho_squared``: existing test statistic and DMSS rho².
        - ``lambda_dmss``: numerical-cancellation depth, ``((σ_0+σ_1)² −
          4σ_2²) / (σ_0+σ_1)²``. Below 0.05 the F value is unreliable.
        - ``F_se``, ``F_ci_low``, ``F_ci_high``: asymptotic SE and 95% CI for
          the population F under the paper's noncentral chi-squared
          distribution at the implied noncentrality.
        - ``strongest_claim_size``, ``strongest_claim_power``: human-readable
          labels of the strongest size and power claims F supports
          (e.g. ``"worst-case size <= 10%"``); ``None`` when F clears no CV.
        - ``verdict``: one of ``"robust"``, ``"borderline"``,
          ``"near-degenerate"``, or ``"trivially-degenerate"``.

        See ``MEMO_F_reliability_diagnostic_2026-04-28.md`` for design and
        calibration. The diagnostic is independent of and complementary to
        the test's Type I error guarantee under the worst-case null
        (DMSS Proposition 4) — it is a precision-of-the-point-estimate
        question about the observed F.

        Returns
        -------
        pd.DataFrame
            Long-form frame with ``L * M * (M - 1) / 2`` rows.
        """
        import pandas as pd

        L = self._number_of_instrument_sets()
        M = self._number_of_models()
        iv_labels = self._instrument_set_labels()
        model_labels = self._model_labels()

        records: List[dict[str, Any]] = []
        for j in range(L):
            f_mat = np.asarray(self.F[j])
            rho_mat = np.asarray(self.rho[j])
            lam_mat = np.asarray(self.lambda_dmss[j]) if self.lambda_dmss is not None else None
            se_mat = np.asarray(self.F_se[j]) if self.F_se is not None else None
            ci_lo_mat = np.asarray(self.F_ci_low[j]) if self.F_ci_low is not None else None
            ci_hi_mat = np.asarray(self.F_ci_high[j]) if self.F_ci_high is not None else None
            verdict_mat = self.verdict[j] if self.verdict is not None else None
            claim_size_mat = self.strongest_claim_size[j] if self.strongest_claim_size is not None else None
            claim_power_mat = self.strongest_claim_power[j] if self.strongest_claim_power is not None else None
            for i in range(M):
                for k in range(i + 1, M):
                    rho_ik = float(rho_mat[i, k])
                    rho2_ik = rho_ik ** 2 if np.isfinite(rho_ik) else float('nan')
                    records.append({
                        'instrument_set': j,
                        'instrument_set_label': iv_labels[j],
                        'model_i': i,
                        'model_j': k,
                        'model_i_label': model_labels[i],
                        'model_j_label': model_labels[k],
                        'F': float(f_mat[i, k]),
                        'rho_squared': rho2_ik,
                        'lambda_dmss': (
                            float(lam_mat[i, k]) if lam_mat is not None else float('nan')
                        ),
                        'F_se': (
                            float(se_mat[i, k]) if se_mat is not None else float('nan')
                        ),
                        'F_ci_low': (
                            float(ci_lo_mat[i, k]) if ci_lo_mat is not None else float('nan')
                        ),
                        'F_ci_high': (
                            float(ci_hi_mat[i, k]) if ci_hi_mat is not None else float('nan')
                        ),
                        'strongest_claim_size': (
                            claim_size_mat[i, k] if claim_size_mat is not None else None
                        ),
                        'strongest_claim_power': (
                            claim_power_mat[i, k] if claim_power_mat is not None else None
                        ),
                        'verdict': (
                            verdict_mat[i, k] if verdict_mat is not None else None
                        ),
                    })
        columns = [
            'instrument_set', 'instrument_set_label',
            'model_i', 'model_j', 'model_i_label', 'model_j_label',
            'F', 'rho_squared', 'lambda_dmss',
            'F_se', 'F_ci_low', 'F_ci_high',
            'strongest_claim_size', 'strongest_claim_power', 'verdict',
        ]
        return pd.DataFrame.from_records(records, columns=columns)

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
        digits = int(options.digits)
        tex: str = summary.to_latex(
            index=False,
            escape=False,
            caption=caption,
            label=label,
            float_format=f"%.{digits}g",
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

    # ------------------------------------------------------------------
    # v0.4 OQ 15: Dearing pass-through diagnostics
    # ------------------------------------------------------------------

    # Supported metrics for :meth:`passthrough_comparison`. Centralised
    # so the error-message and dispatch branches cannot drift.
    _PASSTHROUGH_METRICS: "tuple[str, ...]" = (
        'frobenius', 'offdiag_frobenius', 'max_abs',
    )

    def _passthrough_distance(
        self,
        difference: NDArray[Any],
        metric: str,
    ) -> float:
        """Reduce a pairwise pass-through difference matrix to a scalar.

        Centralises the metric dispatch so :meth:`passthrough_comparison`
        has a single place to look when a new metric is added. The three
        metrics are Dearing-style distinguishability diagnostics; see the
        public method docstring for the mapping to the paper's Remark 4.
        """
        if metric == 'frobenius':
            return float(np.linalg.norm(difference, ord='fro'))
        if metric == 'offdiag_frobenius':
            off = difference - np.diag(np.diag(difference))
            return float(np.linalg.norm(off, ord='fro'))
        if metric == 'max_abs':
            return float(np.max(np.abs(difference)))
        # Not reachable: validated in the public entry point.
        raise ValidationError(  # pragma: no cover - defensive
            f"Internal: unknown metric {metric!r}."
        )

    def passthrough_matrix(
        self,
        model_index: int,
        market_id: Optional[Hashable] = None,
    ) -> Union[NDArray[Any], Dict[Hashable, NDArray[Any]]]:
        """Return the Villas-Boas pass-through matrix for one candidate model.

        Thin wrapper over :func:`pyRVtest.build_passthrough` exposed on
        :class:`ProblemResults` so users do not need to remember a
        separate top-level import. Input validation and the non-vertical
        error path are delegated to ``build_passthrough``.

        Parameters
        ----------
        model_index : int
            Index into ``self.problem._models``. Must refer to a
            :class:`pyRVtest.Vertical` entry (build_passthrough raises
            ``ValueError`` otherwise).
        market_id : hashable, optional
            If ``None`` (default), returns ``{market_id: matrix}`` across
            every market. Otherwise returns the single ``(J_t, J_t)``
            matrix for that market.

        Returns
        -------
        ndarray or dict
            Pass-through matrix per Villas-Boas (2007). Shape
            ``(J_t, J_t)`` for a scalar ``market_id``, or
            ``{market_id: (J_t, J_t)}`` across all markets.

        See Also
        --------
        pyRVtest.build_passthrough : standalone helper the method wraps.
        ProblemResults.passthrough_comparison : pairwise diagnostic.
        """
        result = build_passthrough(
            self.problem, model_index=model_index, market_id=market_id,
        )
        return result

    def passthrough_comparison(
        self,
        metric: str = 'frobenius',
        market_id: Optional[Hashable] = None,
    ) -> 'pd.DataFrame':
        r"""Pairwise pass-through distances per Dearing et al. (2026) Remark 4.

        For every unordered pair of candidate models :math:`(m, m')` with
        :math:`m < m'`, computes a scalar distance between the Villas-Boas
        pass-through matrices :math:`P_m` and :math:`P_{m'}` market by
        market. Dearing et al. (2026) show that distinguishability with
        pass-through-based instruments hinges on whether two models have
        different pass-through matrices (Remark 3) or, more sharply,
        different off-diagonal structure (Remark 4). Near-zero distances
        flag a pair that is hard to separate with pass-through
        instruments; large distances flag a pair that should be
        identifiable.

        Parameters
        ----------
        metric : str, optional
            Which scalar reduction of :math:`P_m - P_{m'}` to report.

            - ``'frobenius'`` (default): Frobenius norm of the full
              difference, :math:`\lVert P_m - P_{m'} \rVert_F`.
            - ``'offdiag_frobenius'``: Frobenius norm restricted to the
              off-diagonal entries. Implements Remark 4's
              distinguishability condition, which is invariant to
              diagonal-only differences.
            - ``'max_abs'``: element-wise maximum absolute difference.
        market_id : hashable, optional
            If ``None`` (default), returns one row per ``(market,
            model_i, model_j)``. Otherwise filters to that single market;
            the returned frame has exactly ``M * (M - 1) / 2`` rows
            (one per unordered model pair).

        Returns
        -------
        pd.DataFrame
            Columns: ``market_id``, ``model_i``, ``model_j``,
            ``model_i_label``, ``model_j_label``, ``distance``,
            ``metric``. The chosen ``metric`` is also recorded on
            ``frame.attrs['metric']`` for parity with the
            ``summary_df`` ``attrs['alpha']`` convention.

        Raises
        ------
        ValidationError
            If ``metric`` is not one of the three supported strings.
            (``ValidationError`` subclasses ``ValueError`` — existing
            ``pytest.raises(ValueError, ...)`` assertions continue to
            work.)
        NotImplementedError
            If ANY candidate model in the problem is not a
            :class:`pyRVtest.Vertical`. Computing pass-through for
            Bertrand / Cournot / RuleOfThumb / ConstantMarkup /
            PerfectCompetition requires per-model derivative formulas
            that land in v0.5. Workaround: restrict the
            :class:`pyRVtest.Problem` to Vertical candidates only, or
            skip this diagnostic.

        Notes
        -----
        No aggregation across markets is performed: users can
        ``df.groupby(['model_i', 'model_j']).mean()`` for a summary
        across markets, or filter to a specific market via the
        ``market_id`` argument.

        See Also
        --------
        pyRVtest.build_passthrough : the standalone pass-through helper
            this method composes over.
        ProblemResults.passthrough_matrix : the per-model counterpart.

        References
        ----------
        Dearing, A., L. Magnolfi, D. Quint, C. Sullivan, and J.
        Waldfogel (2026). "Falsifying Models of Firm Conduct with
        Tax Instruments." Remark 4.
        """
        import pandas as pd

        # --- 1. Validate metric. ---
        if metric not in self._PASSTHROUGH_METRICS:
            raise ValidationError(
                f"Expected metric to be one of "
                f"{list(self._PASSTHROUGH_METRICS)!r}. "
                f"Received metric={metric!r}. "
                f"Fix: pass 'frobenius' for the full-matrix norm, "
                f"'offdiag_frobenius' for Dearing Remark 4's "
                f"off-diagonal condition, or 'max_abs' for the "
                f"element-wise maximum absolute difference."
            )

        # --- 2. Check every candidate model is Vertical. ---
        candidate_models: Sequence[Any] = self.problem._models
        for i, model in enumerate(candidate_models):
            if not isinstance(model, Vertical):
                raise NotImplementedError(
                    "Pass-through comparison currently requires all "
                    "candidate models to be Vertical. "
                    f"Received model index {i} of type "
                    f"{type(model).__name__}, which has no closed-form "
                    "pass-through in pyRVtest v0.4. Computing "
                    "pass-through for Bertrand / Cournot / RuleOfThumb / "
                    "ConstantMarkup is deferred to v0.5. Workaround: "
                    "restrict the Problem to only Vertical candidate "
                    "models, or skip this diagnostic."
                )

        # --- 3. Validate market_id (if provided) up front. ---
        #   build_passthrough also validates per call; pre-validate here
        #   so an invalid id surfaces once, not M times. Use an
        #   element-wise equality check so market ids stored as numpy
        #   scalars compare correctly against Python scalars.
        unique_market_ids = np.asarray(self.problem.unique_market_ids)
        if market_id is not None:
            if not np.any(unique_market_ids == market_id):
                raise ValidationError(
                    f"Expected market_id to appear in "
                    f"problem.unique_market_ids. Received "
                    f"market_id={market_id!r}, which is not in "
                    f"problem.unique_market_ids="
                    f"{list(unique_market_ids)}. Fix: pass a market id "
                    f"from problem.unique_market_ids, or omit "
                    f"market_id to compute pass-through for all markets."
                )
            markets_to_iterate: List[Hashable] = [market_id]
        else:
            markets_to_iterate = list(unique_market_ids.tolist())

        # --- 4. Compute pass-through per model, caching across pairs. ---
        model_labels = self._model_labels()
        n_models = self._number_of_models()
        # ``build_passthrough`` with ``market_id=None`` returns a dict
        # {market_id: matrix}; with a scalar it returns an ndarray. We
        # always use the dict form here and index by market id below so
        # the pair loop has a single code path.
        cached: Dict[int, Dict[Hashable, NDArray[Any]]] = {}
        for m in range(n_models):
            per_market = build_passthrough(self.problem, m, market_id=None)
            # ``build_passthrough(..., market_id=None)`` always returns
            # a dict; narrow the type for mypy.
            assert isinstance(per_market, dict)
            cached[m] = per_market

        # --- 5. Build the long-form records. ---
        records: List[dict[str, Any]] = []
        for t in markets_to_iterate:
            for i in range(n_models):
                for k in range(i + 1, n_models):
                    P_i = cached[i][t]
                    P_k = cached[k][t]
                    difference = np.asarray(P_i) - np.asarray(P_k)
                    distance = self._passthrough_distance(difference, metric)
                    records.append({
                        'market_id': t,
                        'model_i': i,
                        'model_j': k,
                        'model_i_label': model_labels[i],
                        'model_j_label': model_labels[k],
                        'distance': distance,
                        'metric': metric,
                    })

        columns = [
            'market_id', 'model_i', 'model_j',
            'model_i_label', 'model_j_label',
            'distance', 'metric',
        ]
        frame = pd.DataFrame.from_records(records, columns=columns)
        frame.attrs['metric'] = metric
        return frame


__all__ = ['Progress', 'ProblemResults']
