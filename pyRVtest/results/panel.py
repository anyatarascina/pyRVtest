"""Multi-problem aggregation of :class:`ProblemResults`.

v0.4 step 10: ``PanelResults`` aggregates a panel of
:class:`pyRVtest.ProblemResults` — one per market-year, subsample, or
other exchangeable partition — into a single panel-level view. The
motivating use case is the AFSSZ scalable conduct-testing workflow (step
16) where the same RV test is run independently on 910 market-years and
the researcher wants pair-level rejection rates rather than 910 separate
tables.

Design notes
------------

- **Composition, not inheritance.** ``PanelResults`` only calls
  :meth:`ProblemResults.to_dataframe` and
  :meth:`ProblemResults.summary_df` on its children. It does not reach
  into ``TRV``, ``F``, or ``MCS_pvalues`` arrays directly. This keeps
  the class independent of ``ProblemResults`` internals and lets both
  classes evolve.
- **Keys are arbitrary hashables.** ``results={(market_id, year): pr,
  ...}`` is the canonical shape; plain strings and integers also work.
  Iteration order follows insertion order.
- **Validation.** The constructor rejects an empty panel and a panel
  whose children disagree on the number of candidate models. Instrument
  sets can differ across children (one market-year may use a different
  instrument set than another), but the model roster must line up so
  that rejection-rate aggregation is meaningful.

See ``tests/test_panel_results.py`` for the supported usage patterns.
"""

from __future__ import annotations

from collections.abc import Hashable
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Tuple, TYPE_CHECKING, Union

from .. import options
from ._format import _dataframe_to_github_markdown
from .results import ProblemResults


_RosterSignature = Tuple[Tuple[str, str, str, str, str], ...]


def _panel_roster_signature(pr: ProblemResults) -> Optional[_RosterSignature]:
    """Canonical roster signature for comparing panel children.

    Returns a tuple of per-model 5-tuples capturing the downstream-model
    name, upstream-model name, downstream ownership column, upstream
    ownership column, and user-supplied-markups column name. Two
    :class:`ProblemResults` share a roster (in the ordered,
    position-sensitive sense that :meth:`PanelResults.rejection_rates`
    and :meth:`PanelResults.summary_df` rely on) iff their signatures
    are equal.

    Returns ``None`` when ``pr.problem.models`` is not available or
    lacks the expected fields (defensive against older pickles whose
    ``Models`` recarray shape predates the current schema). In that
    case :class:`PanelResults` falls back to the count-only check.
    """
    problem = getattr(pr, 'problem', None)
    if problem is None:
        return None
    models = getattr(problem, 'models', None)
    if models is None:
        return None
    M = int(len(pr.markups))
    try:
        downstream = models['models_downstream']
        upstream = models['models_upstream']
        firm_ids_down = models['firm_ids_downstream']
        firm_ids_up = models['firm_ids_upstream']
        supplied = models['user_supplied_markups_name']
    except (IndexError, ValueError, KeyError, TypeError):
        return None
    signature: List[Tuple[str, str, str, str, str]] = []
    for i in range(M):
        try:
            signature.append((
                str(downstream[i]),
                str(upstream[i]),
                str(firm_ids_down[i]),
                str(firm_ids_up[i]),
                str(supplied[i]),
            ))
        except Exception:  # pragma: no cover — defensive fallback
            return None
    return tuple(signature)

if TYPE_CHECKING:
    import pandas as pd


class PanelResults:
    """Aggregated results across a panel of :class:`ProblemResults`.

    Parameters
    ----------
    results : mapping
        Mapping from panel key (any hashable — e.g. ``(market_id,
        year)``, a string, an integer) to a solved
        :class:`pyRVtest.ProblemResults` instance. Must be non-empty
        and all children must share the same number of candidate
        models.

    Raises
    ------
    TypeError
        If ``results`` is not a mapping or any value is not a
        :class:`ProblemResults`.
    ValueError
        If ``results`` is empty, or children disagree on the number of
        candidate models.

    Notes
    -----
    ``PanelResults`` composes over :meth:`ProblemResults.to_dataframe`
    and :meth:`ProblemResults.summary_df`; it does not touch the raw
    ``TRV`` / ``F`` / ``MCS_pvalues`` arrays.
    """

    _results: Dict[Hashable, ProblemResults]

    def __init__(self, results: Mapping[Hashable, ProblemResults]) -> None:
        if not isinstance(results, Mapping):
            raise TypeError(
                f"Expected results to be a mapping (dict-like) from panel keys "
                f"to ProblemResults. "
                f"Received {type(results).__name__}. "
                f"Fix: pass a mapping {{key: ProblemResults}}; for a list, "
                f"enumerate it into a dict first."
            )
        if len(results) == 0:
            raise ValueError(
                "Expected the results mapping to contain at least one "
                "ProblemResults. "
                "Received an empty mapping. "
                "Fix: supply at least one (key, ProblemResults) entry."
            )
        # Preserve insertion order.
        stored: Dict[Hashable, ProblemResults] = {}
        expected_n_models: Optional[int] = None
        expected_key: Optional[Hashable] = None
        expected_signature: Optional[_RosterSignature] = None
        for key, pr in results.items():
            if not isinstance(pr, ProblemResults):
                raise TypeError(
                    f"Expected every value in the results mapping to be a "
                    f"ProblemResults instance. "
                    f"Received results[{key!r}] of type {type(pr).__name__}. "
                    f"Fix: call problem.solve() to obtain a ProblemResults, "
                    f"then store it under this key."
                )
            n_models = int(len(pr.markups))
            signature = _panel_roster_signature(pr)
            if expected_n_models is None:
                expected_n_models = n_models
                expected_key = key
                expected_signature = signature
            elif n_models != expected_n_models:
                raise ValueError(
                    f"Mismatched model sets: expected all panel children to "
                    f"share the same candidate model roster (rejection-rate "
                    f"aggregation is only well defined when the set of models "
                    f"is fixed). "
                    f"Received key {expected_key!r} with {expected_n_models} "
                    f"models and key {key!r} with {n_models}. "
                    f"Fix: rebuild the panel so every child is solved with the "
                    f"same list of candidate models."
                )
            elif expected_signature is not None and signature is not None \
                    and signature != expected_signature:
                # v0.4.0rc1 follow-up (audit B1 / Lorenzo P1 item 4): same
                # count but different roster identity would otherwise be a
                # silent mislabel at ``summary_df`` / ``rejection_rates``
                # time, because those methods pull labels from the first
                # child. Raise here so the error surfaces at construction.
                diff_i = next(
                    (i for i, (a, b) in enumerate(zip(expected_signature, signature))
                     if a != b),
                    None,
                )
                if diff_i is None:  # pragma: no cover — unreachable given != check
                    diff_i = min(len(expected_signature), len(signature))
                expected_entry = expected_signature[diff_i] if diff_i < len(expected_signature) else None
                received_entry = signature[diff_i] if diff_i < len(signature) else None
                raise ValueError(
                    f"Mismatched model sets: expected all panel children to "
                    f"share the same ordered candidate-model roster "
                    f"(rejection-rate aggregation pulls labels from the first "
                    f"child, so positions must match across children to avoid "
                    f"silent mislabel). "
                    f"Received key {expected_key!r} and key {key!r} with the "
                    f"same number of models but divergent identity at "
                    f"position {diff_i}: "
                    f"{expected_key!r} -> {expected_entry!r}, "
                    f"{key!r} -> {received_entry!r}. "
                    f"Full signatures: "
                    f"{expected_key!r} -> {expected_signature!r}; "
                    f"{key!r} -> {signature!r}. "
                    f"Fix: rebuild the panel so every child is solved with "
                    f"the same ordered list of candidate models."
                )
            stored[key] = pr
        self._results = stored

    # ------------------------------------------------------------------
    # Mapping-like interface
    # ------------------------------------------------------------------

    def keys(self) -> List[Hashable]:
        """Return the panel keys in insertion order.

        Returns
        -------
        list of hashable
            The keys passed to ``__init__``, in insertion order.
        """
        return list(self._results.keys())

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def __getitem__(self, key: Hashable) -> ProblemResults:
        return self._results[key]

    def __contains__(self, key: object) -> bool:
        return key in self._results

    # ------------------------------------------------------------------
    # Long-form export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> 'pd.DataFrame':
        """Return a long-form DataFrame of pairwise results across the panel.

        Each child's :meth:`ProblemResults.to_dataframe` is called and
        the results are concatenated vertically with a ``panel_key``
        column prepended. Existing columns (``instrument_set``,
        ``model_i``, ``model_j``, ``TRV``, ``F``, ``MCS_pvalue_model_i``, and
        their labels) are passed through unchanged.

        Returns
        -------
        pd.DataFrame
            Long-form frame with one row per (panel key, instrument set,
            ordered model pair). If the panel is empty of off-diagonal
            pairs (e.g. M=1), the frame has zero rows but keeps the
            expected columns.
        """
        import pandas as pd

        frames: List[pd.DataFrame] = []
        for key, pr in self._results.items():
            sub = pr.to_dataframe().copy()
            # Insert panel_key as the leftmost column for readability.
            sub.insert(0, 'panel_key', [key] * len(sub))
            frames.append(sub)
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Rejection rates and wide-form summary
    # ------------------------------------------------------------------

    def rejection_rates(self, alpha: float = 0.05) -> 'pd.DataFrame':
        """Return per-pair rejection rates across panel keys.

        For each (instrument set, unordered model pair), compute the
        share of panel keys whose :meth:`ProblemResults.summary_df`
        rejection flag is ``True`` at the requested ``alpha``.

        Parameters
        ----------
        alpha : float, optional
            Two-sided significance level forwarded to
            :meth:`ProblemResults.summary_df`. Defaults to ``0.05``.
            Must lie strictly between 0 and 1.

        Returns
        -------
        pd.DataFrame
            One row per (instrument_set, model_i, model_j) with columns
            ``rejection_rate`` (share of rejecting panel keys) and
            ``n_keys`` (count of panel keys contributing to this row;
            may be less than ``len(self)`` if some children lack the
            instrument set). Model-pair ordering uses ``model_i <
            model_j``.

        Raises
        ------
        ValueError
            If ``alpha`` is not in ``(0, 1)``. Raised by
            :meth:`ProblemResults.summary_df` on the first child; other
            children are not consulted.
        """
        import pandas as pd

        summaries: List[pd.DataFrame] = []
        for pr in self._results.values():
            summaries.append(pr.summary_df(alpha=alpha))
        if not summaries:
            # Unreachable — __init__ rejects empty panels — but keep the
            # branch for mypy and for defensive robustness.
            result = pd.DataFrame(
                columns=['instrument_set', 'model_i', 'model_j',
                         'rejection_rate', 'n_keys']
            )
            result.attrs['alpha'] = alpha
            return result

        stacked = pd.concat(summaries, ignore_index=True)
        group_cols = ['instrument_set', 'model_i', 'model_j']
        agg = stacked.groupby(group_cols, as_index=False).agg(
            rejection_rate=('reject', 'mean'),
            n_keys=('reject', 'size'),
        )
        # Promote rejection_rate to plain float and n_keys to int.
        agg['rejection_rate'] = agg['rejection_rate'].astype(float)
        agg['n_keys'] = agg['n_keys'].astype(int)
        agg.attrs['alpha'] = alpha
        return agg

    def summary_df(self, alpha: float = 0.05) -> 'pd.DataFrame':
        """Return a wide-form rejection-rate summary across the panel.

        One row per (instrument_set, unordered model pair), with
        ``rejection_rate`` at the requested ``alpha``, ``n_keys``
        contributing, and the first child's instrument-set and model
        labels where available (labels are assumed homogeneous across
        the panel — a standard feature of conduct-testing panels).

        Parameters
        ----------
        alpha : float, optional
            Two-sided significance level forwarded to
            :meth:`ProblemResults.summary_df`. Defaults to ``0.05``.

        Returns
        -------
        pd.DataFrame
            Frame with columns ``instrument_set``,
            ``instrument_set_label``, ``model_i``, ``model_j``,
            ``model_i_label``, ``model_j_label``, ``rejection_rate``,
            and ``n_keys``. The ``rejection_rate`` column name does not
            encode ``alpha`` so downstream code can sort on it
            generically; ``alpha`` is recoverable from the caller side.
        """
        rates = self.rejection_rates(alpha=alpha)
        # Pull label columns from the first child's summary_df (labels
        # are a property of the candidate-model roster, which __init__
        # has already validated is homogeneous in count).
        first_child = next(iter(self._results.values()))
        first_summary = first_child.summary_df(alpha=alpha)
        label_cols = [
            'instrument_set', 'instrument_set_label',
            'model_i', 'model_j', 'model_i_label', 'model_j_label',
        ]
        labels = first_summary[label_cols].drop_duplicates(
            subset=['instrument_set', 'model_i', 'model_j']
        )
        merged = rates.merge(
            labels, on=['instrument_set', 'model_i', 'model_j'], how='left'
        )
        result = merged[[
            'instrument_set', 'instrument_set_label',
            'model_i', 'model_j', 'model_i_label', 'model_j_label',
            'rejection_rate', 'n_keys',
        ]]
        result.attrs['alpha'] = alpha
        return result

    # ------------------------------------------------------------------
    # LaTeX and markdown renderings
    # ------------------------------------------------------------------

    def to_latex(
        self,
        path: Optional[Union[str, Path]] = None,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        alpha: float = 0.05,
    ) -> Optional[str]:
        r"""Render the panel :meth:`summary_df` as a LaTeX ``tabular``.

        Wraps :func:`pandas.DataFrame.to_latex` with ``escape=False`` so
        math symbols in model or instrument labels pass through
        unescaped. When ``caption`` or ``label`` is supplied the output
        is wrapped in a floating ``table`` environment.

        Parameters
        ----------
        path : str or Path, optional
            If given, the LaTeX string is written to this path and the
            method returns ``None``. Otherwise (default) the string is
            returned.
        caption : str, optional
            LaTeX caption. Forwarded to
            :func:`pandas.DataFrame.to_latex`.
        label : str, optional
            LaTeX label (e.g. ``"tab:panel_rejection"``). Forwarded to
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
        """Render the panel :meth:`summary_df` as a markdown table.

        Uses the same pipe-table renderer as
        :meth:`ProblemResults.to_markdown` so output is usable in any
        GitHub-style markdown viewer without the optional ``tabulate``
        dependency.

        Parameters
        ----------
        path : str or Path, optional
            If given, the markdown string is written to this path and
            the method returns ``None``. Otherwise (default) the string
            is returned.
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
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self._results)
        # Number of candidate models is guaranteed homogeneous by
        # __init__. Pull from the first child.
        first = next(iter(self._results.values()))
        m = int(len(first.markups))
        return f"PanelResults(n_keys={n}, n_models={m})"


__all__ = ['PanelResults']
