"""Reverse adapter: ModelFormulation -> ConductModel / Vertical (v0.4 step 5b').

After step 5b', the internal pipeline (Models recarray construction,
Problem bookkeeping, _compute_markups) reads from ``ConductModel`` /
``Vertical`` instances as the canonical intermediate representation.
Legacy ``ModelFormulation(model_downstream='bertrand', ...)`` inputs
are converted to the equivalent class-based form via
``from_model_formulation`` at ``Problem.__init__`` before anything
downstream runs. The forward direction (``to_model_formulations``) is
no longer needed and has been removed.

Step 5c keeps ``ModelFormulation`` working as a public user-facing
alias; internally, the alias is always translated to classes via
``from_model_formulation``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from ..formulation import ModelFormulation
from .base import ConductModel
from .custom import CustomConductModel
from .mixed import MixCournotBertrand
from .standard import Bertrand, Cournot, Monopoly, PerfectCompetition
from .vertical import Vertical


__all__ = ['from_model_formulation', 'from_model_formulations']


def from_model_formulation(
        mf: ModelFormulation,
) -> Union[ConductModel, Vertical]:
    """Translate a legacy ``ModelFormulation`` to a ``ConductModel`` or ``Vertical``.

    If the formulation has ``model_upstream`` set, the result is
    ``Vertical(downstream=..., upstream=..., ...)``. Otherwise it's a
    single ``ConductModel`` subclass.
    """
    if mf._model_upstream is not None:
        # ``model_downstream`` is guaranteed non-None in the Vertical branch because
        # ``ModelFormulation.__init__`` requires either ``model_downstream`` or
        # ``user_supplied_markups``, and vertical models always set the downstream.
        assert mf._model_downstream is not None
        down = _conduct_from_string(
            mf._model_downstream,
            ownership=mf._ownership_downstream,
            kappa_specification=mf._kappa_specification_downstream,
            mix_flag=mf._mix_flag,
            custom_model_specification=mf._custom_model_specification,
        )
        up = _conduct_from_string(
            mf._model_upstream,
            ownership=mf._ownership_upstream,
            kappa_specification=mf._kappa_specification_upstream,
        )
        return Vertical(
            downstream=down,
            upstream=up,
            vertical_integration=mf._vertical_integration,
            unit_tax=mf._unit_tax,
            advalorem_tax=mf._advalorem_tax,
            advalorem_payer=mf._advalorem_payer,
            cost_scaling=mf._cost_scaling,
            user_supplied_markups=mf._user_supplied_markups,
        )
    # Simple (non-vertical) case: all config lives on the one class.
    # ``model_downstream`` may be ``None`` here only when the caller relied on
    # ``user_supplied_markups`` alone (a legacy code path we reject in
    # ``_conduct_from_string`` with a clear error).
    assert mf._model_downstream is not None, (
        "ModelFormulation without model_downstream cannot be translated "
        "to a ConductModel (use an explicit model_downstream)."
    )
    return _conduct_from_string(
        mf._model_downstream,
        ownership=mf._ownership_downstream,
        kappa_specification=mf._kappa_specification_downstream,
        mix_flag=mf._mix_flag,
        custom_model_specification=mf._custom_model_specification,
        unit_tax=mf._unit_tax,
        advalorem_tax=mf._advalorem_tax,
        advalorem_payer=mf._advalorem_payer,
        cost_scaling=mf._cost_scaling,
        vertical_integration=mf._vertical_integration,
        user_supplied_markups=mf._user_supplied_markups,
    )


def from_model_formulations(
        mfs: Sequence[ModelFormulation],
) -> List[Union[ConductModel, Vertical]]:
    """Translate a list of ModelFormulations. Raises on non-ModelFormulation entries."""
    out: List[Union[ConductModel, Vertical]] = []
    for i, mf in enumerate(mfs):
        if not isinstance(mf, ModelFormulation):
            raise TypeError(
                f"Expected model_formulations[{i}] to be a ModelFormulation "
                f"instance. "
                f"Received {type(mf).__name__}. "
                f"Fix: wrap each spec in ModelFormulation(...), or migrate to "
                f"the preferred models=[Bertrand(...), ...] API."
            )
        out.append(from_model_formulation(mf))
    return out


def _conduct_from_string(
        model_str: str,
        ownership: Optional[str] = None,
        kappa_specification: Optional[Any] = None,
        mix_flag: Optional[str] = None,
        custom_model_specification: Optional[Dict[str, Any]] = None,
        **extra_config: Any,
) -> ConductModel:
    """Map a conduct-type string to the corresponding class instance.

    ``extra_config`` holds the tax / vertical_integration / user-supplied
    fields that live on the simple (non-vertical) class; for the
    downstream/upstream tiers of a ``Vertical`` these must be left off
    (``Vertical`` owns them). Callers enforce this via which fields they
    pass.
    """
    if model_str == 'bertrand':
        return Bertrand(
            ownership=ownership,
            kappa_specification=kappa_specification,
            **extra_config,
        )
    if model_str == 'cournot':
        return Cournot(
            ownership=ownership,
            kappa_specification=kappa_specification,
            **extra_config,
        )
    if model_str == 'monopoly':
        return Monopoly(
            ownership=ownership,
            kappa_specification=kappa_specification,
            **extra_config,
        )
    if model_str == 'perfect_competition':
        # PerfectCompetition has no ownership / kappa concept. Still passes
        # through the config fields (taxes, vi, user_supplied) for consistency.
        return PerfectCompetition(**extra_config)
    if model_str == 'mix_cournot_bertrand':
        return MixCournotBertrand(
            mix_flag=mix_flag,
            ownership=ownership,
            kappa_specification=kappa_specification,
            **extra_config,
        )
    if model_str == 'other':
        if custom_model_specification is None:
            raise TypeError(
                "Expected custom_model_specification when ModelFormulation has "
                "model_downstream='other' (custom markup formulas need a callable). "
                "Received custom_model_specification=None. "
                "Fix: pass custom_model_specification={'<name>': <callable>}."
            )
        if not isinstance(custom_model_specification, dict) or not custom_model_specification:
            raise TypeError(
                f"Expected custom_model_specification to be a non-empty dict "
                f"mapping a name to a callable. "
                f"Received {type(custom_model_specification).__name__} "
                f"(empty={not custom_model_specification}). "
                f"Fix: pass {{'<name>': <callable>}}."
            )
        name, fn = next(iter(custom_model_specification.items()))
        return CustomConductModel(
            markup_fn=fn,
            ownership=ownership,
            name=name,
            **extra_config,
        )
    if model_str is None:
        raise TypeError(
            "Expected ModelFormulation to specify a non-None model_downstream "
            "string for translation to a ConductModel. "
            "Received model_downstream=None (historically this relied on "
            "user_supplied_markups alone). "
            "Fix: pass model_downstream='bertrand' (or similar) explicitly, or "
            "use the class-based API directly (pyRVtest.Bertrand, etc.)."
        )
    raise TypeError(
        f"Expected model_downstream to be one of the known conduct strings "
        f"('bertrand', 'cournot', 'monopoly', 'perfect_competition', "
        f"'mix_cournot_bertrand', 'other'). "
        f"Received {model_str!r}. "
        f"Fix: pick a supported conduct string."
    )
