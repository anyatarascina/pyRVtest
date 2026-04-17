"""Convert new ConductModel / Vertical instances to legacy ModelFormulation.

v0.4 step 5b. The rest of the pyRVtest pipeline (Models recarray
construction, _compute_markups, compute_demand_adjustment) still reads
from the legacy string/column fields. This adapter takes a user-supplied
list of ``ConductModel`` / ``Vertical`` instances (the new API) and
returns an equivalent list of ``ModelFormulation`` instances (the
legacy API). The downstream code doesn't need to know which path the
user took.

Step 5c goes the other direction: ``ModelFormulation.__init__`` becomes
a deprecation alias that constructs the corresponding class. The two
directions keep both APIs fully round-trippable during the deprecation
window.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Union

from ..formulation import ModelFormulation
from .base import ConductModel
from .custom import CustomConductModel
from .mixed import MixCournotBertrand
from .vertical import Vertical


__all__ = ['to_model_formulations']


def to_model_formulations(
        models: Sequence[Union[ConductModel, Vertical]],
) -> List[ModelFormulation]:
    """Translate a new-API ``models=[...]`` sequence to legacy formulations."""
    if not models:
        raise ValueError("At least one conduct model must be provided.")
    out: List[ModelFormulation] = []
    for i, m in enumerate(models):
        if isinstance(m, Vertical):
            out.append(_vertical_to_formulation(m, i))
        elif isinstance(m, ConductModel):
            out.append(_simple_to_formulation(m, i))
        else:
            raise TypeError(
                f"models[{i}] must be a ConductModel or Vertical instance, "
                f"got {type(m).__name__}. Use pyRVtest.Bertrand(...), etc., "
                f"or the Vertical(downstream=..., upstream=...) wrapper."
            )
    return out


def _simple_to_formulation(m: ConductModel, idx: int) -> ModelFormulation:
    """Single-tier conduct (no upstream) to ModelFormulation."""
    kwargs = _base_config_kwargs(m)
    # Conduct-specific fields.
    if isinstance(m, CustomConductModel):
        kwargs['model_downstream'] = 'other'
        kwargs['custom_model_specification'] = {m.name: m.markup_fn}
    else:
        kwargs['model_downstream'] = m._model_name
    if isinstance(m, MixCournotBertrand):
        # mix_flag already in _base_config_kwargs; no extra work.
        pass
    # Ownership / kappa live on the inner class.
    kwargs['ownership_downstream'] = m.ownership
    if m.kappa_specification is not None:
        kwargs['kappa_specification_downstream'] = m.kappa_specification
    return ModelFormulation(**kwargs)


def _vertical_to_formulation(v: Vertical, idx: int) -> ModelFormulation:
    """Vertical(downstream=X, upstream=Y, ...) to legacy ModelFormulation."""
    # Shared config lives on the Vertical wrapper.
    kwargs: dict = {
        'vertical_integration': v.vertical_integration,
        'unit_tax': v.unit_tax,
        'advalorem_tax': v.advalorem_tax,
        'advalorem_payer': v.advalorem_payer,
        'cost_scaling': v.cost_scaling,
        'user_supplied_markups': v.user_supplied_markups,
    }
    # Downstream conduct + ownership + kappa.
    down = v.downstream
    if isinstance(down, CustomConductModel):
        kwargs['model_downstream'] = 'other'
        kwargs['custom_model_specification'] = {down.name: down.markup_fn}
    else:
        kwargs['model_downstream'] = down._model_name
    kwargs['ownership_downstream'] = down.ownership
    if down.kappa_specification is not None:
        kwargs['kappa_specification_downstream'] = down.kappa_specification
    if isinstance(down, MixCournotBertrand):
        kwargs['mix_flag'] = down.mix_flag
    # Upstream conduct + ownership + kappa.
    up = v.upstream
    if isinstance(up, CustomConductModel):
        raise NotImplementedError(
            "CustomConductModel on the upstream side of Vertical is not "
            "currently supported by the legacy ModelFormulation bridge. "
            "Use a built-in conduct class for the upstream tier, or wait "
            "for a future refactor that consumes ConductModel instances "
            "directly."
        )
    kwargs['model_upstream'] = up._model_name
    kwargs['ownership_upstream'] = up.ownership
    if up.kappa_specification is not None:
        kwargs['kappa_specification_upstream'] = up.kappa_specification
    return ModelFormulation(**kwargs)


def _base_config_kwargs(m: ConductModel) -> dict:
    """Pull shared (non-conduct-specific) config off a bare ConductModel."""
    kwargs: dict = {
        'unit_tax': m.unit_tax,
        'advalorem_tax': m.advalorem_tax,
        'advalorem_payer': m.advalorem_payer,
        'cost_scaling': m.cost_scaling,
        'vertical_integration': m.vertical_integration,
        'user_supplied_markups': m.user_supplied_markups,
    }
    if m.mix_flag is not None:
        kwargs['mix_flag'] = m.mix_flag
    return kwargs
