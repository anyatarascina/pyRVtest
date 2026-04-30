"""Conduct testing problem and supporting data structures."""

import abc
import logging
import time
import warnings
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from pyblp.utilities.algebra import precisely_identify_collinearity
from pyblp.utilities.basics import (
    Array, RecArray, StringRepresentation, extract_matrix, format_seconds, format_table,
    get_indices,
)
from pyblp.configurations.formulation import ColumnFormulation
from . import options
from .exceptions import ValidationError
from .formulation import Absorb, Formulation, ModelFormulation
from .markups import build_ownership
from .models import _LABOR_SIDE_MODEL_NAMES, _PRODUCT_SIDE_MODEL_NAMES
from .data import read_critical_values_tables
from .products import Products
from .results import ProblemResults, Progress
from .solve.endogenous_cost import iv_correct as _iv_correct_stage
from .solve.markups import compute as _markups_stage
# Re-export ``_qr_residualize`` as a module attribute for tests that
# import it from ``pyRVtest.problem`` (tests/test_analytical.py).
from .solve.orthogonalize import qr_residualize as _qr_residualize  # noqa: F401
from .solve.orthogonalize import residualize as _residualize_stage
from .solve.test_engine import (
    compute_block_gram as _compute_block_gram_stage,
    compute_instrument_results as _compute_instrument_results_stage,
    compute_mcs as _compute_mcs_stage,
    extract_block as _extract_block_stage,
)

# v0.4 step 18: per-module logger. Emits INFO-level progress messages that
# previously went through pyblp's ``output()`` helper. Users can silence this
# subsystem specifically with
# ``logging.getLogger("pyRVtest.problem").setLevel(logging.WARNING)``.
logger = logging.getLogger(__name__)


_DEMAND_PARAMS_SIGMA_DEPRECATION_MSG = (
    "demand_params['sigma'] is deprecated; use demand_params['rho'] instead. "
    "The name change aligns pyRVtest with pyblp's nested-logit parameter name. "
    "'sigma' will be removed in v0.6. See docs/migrating_to_v0.4.rst for "
    "details."
)

# v0.4 step 6b: once-per-session flag for the sigma -> rho deprecation warning.
# Module-level rather than class-level because demand_params is consumed by
# Problem.__init__ and we only want one warning per Python session regardless
# of how many Problem() calls trigger it.
_demand_params_sigma_deprecation_warned = False

_MC_CORRECTION_DEPRECATION_MSG = (
    "Problem.solve(mc_correction=...) is deprecated and will be removed in "
    "v0.6. The general mechanism for adjusting marginal cost outside the "
    "behavioral model is the endogenous_cost_component argument to Problem "
    "(which feeds the IV correction stage). "
    "Fix: pass an endogenous_cost_component column at Problem construction "
    "time instead of adding mc_correction at solve time."
)

# Once-per-session flag for the mc_correction deprecation warning, mirroring
# the sigma -> rho pattern above.
_mc_correction_deprecation_warned = False

# v0.4 OQ 14: the generic per-model tax deprecation warning emission +
# shared once-per-session flag live on ``pyRVtest/models/base.py`` so
# they fire at ``ConductModel.__init__`` / ``Vertical.__init__`` time
# (rc1 follow-up, Lorenzo P1 item 7). Re-exported here so the
# conflict-warning path below can share the same guard set.
from .models.base import (
    _legacy_tax_deprecation_warned,
    _MODEL_UNIT_TAX_DEPRECATION_MSG,  # noqa: F401 — backward-compat re-export
    _MODEL_ADVALOREM_TAX_DEPRECATION_MSG,  # noqa: F401 — backward-compat re-export
)

_MODEL_TAX_CONFLICT_MSG_TEMPLATE = (
    "Conflicting {field} specification: model-level {field}={model!r} wins "
    "(legacy precedence) but Problem-level {field}={problem!r} was also set "
    "and will be ignored for this model. Fix: drop the model-level {field} "
    "and rely on the Problem-level value (use {field}_salient=False to opt "
    "individual models out for salience tests)."
)


def _validate_problem_tax_kwargs(
        unit_tax: Optional[str],
        advalorem_tax: Optional[str],
        advalorem_payer: Optional[str],
) -> Optional[str]:
    """Validate Problem-level tax kwargs introduced in v0.4 OQ 14.

    Returns a normalized ``advalorem_payer`` (mapping ``'firms'`` ->
    ``'firm'`` and ``'consumers'`` -> ``'consumer'`` for parity with the
    per-model validator) or ``None``.
    """
    if unit_tax is not None and not isinstance(unit_tax, str):
        raise ValidationError(
            f"Expected Problem(unit_tax=...) to be a column-name string or None. "
            f"Received {type(unit_tax).__name__} ({unit_tax!r}). "
            f"Fix: pass the column name as a string, e.g. Problem(..., unit_tax='tax_col')."
        )
    if advalorem_tax is not None and not isinstance(advalorem_tax, str):
        raise ValidationError(
            f"Expected Problem(advalorem_tax=...) to be a column-name string or None. "
            f"Received {type(advalorem_tax).__name__} ({advalorem_tax!r}). "
            f"Fix: pass the column name as a string, e.g. Problem(..., advalorem_tax='vat_col')."
        )
    if advalorem_payer is not None and advalorem_payer not in {
            'firm', 'consumer', 'firms', 'consumers'}:
        raise ValidationError(
            f"Expected Problem(advalorem_payer=...) to be 'firm' or 'consumer' (or None). "
            f"Received {advalorem_payer!r}. "
            f"Fix: pass advalorem_payer='firm' or 'consumer'."
        )
    if advalorem_tax is not None and advalorem_payer is None:
        raise ValidationError(
            "Expected Problem(advalorem_payer=...) to be 'firm' or 'consumer' when "
            "Problem(advalorem_tax=...) is supplied. "
            "Received advalorem_payer=None. "
            "Fix: set advalorem_payer='firm' or 'consumer'."
        )
    if advalorem_tax is None and advalorem_payer is not None:
        raise ValidationError(
            "Expected Problem(advalorem_tax=...) to be supplied when "
            "Problem(advalorem_payer=...) is set. "
            "Received advalorem_payer without advalorem_tax. "
            "Fix: drop advalorem_payer, or pair it with advalorem_tax='col'."
        )
    # Normalize plural forms to singular so the downstream resolver sees
    # one canonical spelling ('firm'/'consumer').
    if advalorem_payer is not None:
        advalorem_payer = advalorem_payer.replace(
            'consumers', 'consumer').replace('firms', 'firm')
    return advalorem_payer


def _normalize_demand_params_rho(demand_params: dict) -> dict:
    """Translate the deprecated ``demand_params['sigma']`` alias to the canonical
    ``demand_params['rho']`` key. Returns a shallow copy; never mutates the
    caller's dict.

    If both ``rho`` and ``sigma`` are present, raises ``TypeError``.
    If only ``sigma`` is present, emits a once-per-session
    ``DeprecationWarning`` and translates to ``rho`` internally.
    Downstream code then reads ``demand_params['sigma']`` only when the
    deprecated alias is the one the caller supplied; for robustness the
    returned dict has both keys pointing to the same value so any code
    reading either name sees consistent state.
    """
    global _demand_params_sigma_deprecation_warned

    has_rho = 'rho' in demand_params
    has_sigma = 'sigma' in demand_params
    if has_rho and has_sigma:
        raise TypeError(
            "Expected demand_params to contain at most one of 'rho' or 'sigma'. "
            "Received both keys; demand_params cannot contain both 'rho' and "
            "'sigma' ('rho' is canonical; 'sigma' is a deprecated alias). "
            "Fix: drop 'sigma' and keep 'rho' (they have identical semantics); "
            "'sigma' will be removed in v0.6."
        )
    normalized = dict(demand_params)
    if has_sigma and not has_rho:
        if not _demand_params_sigma_deprecation_warned:
            warnings.warn(
                _DEMAND_PARAMS_SIGMA_DEPRECATION_MSG,
                DeprecationWarning,
                stacklevel=3,
            )
            _demand_params_sigma_deprecation_warned = True
        # Keep both 'rho' and 'sigma' in the normalized dict for backward
        # compatibility; downstream code reads 'rho'. (AFSSZ L-level
        # convention). Copy the value under 'rho' too so either name works.
        normalized['rho'] = normalized['sigma']
    elif has_rho and not has_sigma:
        # Canonical form supplied. Mirror to 'sigma' for backward compatibility
        # with any code that reads the deprecated key.
        normalized['sigma'] = normalized['rho']
    return normalized


def _apply_conduct_to_fields(
        m, conduct, product_data, tier,
        models_downstream, models_upstream, firm_ids_downstream, firm_ids_upstream,
        ownership_matrices_downstream, ownership_matrices_upstream,
        custom_model, mix_flag,
):
    """Fill per-tier (downstream / upstream) fields of the Models recarray from
    a ConductModel / CustomConductModel / MixCournotBertrand instance.

    Helper for ``Models.__new__`` (v0.4 step 5b'). Keeps the per-tier logic
    in one place so downstream and upstream paths share the same ownership,
    kappa, and custom-model dispatch.
    """
    from .models import CustomConductModel, MixCournotBertrand
    model_name = conduct._model_name
    if isinstance(conduct, CustomConductModel):
        model_name = 'other'
        custom_model[m] = {conduct.name: conduct.markup_fn}
    if tier == 'downstream':
        models_downstream[m] = model_name
        firm_ids_list = firm_ids_downstream
        ownership_list = ownership_matrices_downstream
    else:
        models_upstream[m] = model_name
        firm_ids_list = firm_ids_upstream
        ownership_list = ownership_matrices_upstream

    # Ownership matrix construction: matches legacy Models.__new__ behavior.
    if model_name == 'monopoly':
        ownership_list[m] = build_ownership(product_data, conduct.ownership, 'monopoly')
        firm_ids_list[m] = 'monopoly'
    elif conduct.ownership is not None:
        ownership_list[m] = build_ownership(
            product_data, conduct.ownership, conduct.kappa_specification,
        )
        firm_ids_list[m] = conduct.ownership

    # mix_flag lives on the downstream conduct when it's a MixCournotBertrand.
    if tier == 'downstream' and isinstance(conduct, MixCournotBertrand) and conduct.mix_flag is not None:
        mix_flag[m] = extract_matrix(product_data, conduct.mix_flag).flatten().astype(bool)


# v0.4 step 14c: labor-side column-name defaults and aliasing helpers.
#
# ``Problem(market_side='labor')`` accepts a ``column_names`` override
# following plan Open Question 6: defaults map 'price' -> 'wages' and
# 'shares' -> 'employment_share', but a caller can pass
# ``column_names={'price': 'wages', 'shares': 'my_emp_share_col'}`` to
# rebind. The canonical pyRVtest 'shares' column treats the value as a
# share (in [0, 1]), so the default column name advertises the units
# rather than naming a raw quantity; users with raw employment counts
# must normalize to market-level employment shares before passing data
# in. Inside the package we keep the canonical column names ``prices``
# and ``shares`` (so ``Products`` and the solve stages stay untouched);
# the aliasing step runs at ``Problem.__init__`` only.

_LABOR_COLUMN_DEFAULTS: Dict[str, str] = {
    'price': 'wages',
    'shares': 'employment_share',
}

_LABOR_COLUMN_NAMES_KEYS = frozenset(_LABOR_COLUMN_DEFAULTS.keys())


def _resolve_labor_column_names(
        column_names: Optional[Mapping[str, str]],
) -> Dict[str, str]:
    """Resolve user-facing labor column-name overrides against defaults.

    Returns a ``{'price': <col>, 'shares': <col>}`` dict where each value is
    the name of the column in ``product_data`` that carries the labor-side
    wage / employment variable. Raises :class:`ValidationError` if the
    override dict contains unexpected keys (caught at init so the user sees
    the typo immediately).
    """
    resolved = dict(_LABOR_COLUMN_DEFAULTS)
    if column_names is None:
        return resolved
    if not isinstance(column_names, Mapping):
        raise ValidationError(
            f"Expected column_names to be a mapping of "
            f"{{'price': <col>, 'shares': <col>}}. "
            f"Received {type(column_names).__name__}. "
            f"Fix: pass column_names={{'price': 'wages', 'shares': 'employment_share'}} "
            f"or omit the argument to accept labor-side defaults."
        )
    extras = set(column_names) - _LABOR_COLUMN_NAMES_KEYS
    if extras:
        raise ValidationError(
            f"Expected column_names keys to be a subset of "
            f"{sorted(_LABOR_COLUMN_NAMES_KEYS)}. "
            f"Received unexpected keys: {sorted(extras)}. "
            f"Fix: use only 'price' and/or 'shares' as override keys, matching "
            f"the PyBLP formulation-pattern convention."
        )
    for k, v in column_names.items():
        if not isinstance(v, str) or not v:
            raise ValidationError(
                f"Expected column_names[{k!r}] to be a non-empty column-name "
                f"string. "
                f"Received {v!r}. "
                f"Fix: pass the name of the column in product_data carrying "
                f"the labor-side {k} variable."
            )
        resolved[k] = v
    return resolved


def _validate_labor_sign_conventions(
        product_data: Mapping[str, Any], resolved_column_names: Mapping[str, str],
) -> None:
    """Enforce labor-side positivity: wages > 0, employment > 0, on every row.

    Fails loudly when a caller encodes wages as negative numbers (product-
    side sign convention leaking into a labor-side call) or when employment
    shares are zero / missing. Per plan §4.5.

    The supply-Jacobian sign (``ds/dw > 0`` for upward-sloping supply) is a
    backend concern; it is validated when :class:`LaborSupplyBackend`
    populates ``compute_jacobian`` in v0.5 (the skeleton does not produce
    numeric output so there is nothing to check here).
    """
    wage_col = resolved_column_names['price']
    emp_col = resolved_column_names['shares']
    wages = extract_matrix(product_data, wage_col)
    employment = extract_matrix(product_data, emp_col)
    if wages is None:
        raise ValidationError(
            f"Expected product_data to contain the labor-side wage column "
            f"{wage_col!r} when market_side='labor'. "
            f"Received product_data without that key. "
            f"Fix: add the column, or override the default via "
            f"Problem(..., market_side='labor', "
            f"column_names={{'price': '<your-wage-col>'}})."
        )
    if employment is None:
        raise ValidationError(
            f"Expected product_data to contain the labor-side employment "
            f"column {emp_col!r} when market_side='labor'. "
            f"Received product_data without that key. "
            f"Fix: add the column, or override the default via "
            f"Problem(..., market_side='labor', "
            f"column_names={{'shares': '<your-employment-col>'}})."
        )
    wages_flat = np.asarray(wages).ravel()
    emp_flat = np.asarray(employment).ravel()
    if not np.all(wages_flat > 0):
        n_bad = int((wages_flat <= 0).sum())
        raise ValidationError(
            f"Expected every entry of {wage_col!r} to be strictly positive "
            f"when market_side='labor' (upward-sloping labor supply requires "
            f"positive wages). "
            f"Received {n_bad} row(s) with wages <= 0 "
            f"(min={float(wages_flat.min())}, max={float(wages_flat.max())}). "
            f"Fix: check the sign convention in your data pipeline — labor-"
            f"side wages must be encoded as positive real numbers (product-"
            f"side callers sometimes encode prices as negative when flipping "
            f"sign by hand; do not do that here). See "
            f"docs/agent_guide.rst#labor-side-usage."
        )
    if not np.all(emp_flat > 0):
        n_bad = int((emp_flat <= 0).sum())
        raise ValidationError(
            f"Expected every entry of {emp_col!r} to be strictly positive "
            f"when market_side='labor'. "
            f"Received {n_bad} row(s) with employment <= 0 "
            f"(min={float(emp_flat.min())}, max={float(emp_flat.max())}). "
            f"Fix: drop zero-employment rows or check the column. See "
            f"docs/agent_guide.rst#labor-side-usage."
        )


def _validate_labor_models(models: Sequence[Any]) -> None:
    """Reject product-side models under market_side='labor' per plan §4.5.

    :class:`CustomConductModel` requires an explicit ``side='labor'``
    opt-in: its user-supplied ``markup_fn`` implicitly picks a sign
    convention (unlike :class:`PerfectCompetition`, whose zero output is
    genuinely side-neutral), so silently accepting it on either side
    would let a product-side formula slip into a labor problem
    unnoticed.
    """
    from .models import ConductModel, CustomConductModel, Vertical
    bad: list[tuple[int, str]] = []
    for i, m in enumerate(models):
        if isinstance(m, Vertical):
            for tier_name, conduct in (('downstream', m.downstream), ('upstream', m.upstream)):
                if conduct is None:
                    continue
                name = getattr(conduct, '_model_name', '')
                if name in _PRODUCT_SIDE_MODEL_NAMES:
                    bad.append((i, f"Vertical.{tier_name}={name}"))
            continue
        if isinstance(m, CustomConductModel):
            # Custom models implicitly pick a sign convention via the
            # user's markup_fn; require an explicit side='labor' opt-in.
            if m.side != 'labor':
                received = 'default (product)' if m.side is None else repr(m.side)
                raise ValidationError(
                    f"Expected every CustomConductModel passed under "
                    f"market_side='labor' to carry an explicit "
                    f"side='labor' opt-in, because a user-supplied "
                    f"markup_fn implicitly picks a sign convention. "
                    f"Received models[{i}]=CustomConductModel with "
                    f"side={received}. "
                    f"Fix: if your markup_fn is written for the "
                    f"labor-side (upward-sloping supply, markdown) "
                    f"convention, pass "
                    f"CustomConductModel(markup_fn=..., side='labor'); "
                    f"otherwise use a labor-side conduct class "
                    f"(Monopsony, BertrandWages, CournotEmployment, "
                    f"NashBargaining) or switch market_side back to "
                    f"'product'."
                )
            continue
        if not isinstance(m, ConductModel):
            continue
        name = getattr(m, '_model_name', '')
        if name in _PRODUCT_SIDE_MODEL_NAMES:
            bad.append((i, name))
    if bad:
        offenders = ', '.join(f"models[{i}]={n}" for i, n in bad)
        raise ValidationError(
            f"Expected every entry of models to be a labor-side conduct "
            f"model when market_side='labor'. "
            f"Received product-side model(s): {offenders}. "
            f"Fix: replace with the labor-side analogue "
            f"(Monopsony, BertrandWages, CournotEmployment, NashBargaining), "
            f"or switch market_side back to 'product' (the default)."
        )


def _validate_product_models(models: Sequence[Any]) -> None:
    """Reject labor-side models (and mis-sided custom models) under product mode.

    Symmetric check to :func:`_validate_labor_models`: a
    :class:`CustomConductModel` constructed with ``side='labor'`` must
    not be passed to a product-side problem. Labor-side conduct classes
    (:class:`Monopsony`, :class:`BertrandWages`,
    :class:`CournotEmployment`, :class:`NashBargaining`) are likewise
    rejected here so the product/labor boundary is enforced symmetrically
    at init time. :class:`PerfectCompetition` stays side-neutral.
    """
    from .models import ConductModel, CustomConductModel, Vertical
    bad: list[tuple[int, str]] = []
    for i, m in enumerate(models):
        if isinstance(m, Vertical):
            for tier_name, conduct in (('downstream', m.downstream), ('upstream', m.upstream)):
                if conduct is None:
                    continue
                name = getattr(conduct, '_model_name', '')
                if name in _LABOR_SIDE_MODEL_NAMES:
                    bad.append((i, f"Vertical.{tier_name}={name}"))
            continue
        if isinstance(m, CustomConductModel):
            if m.side == 'labor':
                raise ValidationError(
                    f"Expected CustomConductModel under the default "
                    f"market_side='product' to carry side='product' or "
                    f"side=None. "
                    f"Received models[{i}]=CustomConductModel with "
                    f"side='labor'. "
                    f"Fix: drop side='labor' (or set side='product') if "
                    f"your markup_fn is written for the classic "
                    f"downstream-markup convention, or switch the "
                    f"Problem to market_side='labor' if the formula is "
                    f"for the upward-sloping labor-supply convention."
                )
            continue
        if not isinstance(m, ConductModel):
            continue
        name = getattr(m, '_model_name', '')
        if name in _LABOR_SIDE_MODEL_NAMES:
            bad.append((i, name))
    if bad:
        offenders = ', '.join(f"models[{i}]={n}" for i, n in bad)
        raise ValidationError(
            f"Expected every entry of models to be a product-side "
            f"conduct model under the default market_side='product'. "
            f"Received labor-side model(s): {offenders}. "
            f"Fix: replace with the product-side analogue (Bertrand, "
            f"Cournot, Monopoly), or pass market_side='labor' to run "
            f"the labor-supply testing path."
        )


def _apply_labor_column_aliases(
        product_data: Mapping[str, Any], resolved_column_names: Mapping[str, str],
) -> Mapping[str, Any]:
    """Alias the labor-side wage/employment columns into prices/shares.

    Returns a new object suitable for ``Products.__new__`` that carries the
    user's labor-side data under the canonical names 'prices' and 'shares'.
    For pandas DataFrames we copy + assign; for dict-like mappings we return
    a new dict with the aliased keys. The original ``product_data`` is not
    mutated.

    Called only when ``market_side='labor'``. A no-op if the user already
    named the columns 'prices' and 'shares' (they get copied back to
    themselves).
    """
    wage_col = resolved_column_names['price']
    emp_col = resolved_column_names['shares']

    # pandas DataFrame branch (most common caller).
    try:
        import pandas as pd
        if isinstance(product_data, pd.DataFrame):
            aliased = product_data.copy()
            aliased['prices'] = aliased[wage_col]
            aliased['shares'] = aliased[emp_col]
            return aliased
    except ImportError:  # pragma: no cover — pandas is a hard dep of pyblp
        pass

    # Generic mapping branch (dict of arrays).
    if isinstance(product_data, Mapping):
        aliased_dict: Dict[str, Any] = dict(product_data)
        wages = extract_matrix(product_data, wage_col)
        employment = extract_matrix(product_data, emp_col)
        aliased_dict['prices'] = wages
        aliased_dict['shares'] = employment
        return aliased_dict

    # Structured / recarray fallback: rebuild as a dict. Rare in practice
    # but keeps the labor path working for NumPy structured arrays.
    aliased_dict = {}
    try:
        field_names = list(product_data.dtype.names)
    except AttributeError as exc:  # pragma: no cover
        raise ValidationError(
            f"Expected product_data to be a pandas DataFrame, dict-like "
            f"mapping, or structured ndarray. "
            f"Received {type(product_data).__name__}. "
            f"Fix: wrap the data in pd.DataFrame(...) or a dict of arrays "
            f"before passing it to Problem(..., market_side='labor')."
        ) from exc
    for name in field_names:
        aliased_dict[name] = product_data[name]
    aliased_dict['prices'] = extract_matrix(product_data, wage_col)
    aliased_dict['shares'] = extract_matrix(product_data, emp_col)
    return aliased_dict


class Models(object):
    r"""Models data structured as a dictionary.

    Attributes
    ----------
    models_downstream: `str`
        Model of conduct for downstream firms. This is used to construct downstream markups. This must be one of the
        allowed models.
    models_upstream: `str, optional`
        Model of conduct for upstream firms. This is used to construct upstream markups. This must be one of the
        allowed models.
    firm_ids_downstream: `ndarray`
        Vector of firm ids used to construct ownership for downstream firms.
    firm_ids_upstream: `ndarray`
        Vector of firm ids used to construct ownership for upstream firms.
    ownership_matrices_downstream: `ndarray`
        Matrix of ownership relationships between downstream firms.
    ownership_matrices_upstream: `ndarray`
        Matrix of ownership relationships between upstream firms.
    vertical_integration: `ndarray, optional`
        Vector indicating which product_ids are vertically integrated (i.e. store brands).
    vertical_integration_index: `ndarray, optional`
        Indicates the index for a particular vertical relationship (which model it corresponds to).
    unit_tax: `ndarray, optional`
        A vector containing information on unit taxes.
    unit_tax_name: `str, optional`
        The column name for the column containing unit taxes.
    advalorem_tax: `ndarray, optional`
        A vector containing information on advalorem taxes.
    advalorem_tax_name: ``str, optional`
        The column name for the column containing advalorem taxes.
    advalorem_payer: `str, optional`
        If there are advalorem taxes in the model, this specifies who the payer of these taxes are. It can be either the
        consumer or the firm.
    cost_scaling: `ndarray, optional`
        The cost scaling parameter.
    cost_scaling_column: `str, optional`
        The name of the column containing the cost scaling parameter.
    constant_markup: `ndarray, optional`
        Per-product dollar markup :math:`\\zeta_j` for models that specify a
        fixed additive markup (Dearing et al. 2026, Example 7). ``None`` for
        models that compute markups from first-order conditions.
    custom_model: `dict, optional`
        A custom formula used to compute markups, optionally specified by the user.
    user_supplied_markups: `ndarray, optional`
        A vector of user-computed markups.
    user_supplied_markups_name: `str, optional`
        The name of the column containing user-supplied markups.

    Examples
    --------
    >>> import pyRVtest  # doctest: +SKIP
    >>> # Models is an internal recarray built by Problem.__init__; users
    >>> # construct it indirectly by passing `models=[Bertrand(...), ...]`
    >>> # to Problem. See the Problem docstring for the full workflow.
    """

    models_downstream: Array
    models_upstream: Array
    firm_ids_downstream: Array
    firm_ids_upstream: Array
    ownership_matrices_downstream: Array
    ownership_matrices_upstream: Array
    vertical_integration: Array
    vertical_integration_index: Array
    custom_model: Array
    unit_tax: Array
    unit_tax_name: Array
    advalorem_tax: Array
    advalorem_tax_name: Array
    advalorem_payer: Array
    cost_scaling: Array
    cost_scaling_column: Array
    constant_markup: Array
    user_supplied_markups: Array
    user_supplied_markups_name: Array

    def __new__(
            cls,
            models: Sequence[Any],
            product_data: Mapping,
            problem_unit_tax: Optional[str] = None,
            problem_advalorem_tax: Optional[str] = None,
            problem_advalorem_payer: Optional[str] = None,
    ) -> RecArray:
        """Structure model data for the pipeline.

        v0.4 step 5b': ``models`` is a sequence of
        ``pyRVtest.models.ConductModel`` or ``pyRVtest.models.Vertical``
        instances (the canonical intermediate form). ``Problem.__init__``
        converts ``ModelFormulation`` inputs to this form via
        ``from_model_formulation`` before calling here. Legacy
        ``ModelFormulation`` instances passed directly are accepted for
        backward compatibility (translated inline).

        v0.4 OQ 14: accepts Problem-level tax kwargs
        (``problem_unit_tax``, ``problem_advalorem_tax``,
        ``problem_advalorem_payer``) that are applied to every model
        with salience flag ``True``, per Scenarios 1-6 in the plan.
        Model-level tax columns (the legacy path) still win by
        precedence; when a model carries one, a once-per-session
        :class:`DeprecationWarning` fires.
        """
        from .models import ConductModel, Vertical
        from .models._adapter import from_model_formulation

        # Allow legacy ModelFormulation instances for backward compat; translate
        # them to ConductModel/Vertical on the fly.
        normalized: list = []
        for i, entry in enumerate(models):
            if isinstance(entry, (ConductModel, Vertical)):
                normalized.append(entry)
            elif isinstance(entry, ModelFormulation):
                normalized.append(from_model_formulation(entry))
            elif entry is None:
                raise TypeError(
                    f"Expected every entry of models to be a ConductModel, "
                    f"Vertical, or ModelFormulation instance. "
                    f"Received models[{i}] = None. "
                    f"Fix: replace the None entry with a conduct-model instance "
                    f"(e.g., pyRVtest.Bertrand(ownership='firm_ids'))."
                )
            else:
                raise TypeError(
                    f"Expected every entry of models to be a ConductModel, "
                    f"Vertical, or ModelFormulation instance. "
                    f"Received models[{i}] of type {type(entry).__name__}. "
                    f"Fix: wrap the specification in the appropriate class "
                    f"(Bertrand, Cournot, Monopoly, Vertical, etc.)."
                )

        M = len(normalized)
        if M < 1:
            raise ValueError(
                "Expected at least one candidate model in models. "
                "Received an empty sequence. "
                "Fix: pass models=[Bertrand(...), Cournot(...)] (or similar) "
                "with at least one ConductModel entry."
            )
        N = product_data.shape[0]

        # initialize per-model fields
        models_downstream = [None] * M
        models_upstream = [None] * M
        firm_ids_downstream = [None] * M
        firm_ids_upstream = [None] * M
        ownership_matrices_downstream = [None] * M
        ownership_matrices_upstream = [None] * M
        vertical_integration = [None] * M
        vertical_integration_index = [None] * M
        custom_model = [None] * M
        unit_tax = [None] * M
        unit_tax_name = [None] * M
        advalorem_tax = [None] * M
        advalorem_tax_name = [None] * M
        advalorem_payer = [None] * M
        cost_scaling = [None] * M
        cost_scaling_column = [None] * M
        constant_markup = [None] * M
        user_supplied_markups = [None] * M
        user_supplied_markups_name = [None] * M
        mix_flag = [None] * M

        for m, entry in enumerate(normalized):
            if isinstance(entry, Vertical):
                down_conduct = entry.downstream
                up_conduct = entry.upstream
                config = entry  # shared config lives on Vertical
            else:
                down_conduct = entry
                up_conduct = None
                config = entry  # shared config lives on the simple class

            # Downstream conduct + ownership + mix_flag + custom_model.
            _apply_conduct_to_fields(
                m, down_conduct, product_data, 'downstream',
                models_downstream, models_upstream, firm_ids_downstream, firm_ids_upstream,
                ownership_matrices_downstream, ownership_matrices_upstream,
                custom_model, mix_flag,
            )
            # Upstream conduct + ownership.
            if up_conduct is not None:
                _apply_conduct_to_fields(
                    m, up_conduct, product_data, 'upstream',
                    models_downstream, models_upstream, firm_ids_downstream, firm_ids_upstream,
                    ownership_matrices_downstream, ownership_matrices_upstream,
                    custom_model, mix_flag,
                )
            # Shared config (vertical_integration, taxes, cost_scaling, user_supplied).
            if config.vertical_integration is not None:
                vertical_integration[m] = extract_matrix(product_data, config.vertical_integration)
                vertical_integration_index[m] = config.vertical_integration
            # v0.4 OQ 14: resolve effective tax source per model per the
            # precedence in plan Scenarios 1-6:
            #   1. legacy model-level unit_tax wins (with DeprecationWarning)
            #   2. otherwise Problem-level unit_tax if unit_tax_salient
            #   3. otherwise zero.
            # ``config`` is either a ConductModel or a Vertical; both carry
            # the ``unit_tax``, ``advalorem_tax``, ``advalorem_payer``, and
            # the two salience flags.
            unit_tax_salient = getattr(config, 'unit_tax_salient', True)
            advalorem_tax_salient = getattr(config, 'advalorem_tax_salient', True)

            if config.unit_tax is not None:
                # Legacy path; warn and honor precedence.
                if 'unit_tax' not in _legacy_tax_deprecation_warned:
                    warnings.warn(
                        _MODEL_UNIT_TAX_DEPRECATION_MSG,
                        DeprecationWarning,
                        stacklevel=3,
                    )
                    _legacy_tax_deprecation_warned.add('unit_tax')
                if problem_unit_tax is not None:
                    # Extra warning: conflicting spec.
                    warnings.warn(
                        _MODEL_TAX_CONFLICT_MSG_TEMPLATE.format(
                            field='unit_tax',
                            model=config.unit_tax,
                            problem=problem_unit_tax,
                        ),
                        DeprecationWarning,
                        stacklevel=3,
                    )
                unit_tax[m] = extract_matrix(product_data, config.unit_tax)
                unit_tax_name[m] = config.unit_tax
            elif problem_unit_tax is not None and unit_tax_salient:
                unit_tax[m] = extract_matrix(product_data, problem_unit_tax)
                unit_tax_name[m] = problem_unit_tax
            else:
                unit_tax[m] = np.zeros((N, 1))

            if config.advalorem_tax is not None:
                if 'advalorem_tax' not in _legacy_tax_deprecation_warned:
                    warnings.warn(
                        _MODEL_ADVALOREM_TAX_DEPRECATION_MSG,
                        DeprecationWarning,
                        stacklevel=3,
                    )
                    _legacy_tax_deprecation_warned.add('advalorem_tax')
                if problem_advalorem_tax is not None:
                    warnings.warn(
                        _MODEL_TAX_CONFLICT_MSG_TEMPLATE.format(
                            field='advalorem_tax',
                            model=config.advalorem_tax,
                            problem=problem_advalorem_tax,
                        ),
                        DeprecationWarning,
                        stacklevel=3,
                    )
                advalorem_tax[m] = extract_matrix(product_data, config.advalorem_tax)
                advalorem_tax_name[m] = config.advalorem_tax
                payer = config.advalorem_payer.replace('consumers', 'consumer').replace('firms', 'firm')
                advalorem_payer[m] = payer
            elif problem_advalorem_tax is not None and advalorem_tax_salient:
                advalorem_tax[m] = extract_matrix(product_data, problem_advalorem_tax)
                advalorem_tax_name[m] = problem_advalorem_tax
                advalorem_payer[m] = problem_advalorem_payer
            else:
                advalorem_tax[m] = np.zeros((N, 1))
            if config.cost_scaling is not None:
                # cost_scaling accepts either a column name (str) or a
                # numeric scalar (float/int, broadcast to every row).
                if isinstance(config.cost_scaling, str):
                    cost_scaling_column[m] = config.cost_scaling
                    cost_scaling[m] = extract_matrix(product_data, config.cost_scaling)
                else:
                    cost_scaling_column[m] = None
                    cost_scaling[m] = np.full((N, 1), float(config.cost_scaling))
            else:
                cost_scaling[m] = np.zeros((N, 1))
            if config.user_supplied_markups is not None:
                user_supplied_markups[m] = extract_matrix(product_data, config.user_supplied_markups)
                user_supplied_markups_name[m] = config.user_supplied_markups

            # v0.4 step 12: ConstantMarkup and RuleOfThumb thread their
            # per-row markup vectors through the Models recarray via the
            # 'constant_markup' field.
            from .models.constant import ConstantMarkup as _ConstantMarkup, RuleOfThumb as _RuleOfThumb
            if isinstance(down_conduct, _RuleOfThumb):
                # RuleOfThumb: markup = (phi-1)/phi * prices, computed here
                # from observed prices rather than demand primitives (O, D, s).
                prices_col = np.asarray(product_data.prices, dtype=options.dtype).reshape(-1, 1)
                constant_markup[m] = ((down_conduct.phi - 1.0) / down_conduct.phi) * prices_col
            elif isinstance(down_conduct, _ConstantMarkup):
                markup_spec = down_conduct.markup
                if isinstance(markup_spec, str):
                    extracted = extract_matrix(product_data, markup_spec)
                    if extracted is None:
                        raise ValidationError(
                            f"Expected product_data to contain the column "
                            f"{markup_spec!r} referenced by "
                            f"ConstantMarkup(markup={markup_spec!r}). "
                            f"Received product_data without that key. "
                            f"Fix: add the column, or switch to a scalar "
                            f"markup."
                        )
                    constant_markup[m] = np.asarray(extracted, dtype=options.dtype).reshape(-1, 1)
                else:
                    constant_markup[m] = np.full(
                        (N, 1), float(markup_spec), dtype=options.dtype,
                    )

        # structure product fields as a mapping
        models_mapping: Dict[Union[str, tuple], Optional[Array]] = {}
        models_mapping.update({
            'models_downstream': models_downstream,
            'models_upstream': models_upstream,
            'firm_ids_downstream': firm_ids_downstream,
            'firm_ids_upstream': firm_ids_upstream,
            'ownership_downstream': ownership_matrices_downstream,
            'ownership_upstream': ownership_matrices_upstream,
            'vertical_integration': vertical_integration,
            'vertical_integration_index': vertical_integration_index,
            'unit_tax': unit_tax,
            'unit_tax_name': unit_tax_name,
            'advalorem_tax': advalorem_tax,
            'advalorem_tax_name': advalorem_tax_name,
            'advalorem_payer': advalorem_payer,
            'cost_scaling_column': cost_scaling_column,
            'cost_scaling': cost_scaling,
            'constant_markup': constant_markup,
            'custom_model_specification': custom_model,
            'user_supplied_markups': user_supplied_markups,
            'user_supplied_markups_name': user_supplied_markups_name,
            'mix_flag': mix_flag
        })
        return models_mapping


class Container(abc.ABC):
    """An abstract container for structured product and instruments data."""

    products: RecArray
    models: RecArray
    _w_formulation: Tuple[ColumnFormulation, ...]
    _Z_formulation: Tuple[ColumnFormulation, ...]
    # v0.4 step 6a: Dict_Z_formulation is a PER-INSTANCE mutable dict. Pre-v0.4
    # it was a class-level attribute with ``= {}`` at the class body, which
    # meant two concurrent Problem instances shared the same dict (the second
    # instance's Z column formulations were appended to the first's). See the
    # per-instance assignment in ``Container.__init__`` below.
    Dict_Z_formulation: Dict[Union[str, tuple], Tuple[Optional[Array], Any]]

    @abc.abstractmethod
    def __init__(self, products: RecArray, models: RecArray) -> None:
        """Store data and column formulations."""
        self.products = products
        self.models = models
        self._w_formulation = self.products.dtype.fields['w'][2]
        # Instance-level dict; any updates below mutate only this Problem's
        # dict, not a class-shared one.
        self.Dict_Z_formulation = {}

        i = 0
        while 'Z{0}'.format(i) in self.products.dtype.fields:
            self._Z_formulation = self.products.dtype.fields['Z{0}'.format(i)][2]
            self.Dict_Z_formulation.update({"_Z{0}_formulation".format(i): self._Z_formulation})
            i += 1


class Problem(Container, StringRepresentation):
    r"""A firm conduct testing-type problem.

    This class is initialized using the relevant data and formulations, and solved with :meth:`Problem.solve`.

    Parameters
    ----------
    cost_formulation: `Formulation`
        :class:`Formulation` is a list of the variables for observed product characteristics. All observed cost shifters
        included in this formulation must be variables in the `product_data`. To use a constant, one would replace `0`
        with `1`. To absorb fixed effects, specify `absorb = 'C(variable)'`, where the `variable` must also be in the
        `product_data`. Including this option implements fixed effects absorption using
        [PYHDFE](https://github.com/jeffgortmaker/pyhdfe), a companion package to PyBLP.

    instrument_formulation: `Formulation or sequence of Formulation`
        :class:`Formulation` is list of the variables used as excluded instruments for testing. For each instrument
        formulation, there should never be a constant. The user can specify as many instrument formulations as desired.
        All instruments must be variables in `product_data`.

        .. note::
            **Our instrument naming conventions differ from PyBLP**. With PyBLP, one specifies the excluded instruments
            for demand estimation via a naming convention in the product_data: each excluded instrument for demand
            estimation begins with `"demand_instrument"` followed by a number ( i.e., `demand_instrument0`). In
            pyRVtest, you specify directly the names of the variables in the `product_data` that you want to use as
            excluded instruments for testing (i.e., if you want to test with one instrument using the variable in the
            `product_data` named, "transportation_cost" one could specify
            `pyRVtest.Formulation('0 + transportation_cost')`.

    model_formulations: `sequence of ModelFormulation`
        :class:`ModelFormulation` defines the models that the researcher wants to test. There must be at least two
        instances of `ModelFormulation` specified to run the firm conduct testing procedure.

    product_data: `structured array-like`
        This is the data containing product and market observable characteristics, as well as instruments.

    pyblp_results`: `structured array-like`
        The results object returned by `pyblp.solve`.

    Examples
    --------
    >>> import pyRVtest  # doctest: +SKIP
    >>> # Requires a fitted pyblp.ProblemResults object or an explicit
    >>> # demand_params dict. See docs/tutorial.rst for an end-to-end
    >>> # example. A representative call is:
    >>> problem = pyRVtest.Problem(  # doctest: +SKIP
    ...     cost_formulation=pyRVtest.Formulation('0 + w'),
    ...     instrument_formulation=pyRVtest.Formulation('0 + iv'),
    ...     product_data=product_data,
    ...     demand_results=pyblp_results,
    ...     models=[pyRVtest.Bertrand(ownership='firm_ids'),
    ...             pyRVtest.PerfectCompetition()],
    ... )
    >>> results = problem.solve()  # doctest: +SKIP
    """

    model_formulations: Sequence[Optional[ModelFormulation]]
    cost_formulation: Formulation
    instrument_formulation: Formulation
    markups: RecArray
    unique_market_ids: Array
    unique_nesting_ids: Array
    unique_product_ids: Array
    T: int
    N: int
    # v0.4 step 6a: Dict_K is a PER-INSTANCE mutable dict. See the matching
    # comment on Container.Dict_Z_formulation — the class-level ``= {}``
    # made two concurrent Problems share instrument counts.
    Dict_K: Dict[Union[str, tuple], Tuple[Optional[Array]]]
    M: int
    EC: int
    H: int
    L: int
    _market_indices: Dict[Hashable, int]
    _product_market_indices: Dict[Hashable, Array]

    _absorb_cost_ids: Optional[Absorb]

    def __init__(
            self, cost_formulation: Formulation, instrument_formulation: Sequence[Formulation],
            product_data: Mapping, demand_results: Mapping = None,
            model_formulations: Sequence[ModelFormulation] = None,
            markup_data: Optional[RecArray] = None,
            endogenous_cost_component: Optional[str] = None,
            demand_params: Optional[dict] = None,
            models: Optional[Sequence[Any]] = None,
            market_side: str = 'product',
            column_names: Optional[Mapping[str, str]] = None,
            unit_tax: Optional[str] = None,
            advalorem_tax: Optional[str] = None,
            advalorem_payer: Optional[str] = None) -> None:
        """Initialize the underlying economy with product and agent data before absorbing fixed effects.

        v0.4 step 14c adds two labor-side keyword arguments:

        market_side : str, optional
            One of ``'product'`` (default — unchanged behavior) or
            ``'labor'``. When set to ``'labor'``, the package reinterprets
            the data as a labor-supply problem: the Jacobian is
            upward-sloping, column-name defaults flip (``wages``,
            ``employment_share``), labor-side conduct models are required,
            and result labels branch to ``markdown`` / ``MRP`` / ``wage``.
            The ``employment_share`` column is treated as a share (values
            in ``[0, 1]``, summing to at most 1 per market); users with
            raw employment counts must normalize first.
        column_names : dict, optional
            Override labor-side column-name defaults. Accepts
            ``{'price': '<wage-col>', 'shares': '<employment-share-col>'}``.
            Only meaningful when ``market_side='labor'``.

        v0.4 OQ 14 adds Problem-level tax arguments:

        unit_tax : str, optional
            Column name in ``product_data`` holding per-unit taxes.
            When set, every model inherits this tax unless the model
            opts out via ``unit_tax_salient=False`` (the salience-test
            mechanism).
        advalorem_tax : str, optional
            Column name holding ad-valorem tax rates (fractions).
            Inherited by every model by default; individual models can
            opt out via ``advalorem_tax_salient=False``.
        advalorem_payer : str, optional
            ``'firm'`` or ``'consumer'``. Required when
            ``advalorem_tax`` is set; indicates who remits the
            ad-valorem tax.
        """

        logger.info("Initializing the problem ...")
        start_time = time.time()

        # v0.4 step 14c: resolve market_side + column_names first so every
        # downstream validation step sees the resolved labor-column mapping.
        if market_side not in ('product', 'labor'):
            raise ValidationError(
                f"Expected market_side to be one of 'product' or 'labor'. "
                f"Received {market_side!r}. "
                f"Fix: pass market_side='product' (default) for the classic "
                f"downstream-markup testing path, or market_side='labor' for "
                f"the labor-side (monopsony / bargaining) path."
            )
        if column_names is not None and market_side != 'labor':
            raise ValidationError(
                f"Expected column_names to be paired with market_side='labor'. "
                f"Received column_names={column_names!r} with "
                f"market_side={market_side!r}. "
                f"Fix: drop column_names, or add market_side='labor'."
            )
        self._market_side = market_side
        if market_side == 'labor':
            self._labor_column_names: Optional[Dict[str, str]] = (
                _resolve_labor_column_names(column_names)
            )
            # Sign validation on the raw labor columns BEFORE aliasing so the
            # user sees the original column name in the error message.
            _validate_labor_sign_conventions(product_data, self._labor_column_names)
            # Alias user-facing wage / employment columns into the canonical
            # 'prices' / 'shares' names used by Products and the solve stages.
            # This keeps the rest of the package unchanged; the labor-side
            # flip is localized to Problem.__init__.
            product_data = _apply_labor_column_aliases(product_data, self._labor_column_names)
        else:
            self._labor_column_names = None

        # v0.4 step 5b + 5b': ``models=`` is the new class-based API;
        # ``model_formulations=`` is the legacy alias. Mutually exclusive.
        # Internally everything is ConductModel / Vertical: we translate
        # model_formulations to that form here, so the downstream pipeline
        # (Models.__new__, _compute_markups, etc.) sees a single canonical
        # type.
        if models is not None and model_formulations is not None:
            raise TypeError(
                "Expected exactly one of `models=` (class-based, preferred) or "
                "`model_formulations=` (legacy string-based) to be set, not both. "
                "Received both. "
                "Fix: pass only one; prefer models=[Bertrand(...), ...] for new code."
            )
        if models is None and model_formulations is not None:
            from .models._adapter import from_model_formulations
            models = from_model_formulations(model_formulations)
            # Preserve the public attribute `self.model_formulations` below
            # for backwards compat in repr / serialization; set it to the
            # original legacy tuple.
            _legacy_model_formulations = tuple(model_formulations)
        else:
            _legacy_model_formulations = None

        # v0.4 step 14c: reject product-side models on a labor-side problem
        # (and vice versa). Defensive: ``models`` may be ``None`` if the
        # caller supplied precomputed markup_data; the check below only
        # fires when we actually have conduct-model instances to inspect.
        if models is not None:
            if market_side == 'labor':
                _validate_labor_models(models)
            else:
                _validate_product_models(models)

        # Validate demand_params
        if demand_params is not None and demand_results is not None:
            raise ValueError(
                "Expected exactly one of demand_params (analytical path) or "
                "demand_results (pyblp path) to be set, not both. "
                "Received both. "
                "Fix: pass only one; use demand_params={...} for the analytical "
                "logit/nested-logit path, or demand_results=<pyblp.ProblemResults> "
                "for pyblp-driven BLP / nested-logit estimation."
            )
        # v0.4 step 14c (Lorenzo methodology note, 2026-04-18): demand_params
        # drives _construct_demand_backend, which currently only builds
        # product-side backends (LogitBackend / NestedLogitBackend). The
        # labor-side LaborSupplyBackend is a v0.4 skeleton with deferred
        # Jacobian / Hessian math. Combining market_side='labor' with
        # demand_params would silently produce a product-side backend
        # (or a confusing "alpha must be < 0" error for legitimately
        # upward-sloping labor supply). Raise NotImplementedError here
        # so callers see a clean v0.5 pointer rather than an AttributeError
        # or a wrong-sign answer.
        if demand_params is not None and market_side == 'labor':
            raise NotImplementedError(
                "Expected demand_params to be paired with the default "
                "market_side='product' in v0.4. "
                "Received demand_params with market_side='labor'. "
                "Fix: analytical labor-supply demand estimation is deferred "
                "to v0.5 (see pyRVtest.backends.labor.LaborSupplyBackend). "
                "Short-term workarounds in v0.4: (a) pass "
                "user_supplied_markups on each labor conduct model (no "
                "demand_params needed); (b) wrap a manually computed "
                "labor-supply Jacobian in pyRVtest.backends.UserSuppliedBackend."
            )
        if demand_params is not None:
            if 'alpha' not in demand_params:
                raise ValueError(
                    "Expected demand_params to contain an 'alpha' key (the price "
                    "coefficient from demand estimation). "
                    "Received demand_params without 'alpha'. "
                    "Fix: include 'alpha': <estimated negative price coefficient>."
                )
            alpha = demand_params['alpha']
            if not isinstance(alpha, (int, float)) or alpha >= 0:
                raise ValueError(
                    f"Expected demand_params['alpha'] to be a negative real number "
                    f"(for downward-sloping demand). "
                    f"Received {alpha!r}. "
                    f"Fix: pass the estimated price coefficient, which must be < 0."
                )
            # v0.4 step 6b: 'rho' is the canonical nested-logit parameter name
            # (aligns with pyblp). 'sigma' remains accepted as a deprecated
            # alias for backwards compatibility; emit a DeprecationWarning once
            # per session. Mutually exclusive.
            demand_params = _normalize_demand_params_rho(demand_params)
            rho = demand_params.get('rho', [])
            if not isinstance(rho, (list, tuple)):
                raise TypeError(
                    f"Expected demand_params['rho'] (or the deprecated alias "
                    f"'sigma') to be a list or tuple of nesting parameters. "
                    f"Received {type(rho).__name__}. "
                    f"Fix: pass demand_params['rho']=[] for plain logit or "
                    f"[<rho_level>, ...] for nested logit."
                )
            for i, s_val in enumerate(rho):
                if s_val < 0 or s_val >= 1:
                    raise ValueError(
                        f"Expected every nesting parameter in "
                        f"demand_params['rho'] to lie in [0, 1). "
                        f"Received demand_params['rho'][{i}] = {s_val} "
                        f"(out of range [0, 1)). "
                        f"Fix: use 0 for plain logit (Berry 1994 convention) or "
                        f"a value in [0, 1) for nested logit."
                    )

        if markup_data is None:
            if models is None:
                raise TypeError(
                    "Expected either `models=` (class-based) or "
                    "`model_formulations=` (legacy) when markup_data is None, "
                    "so that pyRVtest can compute markups from the conduct models. "
                    "Received both as None. "
                    "Fix: pass models=[Bertrand(...), ...] or supply precomputed "
                    "markup_data yourself."
                )
            M = len(models)
        else:
            M = np.shape(markup_data)[0]

        if hasattr(instrument_formulation, '__len__'):
            L = len(instrument_formulation)
        else:
            L = 1

        if not isinstance(cost_formulation, Formulation):
            raise TypeError(
                f"Expected cost_formulation to be a single pyRVtest.Formulation. "
                f"Received {type(cost_formulation).__name__}. "
                f"Fix: pass cost_formulation=Formulation('0 + <cost-shifters>')."
            )

        if L == 1:
            if not isinstance(instrument_formulation, Formulation):
                raise TypeError(
                    f"Expected instrument_formulation to be a single pyRVtest.Formulation "
                    f"when only one instrument set is provided. "
                    f"Received {type(instrument_formulation).__name__}. "
                    f"Fix: pass instrument_formulation=Formulation('0 + <instruments>')."
                )
        elif L > 1:
            if not all(isinstance(f, Formulation) for f in instrument_formulation):
                raise TypeError(
                    "Expected every entry in instrument_formulation to be a "
                    "pyRVtest.Formulation instance. "
                    "Received a sequence containing non-Formulation values. "
                    "Fix: wrap each instrument-set formula in Formulation(...)."
                )

        if endogenous_cost_component is not None:
            if not isinstance(endogenous_cost_component, str):
                raise TypeError(
                    f"Expected endogenous_cost_component to be a column-name string. "
                    f"Received {type(endogenous_cost_component).__name__}. "
                    f"Fix: pass the column name as a string."
                )
            _, w_formulation_check, _ = cost_formulation._build_matrix(product_data)
            endog_terms = [f for f in w_formulation_check if endogenous_cost_component in f.names]
            if not endog_terms:
                raise ValueError(
                    f"Expected endogenous_cost_component to appear in cost_formulation. "
                    f"Received endogenous_cost_component={endogenous_cost_component!r} "
                    f"but the parsed cost_formulation has no matching term. "
                    f"Fix: include '{endogenous_cost_component}' in cost_formulation "
                    f"(e.g., Formulation('0 + {endogenous_cost_component} + ...'))."
                )
            for f in endog_terms:
                if str(f) != endogenous_cost_component:
                    raise ValueError(
                        f"Expected endogenous_cost_component to enter "
                        f"cost_formulation linearly (no interactions or "
                        f"transformations). "
                        f"Received the term {str(f)!r} for "
                        f"endogenous_cost_component={endogenous_cost_component!r}. "
                        f"Fix: split the nonlinear term out of cost_formulation "
                        f"and include the plain linear column."
                    )

        # v0.4 OQ 14: Problem-level tax validation. Normalize the payer
        # string (maps 'firms' -> 'firm', 'consumers' -> 'consumer') so
        # downstream code sees one canonical spelling.
        advalorem_payer = _validate_problem_tax_kwargs(
            unit_tax=unit_tax,
            advalorem_tax=advalorem_tax,
            advalorem_payer=advalorem_payer,
        )
        # Column-existence checks: do them here (not in the validator)
        # because product_data is first available inside __init__.
        if unit_tax is not None and extract_matrix(product_data, unit_tax) is None:
            raise ValidationError(
                f"Expected product_data to contain the Problem-level "
                f"unit_tax column {unit_tax!r}. "
                f"Received product_data without that key. "
                f"Fix: add the column, or drop Problem(unit_tax=...)."
            )
        if advalorem_tax is not None and extract_matrix(product_data, advalorem_tax) is None:
            raise ValidationError(
                f"Expected product_data to contain the Problem-level "
                f"advalorem_tax column {advalorem_tax!r}. "
                f"Received product_data without that key. "
                f"Fix: add the column, or drop Problem(advalorem_tax=...)."
            )

        # v0.4 OQ 14: resolve known-coefficient cost shifters from the
        # cost formulation. The validator on Formulation already checked
        # types; here we check column existence in product_data and the
        # no-overlap-with-parsed-formula invariant. Store the resolved
        # arrays for prices_effective to consume.
        known_coef_arrays: List[Tuple[str, float, Array]] = []
        known_coefs = getattr(cost_formulation, 'known_coefficients', {}) or {}
        if known_coefs:
            # Authoritative check against parsed formula term names.
            _, w_formulation_parsed, _ = cost_formulation._build_matrix(product_data)
            formula_term_names: set = set()
            for term in w_formulation_parsed:
                formula_term_names.update(term.names)
            for col, coef in known_coefs.items():
                if col in formula_term_names:
                    raise ValidationError(
                        f"Expected known_coefficients[{col!r}] to name a "
                        f"column that is NOT already a parsed term in "
                        f"cost_formulation. "
                        f"Received {col!r} both in known_coefficients and "
                        f"as a parsed term of the formula. "
                        f"Fix: drop {col!r} from the formula (the "
                        f"known-coefficient mechanism adds it at a fixed "
                        f"coefficient), or drop it from "
                        f"known_coefficients."
                    )
                arr = extract_matrix(product_data, col)
                if arr is None:
                    raise ValidationError(
                        f"Expected product_data to contain the "
                        f"known-coefficient column {col!r} referenced "
                        f"by cost_formulation.known_coefficients. "
                        f"Received product_data without that key. "
                        f"Fix: add the column to product_data, or drop "
                        f"{col!r} from known_coefficients."
                    )
                # Promote to (N, 1) column; extract_matrix already does
                # this when the value is a 1-D vector, but be defensive.
                arr = np.asarray(arr, dtype=options.dtype).reshape(-1, 1)
                known_coef_arrays.append((col, float(coef), arr))

        products = Products(
            cost_formulation=cost_formulation, instrument_formulation=instrument_formulation, product_data=product_data
        )
        if markup_data is None:
            models_recarray = Models(
                models=models, product_data=product_data,
                problem_unit_tax=unit_tax,
                problem_advalorem_tax=advalorem_tax,
                problem_advalorem_payer=advalorem_payer,
            )
            markups = [None] * M
        else:
            models_recarray = None
            markups = markup_data

        super().__init__(products, models_recarray)

        self.cost_formulation = cost_formulation
        self.instrument_formulation = instrument_formulation
        # v0.4 OQ 14: keep Problem-level tax kwargs on self for repr /
        # inspection / downstream bookkeeping. Stored in normalized form.
        self._problem_unit_tax: Optional[str] = unit_tax
        self._problem_advalorem_tax: Optional[str] = advalorem_tax
        self._problem_advalorem_payer: Optional[str] = advalorem_payer
        # v0.4 OQ 14: resolved known-coefficient shifters consumed in
        # Problem.solve's prices_effective computation. List of
        # (column_name, gamma, (N, 1) array) tuples.
        self._known_coefficient_shifters: List[Tuple[str, float, Array]] = known_coef_arrays
        # Public attribute `model_formulations` kept for backward-compat callers
        # that inspect it; holds the legacy tuple when the user supplied one,
        # else None. The canonical internal representation is the ConductModel
        # / Vertical list stored as `self._models` (set below).
        self.model_formulations = _legacy_model_formulations
        self._models = models
        self.demand_results = demand_results
        self.demand_params = demand_params
        self._product_data_raw = product_data  # kept for demand_params path to access arbitrary columns
        self.markups = markups
        self.endogenous_cost_component = endogenous_cost_component

        # v0.4 step 4e: construct the demand backend exactly once at Problem
        # init. Downstream code (Problem.solve, _perturb_and_build_markups,
        # the unified compute_demand_adjustment in solve/demand_adjustment.py)
        # uses this single object. `None` only when neither demand_results nor
        # demand_params was supplied (e.g., user_supplied_markups-only path).
        self._demand_backend = self._construct_demand_backend()

        self.unique_market_ids = np.unique(self.products.market_ids.flatten())
        self.unique_nesting_ids = np.unique(self.products.nesting_ids.flatten())
        self.unique_product_ids = np.unique(self.products.product_ids.flatten())

        self.N = self.products.shape[0]
        self.T = self.unique_market_ids.size
        self.L = len(self.instrument_formulation) if hasattr(self.instrument_formulation, '__len__') else 1
        # v0.4 step 6a: initialize Dict_K as an instance attribute. The prior
        # class-level `Dict_K = {}` meant two concurrent Problem instances
        # would share the dict and accumulate each other's instrument counts.
        self.Dict_K = {}
        for instrument in range(self.L):
            self.Dict_K.update({"K{0}".format(instrument): self.products["Z{0}".format(instrument)].shape[1]})
        self.M = len(self._models) if self.markups[0] is None else np.shape(self.markups)[0]
        self.EC = self.products.cost_ids.shape[1]
        self.H = self.unique_nesting_ids.size

        self._market_indices = {t: i for i, t in enumerate(self.unique_market_ids)}
        self._product_market_indices = get_indices(self.products.market_ids)

        self._absorb_cost_ids = None
        if self.EC > 0:
            assert cost_formulation is not None
            self._absorb_cost_ids = cost_formulation._build_absorb(self.products.cost_ids)

        if max(options.collinear_atol, options.collinear_rtol) > 0:
            cost_shifters = self.products.w
            fix_hint = (
                "Fix: drop one of the collinear columns, or set "
                "options.collinear_atol = options.collinear_rtol = 0 to disable the check."
            )
            collinear, successful = precisely_identify_collinearity(cost_shifters)
            if not successful:
                raise ValueError(
                    "Expected the cost-shifter matrix w to admit a QR "
                    "decomposition for the collinearity diagnostic. "
                    "Received a matrix that QR could not factor (likely due to "
                    "NaN / inf entries). " + fix_hint
                )
            if collinear.any():
                raise ValueError(
                    "Expected the cost-shifter matrix w to be full column rank. "
                    "Received a matrix with collinear columns. " + fix_hint
                )
            for instrument in range(self.L):
                cost_shifters = self.products.w
                cost_shifters = np.append(cost_shifters, self.products["Z{0}".format(instrument)], axis=1)
                collinear, successful = precisely_identify_collinearity(cost_shifters)
                if not successful:
                    raise ValueError(
                        f"Expected the stacked [w, z{instrument}] matrix to admit "
                        f"a QR decomposition for the collinearity diagnostic. "
                        f"Received a matrix that QR could not factor. " + fix_hint
                    )
                if collinear.any():
                    raise ValueError(
                        f"Expected the stacked [w, z{instrument}] matrix to be "
                        f"full column rank. "
                        f"Received a matrix with collinear columns. " + fix_hint
                    )

        logger.info(f"Initialized the problem after {format_seconds(time.time() - start_time)}.")
        logger.info("")
        logger.info(str(self))

    def __str__(self) -> str:
        """Format economy information as a string."""
        return "\n\n".join([self._format_dimensions(), self._format_formulations(), self._format_model_formulations()])

    def _format_dimensions(self) -> str:
        """Format information about the nonzero dimensions of the economy as a string."""
        header = []
        values = []
        for key in ['T', 'N', 'M', 'L']:
            value = getattr(self, key)
            if value > 0:
                header.append(f" {key} ")
                values.append(str(value))
        for instrument in range(self.L):
            header.append("d_z{0}".format(instrument))
            values.append(str(self.Dict_K["K{0}".format(instrument)]))
        return format_table(header, values, title="Dimensions")

    def _format_formulations(self) -> str:
        """Format information about the formulations of the economy as a string."""
        named_formulations = [(self._w_formulation, "w: Marginal Cost")]
        for instruments in range(self.L):
            named_formulations.append((
                self.Dict_Z_formulation["_Z{0}_formulation".format(instruments)],
                "z{0}: Instruments".format(instruments)
            ))
        data = []
        for formulations, name in named_formulations:
            if any(formulations):
                data.append([name] + [str(f) for f in formulations])
        max_formulations = max(len(r[1:]) for r in data)
        header = ["Column Indices:"] + [f" {i} " for i in range(max_formulations)]
        return format_table(header, *data, title="Formulations")

    def _format_model_formulations(self) -> str:
        """Format information about the model formulations as a string."""
        data = []
        if self.markups[0] is None:
            data.append(["Model - Downstream"] + [self.models["models_downstream"][i] for i in range(self.M)])
            data.append(["Model - Upstream"] + [self.models["models_upstream"][i] for i in range(self.M)])
            data.append(["Firm IDs - Downstream"] + [self.models["firm_ids_downstream"][i] for i in range(self.M)])
            data.append(["Firm IDs - Upstream"] + [self.models["firm_ids_upstream"][i] for i in range(self.M)])
            data.append(["VI Index"] + [self.models["vertical_integration_index"][i] for i in range(self.M)])
            data.append(["Cost Scaling Column"] + [self.models["cost_scaling_column"][i] for i in range(self.M)])
            data.append(["Unit Tax"] + [self.models["unit_tax_name"][i] for i in range(self.M)])
            data.append(["Advalorem Tax"] + [self.models["advalorem_tax_name"][i] for i in range(self.M)])
            data.append(["Advalorem Payer"] + [self.models["advalorem_payer"][i] for i in range(self.M)])
            data.append(
                ["User Supplied Markups"] + [self.models["user_supplied_markups_name"][i] for i in range(self.M)]
            )
            header = [" "] + [f" {i} " for i in range(self.M)]
        else:
            data.append(["Markups Supplied by User"])
            header = [" "]
        return format_table(header, *data, title="Models")

    def solve(
            self, demand_adjustment: Optional[bool] = False,
            clustering_adjustment: Optional[bool] = False,
            costs_type: Optional[str] = 'linear',
            mc_correction: Optional[Array] = None
    ) -> ProblemResults:
        r"""Solve the problem.

        Given demand estimates from PyBLP, we compute implied markups for each model :math:`m` being tested. Marginal
        cost is a linear function of observed cost shifters and an unobserved shock.

        The rest of the testing procedure is done for each pair of models, for each set of instruments. A GMM measure of
        fit is computed for each model-instrument pair. This measure of fit is used to construct the test statistic.

        Parameters
        ----------
        demand_adjustment: Optional[bool]
            (optional, default is False) Configuration that allows user to specify whether to compute a two-step demand
            adjustment. Options are True or False.
        clustering_adjustment: Optional[str]
            (optional, default is unadjusted) Configuration that specifies whether to compute clustered standard errors.
            Options are True or False.
        costs_type: Optional[str]
            (optional, default is ``'linear'``) Functional form for marginal cost. ``'linear'`` uses
            :math:`p - \text{markup}` directly; ``'log'`` substitutes :math:`\log(p - \text{markup})` before
            running the test. Only effective when ``demand_adjustment=False``; ignored otherwise. Raises
            ``ValueError`` if any implied marginal cost is non-positive under ``'log'``.
        mc_correction: Optional[Array]
            *Deprecated since v0.4; will be removed in v0.6.* User-supplied additive correction added to
            ``marginal_cost`` before the test. Must have the same shape as ``problem.markups`` (``M`` x ``N``).
            For supported cost-side adjustments, pass ``endogenous_cost_component`` at :class:`Problem`
            construction time instead.

        Returns
        -------
        `ProblemResults`
            :class:`ProblemResults` of the solved problem.

        Examples
        --------
        >>> import pyRVtest  # doctest: +SKIP
        >>> # Requires a constructed Problem — see pyRVtest.Problem
        >>> # for the full setup with a fitted pyblp.ProblemResults.
        >>> results = problem.solve(demand_adjustment=False)  # doctest: +SKIP
        """

        # v0.4 step 8e: this method is an orchestrator. Each stage lives in
        # ``pyRVtest/solve/*.py``:
        #   1. ``solve.markups.compute``        -> per-model markups via the demand backend
        #   2. (inline)                         -> tax-adjustment + log / mc_correction bookkeeping
        #   3. ``solve.endogenous_cost.iv_correct`` + ``solve.orthogonalize.residualize``
        #                                         -> 2SLS IV correction + FWL residualization
        #   4. ``solve.demand_adjustment.compute_demand_adjustment``
        #                                         -> first-stage demand-side correction
        #   5. ``solve.test_engine.compute_instrument_results``
        #                                         -> RV / F / MCS per instrument set
        logger.info("Solving the problem ...")
        step_start_time = time.time()

        self._validate_solve_args(demand_adjustment, clustering_adjustment)

        M = self.M
        N = self.N
        L = self.L
        markups = self.markups
        critical_values_power, critical_values_size = read_critical_values_tables()

        # -----------------------------------------------------------------
        # Stage 1: build markups via the demand backend.
        # -----------------------------------------------------------------
        markups_upstream = np.zeros(M, dtype=options.dtype)
        markups_downstream = np.zeros(M, dtype=options.dtype)
        if markups[0] is None:
            logger.info('Computing Markups ...')
            markups, markups_downstream, markups_upstream = _markups_stage(self)

        # -----------------------------------------------------------------
        # Stage 2: apply taxes, cost scaling, log-costs, user mc_correction.
        # -----------------------------------------------------------------
        advalorem_tax_adj = [None] * M
        prices_effective = [None] * M
        markups_effective = [None] * M
        marginal_cost = self.products.prices - markups
        unit_tax = self.models["unit_tax"]
        advalorem_tax = self.models["advalorem_tax"]
        cost_scaling = self.models["cost_scaling"]
        # v0.4 OQ 14: sum of known-coefficient cost shifters (gamma_k *
        # x_k). Applied uniformly to every model m: known-coefficient
        # shifters are DGP-level (Dearing et al. 2026), not model-level
        # behavioral choices, so they do not carry a salience flag.
        # Summed once outside the model loop for efficiency.
        known_coef_sum = None
        if self._known_coefficient_shifters:
            known_coef_sum = sum(
                coef * arr for (_col, coef, arr) in self._known_coefficient_shifters
            )
        for m in range(M):
            condition = self.models["advalorem_payer"][m] == "consumer"
            advalorem_tax_adj[m] = 1 / (1 + advalorem_tax[m]) if condition else (1 - advalorem_tax[m])
            prices_effective[m] = (advalorem_tax_adj[m] * self.products.prices / (1 + cost_scaling[m]) - unit_tax[m])
            if known_coef_sum is not None:
                prices_effective[m] = prices_effective[m] - known_coef_sum
            markups_effective[m] = (advalorem_tax_adj[m] / (1 + cost_scaling[m])) * markups[m]
            marginal_cost[m] = prices_effective[m] - markups_effective[m]

        if costs_type == "log" and not demand_adjustment:
            if np.any(marginal_cost < 0):
                raise ValueError(
                    "Expected all implied marginal costs to be positive when "
                    "costs_type='log' (log is undefined for <= 0). "
                    "Received at least one negative marginal cost (price - markup < 0). "
                    "Fix: switch to costs_type='linear', or inspect the candidate "
                    "conduct models whose markups exceed price."
                )
            marginal_cost = np.log(marginal_cost)

        if mc_correction is not None:
            global _mc_correction_deprecation_warned
            if not _mc_correction_deprecation_warned:
                warnings.warn(
                    _MC_CORRECTION_DEPRECATION_MSG,
                    DeprecationWarning,
                    stacklevel=2,
                )
                _mc_correction_deprecation_warned = True
            if marginal_cost.shape != mc_correction.shape:
                raise ValueError(
                    f"Expected mc_correction to match the marginal-cost shape "
                    f"(M, N). "
                    f"Received mc_correction.shape={mc_correction.shape}, "
                    f"marginal_cost.shape={marginal_cost.shape}. "
                    f"Fix: pass an mc_correction array with the same shape as "
                    f"problem.markups."
                )
            marginal_cost = marginal_cost + mc_correction

        # -----------------------------------------------------------------
        # Stage 3: endogenous-cost IV correction + orthogonalize.
        # When ``endogenous_cost_component`` is set, one IV correction per
        # instrument set; without it, a single orthogonalization is shared
        # across all instrument sets.
        # -----------------------------------------------------------------
        cost_param = None
        if self.endogenous_cost_component is not None:
            logger.info('Computing IV correction for endogenous cost component ...')
            marginal_cost_base = marginal_cost.copy()
            cost_param = [None] * L
            omega_per_instrument = [None] * L
            tau_list_per_instrument = [None] * L
            endog_hat_per_instrument = [None] * L

            for l in range(L):
                cp_l, mc_corr_l, endog_hat_l = _iv_correct_stage(self, l, M, N, marginal_cost_base)
                mc_l = marginal_cost_base.copy()
                for m in range(M):
                    mc_l[m] = mc_l[m] + mc_corr_l[m]
                mo_l, omega_l, tau_l = _residualize_stage(self, M, N, markups_effective, mc_l)
                if l == 0:
                    markups_orthogonal = mo_l  # identical across instrument sets; keep first
                    marginal_cost = mc_l        # store first instrument's corrected MC for results
                    tau_list = tau_l
                cost_param[l] = cp_l
                omega_per_instrument[l] = omega_l
                tau_list_per_instrument[l] = tau_l
                endog_hat_per_instrument[l] = endog_hat_l
        else:
            markups_orthogonal, omega, tau_list = _residualize_stage(
                self, M, N, markups_effective, marginal_cost
            )
            omega_per_instrument = [omega] * L
            endog_hat_per_instrument = [None] * L
            tau_list_per_instrument = None

        # -----------------------------------------------------------------
        # Stage 4: first-stage demand-side correction (optional).
        # -----------------------------------------------------------------
        gradient_markups = H_prime_wd = H = h_i = h = None
        gradient_gamma_per_instrument = None
        if demand_adjustment:
            # v0.4 step 4e: single code path via the unified function in
            # solve/demand_adjustment.py. Closes the silent capability gap
            # where the analytical path previously returned
            # gradient_gamma_per_instrument=None.
            from .solve.demand_adjustment import compute_demand_adjustment
            mc_base_for_grad = (
                marginal_cost_base if self.endogenous_cost_component is not None else None
            )
            (
                gradient_markups, H_prime_wd, H, h_i, h, gradient_gamma_per_instrument
            ) = compute_demand_adjustment(
                self._demand_backend, self, M, N, markups,
                advalorem_tax_adj, cost_scaling, mc_base_for_grad,
            )

        # -----------------------------------------------------------------
        # Stage 5: test engine — RV, F, MCS per instrument set.
        # -----------------------------------------------------------------
        g_list = [None] * L
        Q_list = [None] * L
        RV_numerator_list = [None] * L
        RV_denominator_list = [None] * L
        test_statistic_RV_list = [None] * L
        F_statistic_list = [None] * L
        unscaled_F_statistic_list = [None] * L
        MCS_p_values_list = [None] * L
        rho_list = [None] * L
        F_cv_size_list = [None] * L
        F_cv_power_list = [None] * L
        symbols_size_list = [None] * L
        symbols_power_list = [None] * L

        for instrument in range(L):
            grad_gamma_l = (gradient_gamma_per_instrument[instrument]
                            if gradient_gamma_per_instrument is not None else None)
            r = _compute_instrument_results_stage(
                self, instrument, M, N, omega_per_instrument[instrument], demand_adjustment, gradient_markups,
                H_prime_wd, H, h_i, h, clustering_adjustment,
                critical_values_size, critical_values_power, endog_hat_per_instrument[instrument],
                grad_gamma_l,
            )
            g_list[instrument] = r['g']
            Q_list[instrument] = r['Q']
            RV_numerator_list[instrument] = r['RV_numerator']
            RV_denominator_list[instrument] = r['RV_denominator']
            test_statistic_RV_list[instrument] = r['rv_test_statistic']
            F_statistic_list[instrument] = r['F']
            unscaled_F_statistic_list[instrument] = r['unscaled_F']
            MCS_p_values_list[instrument] = r['mcs_pvalues']
            rho_list[instrument] = r['rho']
            F_cv_size_list[instrument] = r['F_cv_size']
            F_cv_power_list[instrument] = r['F_cv_power']
            symbols_size_list[instrument] = r['symbols_size']
            symbols_power_list[instrument] = r['symbols_power']

        results = ProblemResults(Progress(
            self, markups, markups_downstream, markups_upstream, markups_orthogonal, marginal_cost,
            tau_list, g_list, Q_list, RV_numerator_list, RV_denominator_list,
            test_statistic_RV_list, F_statistic_list, MCS_p_values_list, rho_list, unscaled_F_statistic_list,
            F_cv_size_list, F_cv_power_list, symbols_size_list, symbols_power_list, cost_param,
            tau_list_per_instrument
        ))
        logger.info(f"Solved the problem after {format_seconds(time.time() - step_start_time)}.")
        logger.info("")
        logger.info(str(results))
        return results

    def _validate_solve_args(self, demand_adjustment: bool, clustering_adjustment: bool) -> None:
        """Validate arguments passed to solve."""
        if not isinstance(demand_adjustment, bool):
            raise TypeError(
                f"Expected demand_adjustment to be True or False. "
                f"Received {type(demand_adjustment).__name__}. "
                f"Fix: pass demand_adjustment=True or False."
            )
        if not isinstance(clustering_adjustment, bool):
            raise TypeError(
                f"Expected clustering_adjustment to be True or False. "
                f"Received {type(clustering_adjustment).__name__}. "
                f"Fix: pass clustering_adjustment=True or False."
            )
        if clustering_adjustment and np.shape(self.products.clustering_ids)[1] != 1:
            raise ValueError(
                "Expected product_data to contain a 'clustering_ids' column "
                "when clustering_adjustment=True. "
                "Received product_data without a (one-dimensional) clustering_ids column. "
                "Fix: add 'clustering_ids' to product_data, or set "
                "clustering_adjustment=False."
            )
        # demand_adjustment + endogenous_cost_component is now supported; the gradient
        # accounts for the dependence of gamma_m on theta via per-instrument finite differences.
        if demand_adjustment and self.demand_params is not None:
            # Validate that demand adjustment extras are provided
            dp = self.demand_params
            missing = []
            if 'beta' not in dp:
                missing.append('beta')
            if 'demand_instrument_columns' not in dp:
                missing.append('demand_instrument_columns')
            if 'x_columns' not in dp:
                missing.append('x_columns')
            if missing:
                raise ValueError(
                    f"Expected demand_params to carry demand-adjustment state "
                    f"(beta, x_columns, demand_instrument_columns) when "
                    f"Problem.solve(demand_adjustment=True) is called. "
                    f"Received demand_params missing: {', '.join(missing)}. "
                    f"Fix: add the missing keys to demand_params, or call "
                    f"solve(demand_adjustment=False) (these inputs are only needed "
                    f"for the first-stage correction)."
                )
        for m in range(self.M):
            if self._models[m].user_supplied_markups is not None:
                if demand_adjustment:
                    raise ValueError(
                        "Expected no user-supplied markups when "
                        "demand_adjustment=True (the finite-difference gradient "
                        "needs a demand system to perturb, which user-supplied "
                        "markups bypass). "
                        "Received at least one model with user_supplied_markups set. "
                        "Fix: set demand_adjustment=False when passing "
                        "user_supplied_markups, or compute markups from a demand "
                        "system instead."
                    )

    def _construct_demand_backend(self):
        """Build the backend from demand_results or demand_params at Problem init time.

        v0.4 step 4e: single construction point for the backend. Returns `None` when
        neither demand_results nor demand_params is supplied (user_supplied_markups-only
        path). Otherwise:
          - `demand_results is not None`: route to ``NestedLogitBackend`` when
            single-scalar-rho nested logit is detected (precision gain in
            ``d(D)/d(rho)``; pyblp finite-diff has O(eps^2) error there).
            Otherwise ``PyBLPBackend`` — plain logit sees no precision gain
            (pyblp finite-diff of ``D/alpha`` is exact because ``compute_delta``
            restores shares); per-nest rho and BLP cannot be represented by
            the analytical backends.
          - `demand_params is not None` with non-empty nonzero rho -> `NestedLogitBackend`.
          - `demand_params is not None` with empty or all-zero rho -> `LogitBackend`.

        Demand-adjustment state (beta, x_columns, demand_instrument_columns, W_demand)
        is forwarded when present so `SupportsDemandAdjustment` methods work. The raw
        `product_data` (not the structured `Products` recarray) is passed because the
        backends need access to arbitrary columns like `nesting_ids`, `x1`, etc.
        """
        if self.demand_results is not None:
            return self._backend_from_demand_results(self.demand_results)

        if self.demand_params is not None:
            dp = self.demand_params
            rho_nonzero = [s for s in dp.get('rho', []) if s > 0]
            shared_kwargs = dict(
                alpha=dp['alpha'],
                product_data=self._product_data_raw,
                beta=dp.get('beta'),
                x_columns=dp.get('x_columns'),
                demand_instrument_columns=dp.get('demand_instrument_columns'),
                W_demand=dp.get('W_demand'),
            )
            if rho_nonzero:
                from .backends.nested_logit import NestedLogitBackend
                return NestedLogitBackend(
                    rho=dp.get('rho', []),
                    nesting_ids_columns=dp.get('nesting_ids_columns'),
                    **shared_kwargs,
                )
            from .backends.logit import LogitBackend
            return LogitBackend(**shared_kwargs)

        return None

    def _backend_from_demand_results(self, r):
        """Route pyblp ProblemResults to the best-matched backend.

        * **Plain logit** (``K2==0`` and ``r.rho.size==0``): try routing
          to ``LogitBackend``. Precision gain is modest (pyblp's finite-
          diff of ``d(D)/d(alpha)`` is ~O(1e-9) accurate because D is
          linear in alpha at fixed shares) but routing keeps the code
          path consistent with the nested case.

        * **Single-scalar-rho nested logit** (``K2==0`` and ``r.rho.size==1``):
          try routing to ``NestedLogitBackend(rho=[rho])``. Analytical
          ``d(D)/d(rho)`` is exact; pyblp's finite-diff has genuine
          O(eps^2) truncation error because D is nonlinear in rho.
          Precision gain is material.

        * **Per-nest rho** (``K2==0`` and ``r.rho.size>1``): stay on
          ``PyBLPBackend``. AFSSZ L=1 formulation has one rho (nesting
          parameter); pyblp's Cardell-Nevo formulation has one rho per
          nest. The derivatives ``d(D)/d(rho_h)`` don't match the
          AFSSZ ``d(D)/d(rho_1)``.

        * **BLP** (``K2>0``): stay on ``PyBLPBackend``. No analytical
          ``d(D)/d(theta)`` through the BLP contraction mapping.

        Falls back to ``PyBLPBackend`` if the raw product_data doesn't
        carry the columns the analytical backend needs (e.g., pyblp-
        generated fixed-effect dummies).
        """
        from .backends.pyblp import PyBLPBackend
        K2 = r.problem.K2
        if K2 > 0:
            return PyBLPBackend(r)
        rho_arr = np.atleast_1d(np.asarray(r.rho).flatten())
        # Per-nest rho -> stay on PyBLPBackend.
        if rho_arr.size > 1:
            return PyBLPBackend(r)

        # Plain logit (rho.size == 0) or single-scalar-rho nested logit
        # (rho.size == 1): try analytical. Extract shared state.
        beta_labels = list(r.beta_labels)
        if 'prices' not in beta_labels:
            return PyBLPBackend(r)
        price_idx = beta_labels.index('prices')
        alpha = float(np.asarray(r.beta).flatten()[price_idx])
        x_columns = [lab for lab in beta_labels if lab != 'prices']
        beta_full = np.asarray(r.beta).flatten()
        beta_nonprice = np.asarray(
            [b for b, lab in zip(beta_full, beta_labels) if lab != 'prices']
        )

        # pyblp's ZD dtype does not carry sub-column names as a 3-tuple; we
        # need names to feed `demand_instrument_columns`. Infer from the
        # raw product_data: `demand_instrumentsN` columns + non-price X1
        # columns, matching pyblp's default ZD construction. If the inferred
        # count does not match the actual ZD width, the user passed a
        # custom ZD formulation and we should bail.
        raw = self._product_data_raw
        raw_cols = self._raw_product_data_columns(raw)
        excluded_instrument_cols = sorted(
            [c for c in raw_cols if str(c).startswith('demand_instruments')]
        )
        inferred_zd_cols = excluded_instrument_cols + x_columns
        expected_n_zd = r.problem.products.ZD.shape[1]
        if len(inferred_zd_cols) != expected_n_zd:
            return PyBLPBackend(r)

        # pyblp's `Formulation('1 + ...')` packs a literal '1' column in X1 / ZD.
        # Synthesize it if that's the only missing column.
        needed = set(x_columns + inferred_zd_cols)
        missing = needed - raw_cols
        if missing == {'1'}:
            raw = self._augment_with_intercept_column(raw)
        elif missing:
            return PyBLPBackend(r)

        weight_choice = getattr(options, 'demand_adjustment_weight', 'W')
        W_demand = r.W if weight_choice == 'W' else r.updated_W

        shared_kwargs = dict(
            alpha=alpha,
            product_data=raw,
            beta=beta_nonprice,
            x_columns=x_columns,
            demand_instrument_columns=inferred_zd_cols,
            W_demand=W_demand,
        )

        if rho_arr.size == 1:
            from .backends.nested_logit import NestedLogitBackend
            return NestedLogitBackend(
                rho=[float(rho_arr[0])],
                nesting_ids_columns=None,  # backend infers from product_data
                **shared_kwargs,
            )
        from .backends.logit import LogitBackend
        return LogitBackend(**shared_kwargs)

    @staticmethod
    def _raw_product_data_columns(raw):
        """Return the column names of the raw product_data regardless of whether
        it's a DataFrame, structured recarray, or dict.
        """
        if hasattr(raw, 'columns'):
            return set(map(str, raw.columns))
        if hasattr(raw, 'dtype') and raw.dtype.names:
            return set(raw.dtype.names)
        if hasattr(raw, 'keys'):
            return set(raw.keys())
        return set()

    @staticmethod
    def _augment_with_intercept_column(raw):
        """Return a copy of raw product_data with a '1' column (all ones).

        pyblp's `Formulation('1 + ...')` packs a literal '1' column into X1/ZD.
        Users typically don't include a column literally named '1' in their
        product_data; synthesize it for the analytical-backend path. Never
        mutates the input.
        """
        import pandas as pd
        if hasattr(raw, 'assign'):  # DataFrame
            return raw.assign(**{'1': 1.0})
        if hasattr(raw, 'dtype') and raw.dtype.names:
            df = pd.DataFrame({name: raw[name] for name in raw.dtype.names})
            df['1'] = 1.0
            return df
        if hasattr(raw, 'keys'):
            out = {k: raw[k] for k in raw.keys()}
            any_col = next(iter(out.values()))
            out['1'] = np.ones(len(any_col))
            return out
        raise TypeError(
            f"pyRVtest internal error: expected product_data to be a pandas "
            f"DataFrame, a structured numpy array, or a dict-like mapping for "
            f"intercept augmentation. "
            f"Received {type(raw).__name__}."
        )

    def _perturb_and_build_markups(self):
        """Build per-model markups using this Problem's demand backend.

        Thin delegation to :func:`pyRVtest.solve.markups.compute` after
        the v0.4 step 8b extraction. Kept as a method for backward-
        compatible access from test fixtures and
        :func:`pyRVtest.solve.passthrough.build_passthrough`.
        """
        return _markups_stage(self)

    def _prepare_orthogonal_variables(self, M: int, N: int, markups_effective: list, marginal_cost: Array):
        """Absorb fixed effects and residualize markups and marginal costs w.r.t. cost shifters.

        Thin delegation to :func:`pyRVtest.solve.orthogonalize.residualize`
        after the v0.4 step 8a extraction. Kept as a method for
        backward-compatible access from subclasses / tests.
        """
        return _residualize_stage(self, M, N, markups_effective, marginal_cost)

    def _compute_instrument_results(
            self, instrument: int, M: int, N: int, omega: Array,
            demand_adjustment: bool, gradient_markups: Optional[Array],
            H_prime_wd: Optional[Array], H: Optional[Array],
            h_i: Optional[Array], h: Optional[Array],
            clustering_adjustment: bool,
            critical_values_size: Array, critical_values_power: Array,
            endog_hat: Optional[Array] = None,
            gradient_gamma: Optional[Array] = None
    ) -> dict:
        """Compute all test statistics for a single instrument set.

        Thin delegation to
        :func:`pyRVtest.solve.test_engine.compute_instrument_results`
        after the v0.4 step 8d extraction. Kept as a method for
        backward-compatible access from subclasses / tests.
        """
        return _compute_instrument_results_stage(
            self, instrument, M, N, omega, demand_adjustment,
            gradient_markups, H_prime_wd, H, h_i, h,
            clustering_adjustment, critical_values_size, critical_values_power,
            endog_hat, gradient_gamma,
        )

    def _compute_mcs(
            self, rv_test_statistic: Array, sigma_mcs: Array,
            model_confidence_set_variance: Array, M: int, all_model_combinations: list
    ) -> Array:
        """Compute model confidence set p-values by iteratively eliminating the worst-fitting model.

        Thin delegation to :func:`pyRVtest.solve.test_engine.compute_mcs`
        after the v0.4 step 8d extraction. Kept as a method for
        backward-compatible access from subclasses / tests.
        """
        return _compute_mcs_stage(
            rv_test_statistic, sigma_mcs, model_confidence_set_variance, M, all_model_combinations,
        )

    def _compute_iv_correction(self, instrument: int, M: int, N: int, marginal_cost: list):
        """Run per-model 2SLS to estimate the coefficient on the endogenous cost component.

        Thin delegation to
        :func:`pyRVtest.solve.endogenous_cost.iv_correct` after the v0.4
        step 8c extraction. Kept as a method for backward-compatible
        access from :func:`pyRVtest.solve.demand_adjustment.compute_demand_adjustment`.
        """
        return _iv_correct_stage(self, instrument, M, N, marginal_cost)

    def _compute_block_gram(self, N: int, clustering_adjustment: bool, var: Array) -> Array:
        """Compute the (M*K, M*K) block Gram matrix for all model pairs at once.

        Thin delegation to
        :func:`pyRVtest.solve.test_engine.compute_block_gram` after the
        v0.4 step 8d extraction. Kept as a method for backward-compatible
        access from subclasses / tests.
        """
        return _compute_block_gram_stage(self, N, clustering_adjustment, var)

    @staticmethod
    def _extract_block(gram: Array, i: int, m: int, K: int) -> Array:
        """Extract a (K, K) block from the block Gram matrix.

        Thin delegation to :func:`pyRVtest.solve.test_engine.extract_block`
        after the v0.4 step 8d extraction.
        """
        return _extract_block_stage(gram, i, m, K)
