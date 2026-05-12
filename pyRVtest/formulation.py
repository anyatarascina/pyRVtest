"""Formulation of data matrices and absorption of fixed effects."""

import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, Union

import numpy as np

from pyblp.utilities.basics import Array  # noqa: F401
from pyblp.configurations.formulation import Absorb, Formulation as _PyblpFormulation

from .exceptions import ValidationError


__all__ = ['Absorb', 'Formulation', 'ModelFormulation']


class Formulation(_PyblpFormulation):  # type: ignore[misc]  # pyblp has no stubs
    r"""pyRVtest Formulation: PyBLP's Formulation plus known-coefficient cost shifters.

    pyRVtest subclasses :class:`pyblp.Formulation` to add
    ``known_coefficients`` that lets users
    specify cost shifters with known (non-estimated) coefficients.
    These shifters enter the effective-price computation in
    :meth:`Problem.solve` directly, in the same slot as per-unit
    taxes: ``prices_effective = advalorem_tax_adj * p / (1 +
    cost_scaling) - unit_tax - sum(gamma_k * known_shifter_k)``.

    Unlike regular cost shifters in the formula string (which are
    estimated into the cost function and residualized out), known-
    coefficient shifters enter with a coefficient supplied by the
    researcher. Dearing, Magnolfi, Quint, Sullivan, and Waldfogel
    (2024) work with a general class of such shifters; per-unit taxes
    are the leading special case.

    Parameters
    ----------
    formula : str
        R-style formula; same semantics as :class:`pyblp.Formulation`.
    absorb : str, optional
        Forwarded to :class:`pyblp.Formulation`.
    absorb_method : str, optional
        Forwarded to :class:`pyblp.Formulation`.
    absorb_options : dict, optional
        Forwarded to :class:`pyblp.Formulation`.
    known_coefficients : dict, optional
        Mapping from column name (str) to numeric scalar coefficient
        ``gamma_k``. Each column in ``known_coefficients`` must be in
        ``product_data`` (checked when ``Problem`` is constructed;
        ``Formulation`` alone does not know the data). Columns must
        NOT appear in ``formula`` — doing so is redundant and would
        specify two coefficients for the same column. Values must be
        finite.

    Examples
    --------
    >>> from pyRVtest import Formulation
    >>> f = Formulation('0 + w', known_coefficients={'input_price': 0.75})
    >>> f.known_coefficients
    {'input_price': 0.75}
    """

    def __init__(
            self,
            formula: str,
            absorb: Optional[str] = None,
            absorb_method: Optional[str] = None,
            absorb_options: Optional[Mapping[str, Any]] = None,
            known_coefficients: Optional[Dict[str, Union[float, int]]] = None,
    ) -> None:
        super().__init__(
            formula=formula,
            absorb=absorb,
            absorb_method=absorb_method,
            absorb_options=absorb_options,
        )
        self.known_coefficients: Dict[str, float] = _validate_known_coefficients(
            known_coefficients, formula,
        )


def _validate_known_coefficients(
        known_coefficients: Optional[Dict[str, Union[float, int]]],
        formula: str,
) -> Dict[str, float]:
    """Validate ``known_coefficients`` at ``Formulation`` construction.

    Returns a new dict of str-keyed float values. An unset / ``None`` /
    empty dict resolves to an empty dict. This function checks types
    and numeric finiteness; the column-existence check lives at
    ``Problem.__init__`` where ``product_data`` is available, and the
    no-overlap-with-formula check below is a lightweight textual
    heuristic (full patsy-term inspection also runs at
    ``Problem.__init__`` via ``cost_formulation._build_matrix``).
    """
    if known_coefficients is None:
        return {}
    if not isinstance(known_coefficients, dict):
        raise ValidationError(
            f"Expected known_coefficients to be a dict of "
            f"{{column_name: coefficient}}. "
            f"Received {type(known_coefficients).__name__} "
            f"({known_coefficients!r}). "
            f"Fix: pass known_coefficients={{'col': 0.75}} or omit the "
            f"argument."
        )
    validated: Dict[str, float] = {}
    for col, coef in known_coefficients.items():
        if not isinstance(col, str) or not col:
            raise ValidationError(
                f"Expected every key of known_coefficients to be a "
                f"non-empty column-name string. "
                f"Received {col!r} (type "
                f"{type(col).__name__}). "
                f"Fix: use the name of the column in product_data as "
                f"a string key."
            )
        # Reject bool explicitly (isinstance(True, int) is True).
        if isinstance(coef, bool) or not isinstance(coef, (int, float)):
            raise ValidationError(
                f"Expected known_coefficients[{col!r}] to be a numeric "
                f"scalar (float or int). "
                f"Received {type(coef).__name__} ({coef!r}). "
                f"Fix: pass a numeric coefficient, e.g. "
                f"known_coefficients={{{col!r}: 1.0}}."
            )
        if not np.isfinite(coef):
            raise ValidationError(
                f"Expected known_coefficients[{col!r}] to be a finite "
                f"numeric scalar. "
                f"Received {coef!r}. "
                f"Fix: pass a finite numeric coefficient."
            )
        # Textual heuristic: bail early when a known-coef column is
        # obviously already a term in the formula (matches whole-word).
        # Full validation happens against parsed terms at
        # Problem.__init__ time (see problem.py).
        if _column_in_formula_literal(col, formula):
            raise ValidationError(
                f"Expected known_coefficients[{col!r}] to name a "
                f"column that is NOT already a term in the formula "
                f"{formula!r} (specifying both would double-count "
                f"the column). "
                f"Received {col!r} both in known_coefficients and in "
                f"the formula. "
                f"Fix: drop {col!r} from the formula (the "
                f"known-coefficient mechanism adds it at a fixed "
                f"coefficient), or drop it from known_coefficients "
                f"(the formula estimates a free coefficient)."
            )
        validated[col] = float(coef)
    return validated


def _column_in_formula_literal(col: str, formula: str) -> bool:
    """Heuristic: does ``col`` appear as a whole-word token in ``formula``?

    Catches the common error case where a user writes
    ``Formulation('0 + input_price', known_coefficients={'input_price': 1})``.
    Not exhaustive (a parsed patsy term with an ``I(input_price**2)``
    wrapper would evade this); the authoritative check lives at
    ``Problem.__init__`` against the parsed ColumnFormulation names,
    but catching the easy case here gives users a clean
    ``Formulation(...)`` error rather than a ``Problem(...)`` one.
    """
    import re
    pattern = r'(?<![A-Za-z0-9_])' + re.escape(col) + r'(?![A-Za-z0-9_])'
    return re.search(pattern, formula) is not None


_MODELFORMULATION_DEPRECATION_MSG = (
    "ModelFormulation is deprecated and will be removed in v0.6. Migrate to "
    "the class-based ConductModel API: use pyRVtest.Bertrand, "
    "pyRVtest.Cournot, pyRVtest.Monopoly, pyRVtest.PerfectCompetition, "
    "pyRVtest.MixCournotBertrand, pyRVtest.PartialCollusion, or "
    "pyRVtest.CustomConductModel for single-tier conduct, and "
    "pyRVtest.Vertical(downstream=..., upstream=..., ...) for bilateral "
    "oligopoly. Pass the resulting list to Problem via the `models=` "
    "keyword instead of `model_formulations=`. See "
    "docs/migrating_to_v0.4.rst for per-case examples."
)


class ModelFormulation(object):
    r"""Configuration for designing matrices and absorbing fixed effects.

    For each model, the user can specify the downstream and upstream (optional) models, the downstream and upstream
    ownership structure, a custom model and markup formula, and vertical integration. The user can also choose to forgo
    markup computation and specify their own markups with `user_supplied_markups`. Additionally, there are
    specifications related to testing conduct with taxes.

    There is a built-in library of models that the researcher can choose from.

    Here, we have another difference with PyBLP.  In PyBLP, if one wants to build an ownership matrix, there must be a
    variable called `firm_id` in the `product_data`.  With pyRVtest, the researcher can pass any variable in the
    `product_data` as `ownership_downstream` and from this, the ownership matrix in each market will be built.

    .. note::
        We are working on adding additional models to this library as well as options for the researcher to specify
        their own markup function.)

    Parameters
    ----------
    model_downstream : `str, optional`
        The model of conduct for downstream firms (or if no vertical structure, the model of conduct). One of
        "bertrand", "cournot", "monopoly", "perfect_competition", or "other".
    model_upstream : `str, optional`
        The model of conduct for upstream firms. One of "bertrand", "cournot", "monopoly", "perfect_competition", or
        "other".
    ownership_downstream: `str, optional`
        Column indicating which firm ids to use for ownership matrix construction for downstream firms.
    ownership_upstream: `str, optional`
        Column indicating which firm ids to use for ownership matrix construction for upstream firms.
    custom_model_specification: `dict, optional`
        A dictionary containing an optional custom markup formula specified by the user. The specified function must
        consist of objects computed within the package.
    vertical_integration: `str, optional`
        The column name for the data column which indicates the vertical ownership structure.
    unit_tax: `str, optional`
        The column name for the vector containing information on unit taxes.
    advalorem_tax: `str, optional`
        The column name for the vector containing information on advalorem taxes.
    advalorem_payer: `str, optional`
        A string indicating who pays for the advalorem tax in the given model.
    cost_scaling: `str, optional`
        The column name for the cost scaling parameter.
    kappa_specification_downstream: `Union[str, Callable[[Any, Any], float]]], optional`
        Information on the degree of cooperation among downstream firms for each market.
    kappa_specification_upstream: `Union[str, Callable[[Any, Any], float]]], optional`
        Information on the degree of cooperation among upstream firms for each market.
    user_supplied_markups: `str, optional`
        The name of the column containing user-supplied markups.
    mix_flag: `str, optional`
        Column name for a boolean vector indicating which products compete in prices (True = Bertrand, False = Cournot)
        within each market. Required when `model_downstream='mix_cournot_bertrand'`.

    Examples
    --------
    ``ModelFormulation`` is the legacy string-based API; new
    code should prefer the class-based :class:`pyRVtest.Bertrand`,
    :class:`pyRVtest.Cournot`, etc. The legacy form still works and
    emits a ``DeprecationWarning`` on first construction:

    >>> import warnings
    >>> from pyRVtest import ModelFormulation
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('ignore', DeprecationWarning)
    ...     mf = ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids')
    >>> mf._model_downstream
    'bertrand'
    >>> mf._ownership_downstream
    'firm_ids'
    """

    _model_downstream: Optional[str]
    _model_upstream: Optional[str]
    _ownership_downstream: Optional[str]
    _ownership_upstream: Optional[str]
    _custom_model_specification: Optional[Dict[str, Any]]
    _vertical_integration: Optional[str]
    _unit_tax: Optional[str]
    _advalorem_tax: Optional[str]
    _advalorem_payer: Optional[str]
    _cost_scaling: Optional[str]
    _kappa_specification_downstream: Optional[Union[str, Callable[[Any, Any], float]]]
    _kappa_specification_upstream: Optional[Union[str, Callable[[Any, Any], float]]]
    _user_supplied_markups: Optional[str]
    _mix_flag: Optional[str]
    _unit_tax_salient: bool
    _advalorem_tax_salient: bool

    # once-per-session DeprecationWarning. Class-level flag so
    # the warning fires on the first ModelFormulation construction of the
    # Python session and not again. Tests that want to verify the warning
    # directly reset this flag to False before checking.
    _deprecation_warned: bool = False

    def __init__(
            self, model_downstream: Optional[str] = None, model_upstream: Optional[str] = None,
            ownership_downstream: Optional[str] = None, ownership_upstream: Optional[str] = None,
            custom_model_specification: Optional[Dict[str, Any]] = None, vertical_integration: Optional[str] = None,
            unit_tax: Optional[str] = None, advalorem_tax: Optional[str] = None, advalorem_payer: Optional[str] = None,
            cost_scaling: Optional[str] = None,
            kappa_specification_downstream: Optional[Union[str, Callable[[Any, Any], float]]] = None,
            kappa_specification_upstream: Optional[Union[str, Callable[[Any, Any], float]]] = None,
            user_supplied_markups: Optional[str] = None, mix_flag: Optional[str] = None,
            unit_tax_salient: bool = True, advalorem_tax_salient: bool = True) -> None:
        """Parse the formula into patsy terms and SymPy expressions. In the process, validate it as much as possible
        without any data.
        """

        # fire a DeprecationWarning on the first ModelFormulation
        # construction of the session. Users should migrate to the class-based
        # ConductModel API (pyRVtest.Bertrand, pyRVtest.Cournot, etc.).
        if not ModelFormulation._deprecation_warned:
            warnings.warn(
                _MODELFORMULATION_DEPRECATION_MSG,
                DeprecationWarning,
                stacklevel=2,
            )
            ModelFormulation._deprecation_warned = True

        # validate the parameters
        model_set = {'monopoly', 'cournot', 'bertrand', 'perfect_competition', 'mix_cournot_bertrand', 'other'}
        if model_downstream is None and user_supplied_markups is None:
            raise TypeError(
                "Expected either model_downstream (a conduct string) or "
                "user_supplied_markups (a column name) to be set. "
                "Received both as None. "
                "Fix: pass model_downstream='bertrand' (or similar), or "
                "user_supplied_markups='<column>'."
            )
        if model_downstream is not None and model_downstream not in model_set:
            raise TypeError(
                f"Expected model_downstream to be one of {sorted(model_set)}. "
                f"Received {model_downstream!r}. "
                f"Fix: pick one of the supported conduct strings."
            )
        if model_upstream is not None and model_upstream not in model_set:
            raise TypeError(
                f"Expected model_upstream to be one of {sorted(model_set)}. "
                f"Received {model_upstream!r}. "
                f"Fix: pick one of the supported conduct strings."
            )
        if model_upstream is not None and model_downstream in {'cournot'} and model_upstream in {'cournot'}:
            raise TypeError(
                "Expected at most one tier of a vertical model to use Cournot "
                "(the Villas-Boas passthrough derivation is not defined for "
                "Cournot-on-Cournot). "
                "Received model_downstream='cournot' and model_upstream='cournot'. "
                "Fix: use Bertrand or Monopoly for one of the two tiers."
            )
        if ownership_downstream is not None and not isinstance(ownership_downstream, str):
            raise TypeError(
                f"Expected ownership_downstream to be a column name (str) or None. "
                f"Received {type(ownership_downstream).__name__}. "
                f"Fix: pass the column name as a string, e.g. 'firm_ids'."
            )
        if ownership_upstream is not None and not isinstance(ownership_upstream, str):
            raise TypeError(
                f"Expected ownership_upstream to be a column name (str) or None. "
                f"Received {type(ownership_upstream).__name__}. "
                f"Fix: pass the column name as a string, e.g. 'manufacturer_ids'."
            )
        if model_upstream is not None and not isinstance(ownership_upstream, str):
            raise TypeError(
                "Expected ownership_upstream to be a column-name string when "
                "model_upstream is set (vertical models need upstream ownership). "
                "Received ownership_upstream as None (or non-str). "
                "Fix: supply ownership_upstream='<manufacturer-id-column>'."
            )
        if vertical_integration is not None and not isinstance(vertical_integration, str):
            raise TypeError(
                f"Expected vertical_integration to be a column name (str) or None. "
                f"Received {type(vertical_integration).__name__}. "
                f"Fix: pass the column name as a string."
            )
        if unit_tax is not None and not isinstance(unit_tax, str):
            raise TypeError(
                f"Expected unit_tax to be a column name (str) or None. "
                f"Received {type(unit_tax).__name__}. "
                f"Fix: pass the column name as a string."
            )
        if advalorem_tax is not None and not isinstance(advalorem_tax, str):
            raise TypeError(
                f"Expected advalorem_tax to be a column name (str) or None. "
                f"Received {type(advalorem_tax).__name__}. "
                f"Fix: pass the column name as a string."
            )
        if advalorem_payer is not None and advalorem_payer not in {'firm', 'consumer', 'firms', 'consumers'}:
            raise TypeError(
                f"Expected advalorem_payer to be 'firm' or 'consumer' (or None). "
                f"Received {advalorem_payer!r}. "
                f"Fix: set advalorem_payer='firm' or 'consumer' to indicate who "
                f"remits the ad-valorem tax."
            )
        if advalorem_tax is not None and advalorem_payer is None:
            raise TypeError(
                "Expected advalorem_payer to be 'firm' or 'consumer' when "
                "advalorem_tax is supplied. "
                "Received advalorem_payer=None. "
                "Fix: set advalorem_payer='firm' or 'consumer'."
            )
        if cost_scaling is not None and not isinstance(cost_scaling, str):
            raise TypeError(
                f"Expected cost_scaling to be a column name (str) or None. "
                f"Received {type(cost_scaling).__name__}. "
                f"Fix: pass the column name as a string."
            )
        if mix_flag is not None and not isinstance(mix_flag, str):
            raise TypeError(
                f"Expected mix_flag to be a column name (str) or None. "
                f"Received {type(mix_flag).__name__}. "
                f"Fix: pass the column name of the per-product Bertrand/Cournot "
                f"boolean flag."
            )
        if model_downstream == 'mix_cournot_bertrand' and mix_flag is None:
            raise TypeError(
                "Expected mix_flag to be set when model_downstream='mix_cournot_bertrand' "
                "(the Schur-complement FOC needs to know which products are Bertrand). "
                "Received mix_flag=None. "
                "Fix: pass mix_flag='<boolean-column>' (True=Bertrand, False=Cournot)."
            )
        if mix_flag is not None and model_downstream != 'mix_cournot_bertrand':
            raise TypeError(
                f"Expected mix_flag to be set only when "
                f"model_downstream='mix_cournot_bertrand'. "
                f"Received mix_flag={mix_flag!r} with model_downstream={model_downstream!r}. "
                f"Fix: drop mix_flag, or set model_downstream='mix_cournot_bertrand'."
            )
        # per-model salience flags for Problem-level taxes.
        if not isinstance(unit_tax_salient, bool):
            raise TypeError(
                f"Expected unit_tax_salient to be True or False. "
                f"Received {type(unit_tax_salient).__name__} "
                f"({unit_tax_salient!r}). "
                f"Fix: pass unit_tax_salient=True (default) or False."
            )
        if not isinstance(advalorem_tax_salient, bool):
            raise TypeError(
                f"Expected advalorem_tax_salient to be True or False. "
                f"Received {type(advalorem_tax_salient).__name__} "
                f"({advalorem_tax_salient!r}). "
                f"Fix: pass advalorem_tax_salient=True (default) or False."
            )
        if model_downstream == 'other' and custom_model_specification is None and user_supplied_markups is None:
            raise TypeError(
                "Expected custom_model_specification (or user_supplied_markups) "
                "when model_downstream='other'. "
                "Received both as None. "
                "Fix: pass custom_model_specification={'name': callable}, or "
                "switch to user_supplied_markups='<column>'."
            )

        # parse the formulas into patsy terms
        self._model_downstream = model_downstream
        self._model_upstream = model_upstream
        self._ownership_downstream = ownership_downstream
        self._ownership_upstream = ownership_upstream
        self._custom_model_specification = custom_model_specification
        self._vertical_integration = vertical_integration
        self._unit_tax = unit_tax
        self._advalorem_tax = advalorem_tax
        self._advalorem_payer = advalorem_payer
        self._kappa_specification_downstream = kappa_specification_downstream
        self._kappa_specification_upstream = kappa_specification_upstream
        self._cost_scaling = cost_scaling
        self._user_supplied_markups = user_supplied_markups
        self._mix_flag = mix_flag
        self._unit_tax_salient = unit_tax_salient
        self._advalorem_tax_salient = advalorem_tax_salient

    def __reduce__(self) -> Tuple[Type['ModelFormulation'], Tuple[Any, ...]]:
        """Handle pickling."""
        return (self.__class__, (
            self._model_downstream, self._model_upstream,
            self._ownership_downstream, self._ownership_upstream,
            self._custom_model_specification, self._vertical_integration,
            self._unit_tax, self._advalorem_tax, self._advalorem_payer,
            self._cost_scaling,
            self._kappa_specification_downstream, self._kappa_specification_upstream,
            self._user_supplied_markups, self._mix_flag,
            self._unit_tax_salient, self._advalorem_tax_salient,
        ))

    def __str__(self) -> str:
        """Format the terms as a string."""
        # NOTE: historically this list is typed ``List[str]`` but the fields are
        # ``Optional[str]``. Runtime behavior preserved; use ``Any`` element type
        # so that a downstream-only configuration (``_model_upstream is None``)
        # matches the original ``' + '.join([s, None])`` path (which raises
        # TypeError lazily, as before).
        names: List[Any] = [self._model_downstream, self._model_upstream]
        return ' + '.join(names)
