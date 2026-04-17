"""Conduct testing problem and supporting data structures."""

import abc
import contextlib
import itertools
import math
import os
import time
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from pyblp.utilities.algebra import precisely_identify_collinearity
from pyblp.utilities.basics import (
    Array, Data, Groups, RecArray, StringRepresentation, extract_matrix, format_seconds, format_table,
    get_indices, output, structure_matrices
)
from pyblp.configurations.formulation import ColumnFormulation
from scipy.linalg import inv
from scipy.stats import norm
from . import options
from .formulation import Absorb, Formulation, ModelFormulation
from .markups import build_ownership, _compute_markups
from .data import read_critical_values_tables
from .products import Products
from .results import ProblemResults, Progress


def _qr_residualize(Y: Array, X: Array) -> Array:
    """Project out X from Y via QR decomposition (equivalent to OLS residuals, without statsmodels overhead).

    Works for 1D or 2D Y. If X has no columns, Y is returned unchanged.
    """
    if X.shape[1] == 0:
        return Y
    Q, _ = np.linalg.qr(X, mode='reduced')
    return Y - Q @ (Q.T @ Y)


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
    custom_model: `dict, optional`
        A custom formula used to compute markups, optionally specified by the user.
    user_supplied_markups: `ndarray, optional`
        A vector of user-computed markups.
    user_supplied_markups_name: `str, optional`
        The name of the column containing user-supplied markups.

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
    user_supplied_markups: Array
    user_supplied_markups_name: Array

    def __new__(
            cls,
            models: Sequence[Any],
            product_data: Mapping,
    ) -> RecArray:
        """Structure model data for the pipeline.

        v0.4 step 5b': ``models`` is a sequence of
        ``pyRVtest.models.ConductModel`` or ``pyRVtest.models.Vertical``
        instances (the canonical intermediate form). ``Problem.__init__``
        converts ``ModelFormulation`` inputs to this form via
        ``from_model_formulation`` before calling here. Legacy
        ``ModelFormulation`` instances passed directly are accepted for
        backward compatibility (translated inline).
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
                raise TypeError(f"models[{i}] is None; all entries must be ConductModel, Vertical, or ModelFormulation.")
            else:
                raise TypeError(
                    f"models[{i}] must be a ConductModel, Vertical, or ModelFormulation instance; "
                    f"got {type(entry).__name__}."
                )

        M = len(normalized)
        if M < 1:
            raise ValueError("At least one model must be specified.")
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
            if config.unit_tax is not None:
                unit_tax[m] = extract_matrix(product_data, config.unit_tax)
                unit_tax_name[m] = config.unit_tax
            else:
                unit_tax[m] = np.zeros((N, 1))
            if config.advalorem_tax is not None:
                advalorem_tax[m] = extract_matrix(product_data, config.advalorem_tax)
                advalorem_tax_name[m] = config.advalorem_tax
                payer = config.advalorem_payer.replace('consumers', 'consumer').replace('firms', 'firm')
                advalorem_payer[m] = payer
            else:
                advalorem_tax[m] = np.zeros((N, 1))
            if config.cost_scaling is not None:
                cost_scaling_column[m] = config.cost_scaling
                cost_scaling[m] = extract_matrix(product_data, config.cost_scaling)
            else:
                cost_scaling[m] = np.zeros((N, 1))
            if config.user_supplied_markups is not None:
                user_supplied_markups[m] = extract_matrix(product_data, config.user_supplied_markups)
                user_supplied_markups_name[m] = config.user_supplied_markups

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
    Dict_Z_formulation: Dict[Union[str, tuple], Tuple[Optional[Array], Any]] = {}

    @abc.abstractmethod
    def __init__(self, products: RecArray, models: RecArray) -> None:
        """Store data and column formulations."""
        self.products = products
        self.models = models
        self._w_formulation = self.products.dtype.fields['w'][2]

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
    Dict_K: Dict[Union[str, tuple], Tuple[Optional[Array]]] = {}
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
            models: Optional[Sequence[Any]] = None) -> None:
        """Initialize the underlying economy with product and agent data before absorbing fixed effects."""

        output("Initializing the problem ...")
        start_time = time.time()

        # v0.4 step 5b + 5b': ``models=`` is the new class-based API;
        # ``model_formulations=`` is the legacy alias. Mutually exclusive.
        # Internally everything is ConductModel / Vertical: we translate
        # model_formulations to that form here, so the downstream pipeline
        # (Models.__new__, _compute_markups, etc.) sees a single canonical
        # type.
        if models is not None and model_formulations is not None:
            raise TypeError(
                "Specify either `models=` (new class-based API) or "
                "`model_formulations=` (legacy string-based API), not both."
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

        # Validate demand_params
        if demand_params is not None and demand_results is not None:
            raise ValueError("Specify demand_params or demand_results, not both.")
        if demand_params is not None:
            if 'alpha' not in demand_params:
                raise ValueError("demand_params must contain 'alpha' (price coefficient).")
            alpha = demand_params['alpha']
            if not isinstance(alpha, (int, float)) or alpha >= 0:
                raise ValueError("demand_params['alpha'] must be a negative number.")
            sigma = demand_params.get('sigma', [])
            if not isinstance(sigma, (list, tuple)):
                raise TypeError("demand_params['sigma'] must be a list of nesting parameters.")
            for i, s_val in enumerate(sigma):
                if s_val < 0 or s_val >= 1:
                    raise ValueError(
                        f"demand_params['sigma'][{i}] = {s_val} out of range [0, 1). "
                        f"sigma = 0 is plain logit (Berry 1994 convention)."
                    )

        if markup_data is None:
            if models is None:
                raise TypeError(
                    "Either `models=` (class-based) or `model_formulations=` "
                    "(legacy) must be provided when markup_data is None."
                )
            M = len(models)
        else:
            M = np.shape(markup_data)[0]

        if hasattr(instrument_formulation, '__len__'):
            L = len(instrument_formulation)
        else:
            L = 1

        if not isinstance(cost_formulation, Formulation):
            raise TypeError("cost_formulation must be a single Formulation instance.")

        if L == 1:
            if not isinstance(instrument_formulation, Formulation):
                raise TypeError("instrument_formulation must be a single Formulation instance.")
        elif L > 1:
            if not all(isinstance(f, Formulation) for f in instrument_formulation):
                raise TypeError("Each formulation in instrument_formulation must be a Formulation.")

        if endogenous_cost_component is not None:
            if not isinstance(endogenous_cost_component, str):
                raise TypeError("endogenous_cost_component must be a string column name.")
            _, w_formulation_check, _ = cost_formulation._build_matrix(product_data)
            endog_terms = [f for f in w_formulation_check if endogenous_cost_component in f.names]
            if not endog_terms:
                raise ValueError(
                    f"endogenous_cost_component '{endogenous_cost_component}' must appear in cost_formulation."
                )
            for f in endog_terms:
                if str(f) != endogenous_cost_component:
                    raise ValueError(
                        f"endogenous_cost_component '{endogenous_cost_component}' must enter cost_formulation "
                        f"linearly (no interactions or transformations). Found term: '{f}'."
                    )

        products = Products(
            cost_formulation=cost_formulation, instrument_formulation=instrument_formulation, product_data=product_data
        )
        if markup_data is None:
            models_recarray = Models(models=models, product_data=product_data)
            markups = [None] * M
        else:
            models_recarray = None
            markups = markup_data

        super().__init__(products, models_recarray)

        self.cost_formulation = cost_formulation
        self.instrument_formulation = instrument_formulation
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
            common_message = "To disable collinearity checks, set options.collinear_atol = options.collinear_rtol = 0."
            collinear, successful = precisely_identify_collinearity(cost_shifters)
            if not successful:
                raise ValueError(
                    f"Failed to compute the QR decomposition of w while checking for collinearity issues. "
                    f"{common_message}"
                )
            if collinear.any():
                raise ValueError(
                    f"Detected collinearity issues with w. "
                    f"{common_message}"
                )
            for instrument in range(self.L):
                cost_shifters = self.products.w
                cost_shifters = np.append(cost_shifters, self.products["Z{0}".format(instrument)], axis=1)
                collinear, successful = precisely_identify_collinearity(cost_shifters)
                if not successful:
                    raise ValueError(
                        f"Failed to compute the QR decomposition of [w,z" + str(instrument) + "] while checking for "
                        f"collinearity issues. "
                        f"{common_message}"
                    )
                if collinear.any():
                    raise ValueError(
                        f"Detected collinearity issues with [w,z" + str(instrument) + "]."
                        f"{common_message}"
                    )

        output(f"Initialized the problem after {format_seconds(time.time() - start_time)}.")
        output("")
        output(self)

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

        Returns
        -------
        `ProblemResults`
            :class:`ProblemResults` of the solved problem.

        """

        output("Solving the problem ...")
        step_start_time = time.time()

        M = self.M
        N = self.N
        L = self.L
        markups = self.markups
        critical_values_power, critical_values_size = read_critical_values_tables()

        self._validate_solve_args(demand_adjustment, clustering_adjustment)

        markups_upstream = np.zeros(M, dtype=options.dtype)
        markups_downstream = np.zeros(M, dtype=options.dtype)
        advalorem_tax_adj = [None] * M
        prices_effective = [None] * M
        markups_effective = [None] * M

        if markups[0] is None:
            output('Computing Markups ...')
            markups, markups_downstream, markups_upstream = self._perturb_and_build_markups()

        marginal_cost = self.products.prices - markups

        unit_tax = self.models["unit_tax"]
        advalorem_tax = self.models["advalorem_tax"]
        cost_scaling = self.models["cost_scaling"]
        for m in range(M):
            condition = self.models["advalorem_payer"][m] == "consumer"
            advalorem_tax_adj[m] = 1 / (1 + advalorem_tax[m]) if condition else (1 - advalorem_tax[m])
            prices_effective[m] = (advalorem_tax_adj[m] * self.products.prices / (1 + cost_scaling[m]) - unit_tax[m])
            markups_effective[m] = (advalorem_tax_adj[m] / (1 + cost_scaling[m])) * markups[m]
            marginal_cost[m] = prices_effective[m] - markups_effective[m]

        if costs_type == "log" and not demand_adjustment:
            if np.any(marginal_cost < 0):
                raise ValueError("Can't generate log costs with negative marginal cost!")
            marginal_cost = np.log(marginal_cost)

        if mc_correction is not None:
            if marginal_cost.shape != mc_correction.shape:
                raise ValueError(
                    "The dimensions of marginal cost correction don't match the dimensions of the marginal cost."
                )
            marginal_cost = marginal_cost + mc_correction

        cost_param = None

        if self.endogenous_cost_component is not None:
            output('Computing IV correction for endogenous cost component ...')
            marginal_cost_base = marginal_cost.copy()
            cost_param = [None] * L
            omega_per_instrument = [None] * L
            tau_list_per_instrument = [None] * L
            endog_hat_per_instrument = [None] * L

            for l in range(L):
                cp_l, mc_corr_l, endog_hat_l = self._compute_iv_correction(l, M, N, marginal_cost_base)
                mc_l = marginal_cost_base.copy()
                for m in range(M):
                    mc_l[m] = mc_l[m] + mc_corr_l[m]
                mo_l, omega_l, tau_l = self._prepare_orthogonal_variables(M, N, markups_effective, mc_l)
                if l == 0:
                    markups_orthogonal = mo_l  # identical across instrument sets; keep first
                    marginal_cost = mc_l        # store first instrument's corrected MC for results
                    tau_list = tau_l
                cost_param[l] = cp_l
                omega_per_instrument[l] = omega_l
                tau_list_per_instrument[l] = tau_l
                endog_hat_per_instrument[l] = endog_hat_l

        else:
            markups_orthogonal, omega, tau_list = self._prepare_orthogonal_variables(
                M, N, markups_effective, marginal_cost
            )
            omega_per_instrument = [omega] * L
            endog_hat_per_instrument = [None] * L
            tau_list_per_instrument = None

        gradient_markups = H_prime_wd = H = h_i = h = None
        gradient_gamma_per_instrument = None
        if demand_adjustment:
            # v0.4 step 4e: single code path via the unified function in
            # solve/demand_adjustment.py. Replaces the former split into
            # _compute_analytical_demand_adjustment (demand_params) and
            # _compute_demand_adjustment_gradient (demand_results). Also
            # closes the silent capability gap where the analytical path
            # returned gradient_gamma_per_instrument=None, silently
            # disabling the endogenous-cost correction for demand_params
            # users.
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
            r = self._compute_instrument_results(
                instrument, M, N, omega_per_instrument[instrument], demand_adjustment, gradient_markups,
                H_prime_wd, H, h_i, h, clustering_adjustment,
                critical_values_size, critical_values_power, endog_hat_per_instrument[instrument],
                grad_gamma_l
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
        output(f"Solved the problem after {format_seconds(time.time() - step_start_time)}.")
        output("")
        output(results)
        return results

    def _validate_solve_args(self, demand_adjustment: bool, clustering_adjustment: bool) -> None:
        """Validate arguments passed to solve."""
        if not isinstance(demand_adjustment, bool):
            raise TypeError("demand_adjustment must be a boolean (one of True or False).")
        if not isinstance(clustering_adjustment, bool):
            raise TypeError("clustering_adjustment must be a boolean (one of True or False).")
        if clustering_adjustment and np.shape(self.products.clustering_ids)[1] != 1:
            raise ValueError("product_data.clustering_ids must be specified with clustering_adjustment True.")
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
                    f"demand_adjustment=True with demand_params requires: {', '.join(missing)}. "
                    f"These are only needed for demand adjustment; omit them if demand_adjustment=False."
                )
        for m in range(self.M):
            if self._models[m].user_supplied_markups is not None:
                if demand_adjustment:
                    raise ValueError(
                        "demand_adjustment is not supported with user-supplied markups, because the "
                        "finite-difference gradient requires a demand system to perturb."
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
          - `demand_params is not None` with non-empty nonzero sigma -> `NestedLogitBackend`.
          - `demand_params is not None` with empty or all-zero sigma -> `LogitBackend`.

        Demand-adjustment state (beta, x_columns, demand_instrument_columns, W_demand)
        is forwarded when present so `SupportsDemandAdjustment` methods work. The raw
        `product_data` (not the structured `Products` recarray) is passed because the
        backends need access to arbitrary columns like `nesting_ids`, `x1`, etc.
        """
        if self.demand_results is not None:
            return self._backend_from_demand_results(self.demand_results)

        if self.demand_params is not None:
            dp = self.demand_params
            sigma_nonzero = [s for s in dp.get('sigma', []) if s > 0]
            shared_kwargs = dict(
                alpha=dp['alpha'],
                product_data=self._product_data_raw,
                beta=dp.get('beta'),
                x_columns=dp.get('x_columns'),
                demand_instrument_columns=dp.get('demand_instrument_columns'),
                W_demand=dp.get('W_demand'),
            )
            if sigma_nonzero:
                from .backends.nested_logit import NestedLogitBackend
                return NestedLogitBackend(
                    sigma=dp.get('sigma', []),
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
          try routing to ``NestedLogitBackend(sigma=[rho])``. Analytical
          ``d(D)/d(rho)`` is exact; pyblp's finite-diff has genuine
          O(eps^2) truncation error because D is nonlinear in rho.
          Precision gain is material.

        * **Per-nest rho** (``K2==0`` and ``r.rho.size>1``): stay on
          ``PyBLPBackend``. AFSSZ L=1 formulation has one sigma; pyblp's
          Cardell-Nevo formulation has one rho per nest. The derivatives
          ``d(D)/d(rho_h)`` don't match ``d(D)/d(sigma_1)``.

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
                sigma=[float(rho_arr[0])],
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
            f"Unsupported product_data type for intercept augmentation: {type(raw)}"
        )

    def _perturb_and_build_markups(self):
        """Call _compute_markups with this Problem's constructed backend.

        v0.4 step 4g: after step 4f deleted the inline demand-adjustment
        methods that mutated ``self.demand_results._sigma`` / ``._pi`` /
        ``._beta`` / ``._rho`` directly, nothing mutates demand state behind
        the backend's cache. Routing through ``self._demand_backend`` is
        therefore safe: the cache is invalidated correctly by
        ``backend.perturbed(...)`` everywhere it's used. Legacy
        ``demand_jacobian`` / ``demand_alpha`` / ``demand_sigma`` kwargs on
        ``_compute_markups`` are gone; ``compute_jacobian`` / ``compute_hessian``
        on the backend subsume them.
        """
        return _compute_markups(
            self.products, self.demand_results, self.models["models_downstream"],
            self.models["ownership_downstream"], self.models["models_upstream"],
            self.models["ownership_upstream"], self.models["vertical_integration"],
            self.models["custom_model_specification"], self.models["user_supplied_markups"],
            self.models["mix_flag"], demand_backend=self._demand_backend,
        )

    def _prepare_orthogonal_variables(self, M: int, N: int, markups_effective: list, marginal_cost: Array):
        """Absorb fixed effects and residualize markups and marginal costs w.r.t. cost shifters."""
        markups_orthogonal = np.zeros((M, N), dtype=options.dtype)
        marginal_cost_orthogonal = np.zeros((M, N), dtype=options.dtype)

        # When an endogenous cost component is present, its coefficient was estimated via IV and the correction
        # has already been applied to marginal_cost. The OLS projection therefore uses only the exogenous columns
        # of w so that tau_list corresponds to the exogenous cost-shifter coefficients.
        if self.endogenous_cost_component is not None:
            endog_col_idx = next(
                i for i, f in enumerate(self._w_formulation)
                if str(f) == self.endogenous_cost_component
            )
            exog_col_indices = [i for i in range(self.products.w.shape[1]) if i != endog_col_idx]
            w_for_ols = self.products.w[:, exog_col_indices]
        else:
            w_for_ols = self.products.w

        tau_list = np.zeros((M, w_for_ols.shape[1]), dtype=options.dtype)
        omega = np.zeros((M, N), dtype=options.dtype)

        if self._absorb_cost_ids is not None:
            output("Absorbing cost-side fixed effects ...")
            w_for_ols, _ = self._absorb_cost_ids(w_for_ols)
            for m in range(M):
                value, _ = self._absorb_cost_ids(markups_effective[m])
                markups_orthogonal[m] = np.squeeze(value)
                value, _ = self._absorb_cost_ids(marginal_cost[m])
                marginal_cost_orthogonal[m] = np.squeeze(value)
        else:
            for m in range(M):
                markups_orthogonal[m] = np.squeeze(markups_effective[m])
                marginal_cost_orthogonal[m] = np.squeeze(marginal_cost[m])

        if w_for_ols.any():
            Q_w, R_w = np.linalg.qr(w_for_ols, mode='reduced')
            for m in range(M):
                markups_orthogonal[m] = markups_orthogonal[m] - Q_w @ (Q_w.T @ markups_orthogonal[m])
                mc_vec = marginal_cost_orthogonal[m]
                tau_list[m] = np.linalg.solve(R_w, Q_w.T @ mc_vec)
                omega[m] = mc_vec - Q_w @ (Q_w.T @ mc_vec)
        else:
            omega = marginal_cost_orthogonal

        return markups_orthogonal, omega, tau_list

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
        """Compute all test statistics for a single instrument set."""
        instruments = self.products["Z{0}".format(instrument)]
        K = np.shape(instruments)[1]
        K_effective = K - 1 if endog_hat is not None else K

        # Use only exogenous cost-shifter columns when an endogenous component has been IV-corrected
        if self.endogenous_cost_component is not None:
            endog_col_idx = next(
                i for i, f in enumerate(self._w_formulation)
                if str(f) == self.endogenous_cost_component
            )
            exog_col_indices = [i for i in range(self.products.w.shape[1]) if i != endog_col_idx]
            w_for_ols = self.products.w[:, exog_col_indices]
        else:
            w_for_ols = self.products.w

        if self._absorb_cost_ids is not None:
            Z_orthogonal, _ = self._absorb_cost_ids(instruments)
            w_absorbed, _ = self._absorb_cost_ids(w_for_ols)
            endog_hat_absorbed = endog_hat
            if endog_hat is not None:
                endog_hat_absorbed, _ = self._absorb_cost_ids(endog_hat)
        else:
            Z_orthogonal = instruments
            w_absorbed = w_for_ols
            endog_hat_absorbed = endog_hat

        # Residualize instruments on exogenous cost-shifters and (when applicable) first-stage
        # fitted values of the endogenous cost component jointly, so that Z_orthogonal is orthogonal
        # to both simultaneously. The rank reduction from adding endog_hat is handled by pinv.
        # All inputs must be in the same basis (absorbed or raw) for valid FWL.
        if endog_hat_absorbed is not None:
            controls = np.hstack([w_absorbed, endog_hat_absorbed]) if w_absorbed.shape[1] > 0 else endog_hat_absorbed
            Z_orthogonal = np.reshape(_qr_residualize(Z_orthogonal, controls), [N, K])
        elif w_absorbed.any():
            Z_orthogonal = np.reshape(_qr_residualize(Z_orthogonal, w_absorbed), [N, K])

        W_inverse = np.reshape(1 / N * (Z_orthogonal.T @ Z_orthogonal), [K, K])
        if options.pseudo_inverses:
            weight_matrix = np.linalg.pinv(W_inverse)
        else:
            try:
                weight_matrix = np.linalg.inv(W_inverse)
            except np.linalg.LinAlgError:
                weight_matrix = np.linalg.pinv(W_inverse)

        # GMM moments and fit for each model
        g = np.zeros((M, K), dtype=options.dtype)
        Q = np.zeros(M, dtype=options.dtype)
        for m in range(M):
            g[m] = 1 / N * (Z_orthogonal.T @ omega[m])
            Q[m] = g[m].T @ weight_matrix @ g[m]

        # RV numerator
        test_statistic_numerator = np.zeros((M, M))
        for m in range(M):
            for i in range(m):
                test_statistic_numerator[i, m] = math.sqrt(N) * (Q[i] - Q[m])

        # psi for each model and RV denominator
        # Use eigendecomposition with non-negative clipping to avoid complex values that arise when
        # floating-point errors push a zero eigenvalue (from a rank-deficient Z_orthogonal) slightly
        # negative, which would cause fractional_matrix_power to return complex results.
        _eigvals, _eigvecs = np.linalg.eigh((weight_matrix + weight_matrix.T) / 2)
        _eigvals = np.maximum(_eigvals, 0)
        W_12 = (_eigvecs * (_eigvals ** 0.50)) @ _eigvecs.T
        W_34 = (_eigvecs * (_eigvals ** 0.75)) @ _eigvecs.T
        psi = np.zeros((M, N, K), dtype=options.dtype)
        if demand_adjustment:
            adjustment_value = np.zeros((M, K, H_prime_wd.shape[1]), dtype=options.dtype)

        # Precompute first-stage correction ingredients when endogenous cost component is present.
        # Per Appendix B of Duarte-Magnolfi-Quint-Solvsten-Sullivan (2025), the influence function
        # psi includes a correction for estimation of the linear predictor q_tilde.
        endog_correction_data = None
        if endog_hat is not None:
            endog_col_idx_local = next(
                i for i, f in enumerate(self._w_formulation)
                if str(f) == self.endogenous_cost_component
            )
            endog_col = self.products.w[:, [endog_col_idx_local]]  # (N, 1) raw endogenous variable
            q_e = endog_col - endog_hat                            # (N, 1) first-stage residual

            # z^r = z residualized on w only (not on endog_hat)
            z_r = _qr_residualize(instruments, w_for_ols) if w_for_ols.shape[1] > 0 else instruments.copy()
            if self._absorb_cost_ids is not None:
                z_r, _ = self._absorb_cost_ids(z_r)
            z_r = z_r.reshape(N, K)

            # Z_prec = (1/n sum z^r z^{r'})^{-1}
            Z_cov = (1 / N) * z_r.T @ z_r                         # (K, K)
            Z_prec = np.linalg.pinv(Z_cov)                        # (K, K)

            # lambda_q: coefficient on endog_hat in projection z^e = lambda_q * q_tilde + Lambda_w * w
            # This is the coefficient from projecting z on [endog_hat, w], taking the endog_hat part
            proj_X = np.hstack([endog_hat, w_for_ols]) if w_for_ols.shape[1] > 0 else endog_hat
            Q_proj, R_proj = np.linalg.qr(proj_X, mode='reduced')
            lambda_coefs = np.linalg.solve(R_proj, Q_proj.T @ instruments)  # (d_proj, K)
            lambda_q = lambda_coefs[0, :]  # (K,) — coefficient on endog_hat (first column)

            # Precompute the (K, K) matrix M_correction = W^{3/4} W^+ Z_prec, used in correction
            W_plus = weight_matrix  # this is already the pseudo-inverse of W_inverse
            M_corr = W_34 @ W_plus @ Z_prec  # (K, K)

            endog_correction_data = (z_r, q_e, lambda_q, M_corr, Z_prec, W_plus)

        for m in range(M):
            psi_bar = W_12 @ g[m] - .5 * W_34 @ W_inverse @ W_34 @ g[m]
            W_34_Zg = (Z_orthogonal @ W_34 @ g[m])[:, np.newaxis]
            mc_col = omega[m][:, np.newaxis]
            psi_i = (mc_col * Z_orthogonal) @ W_12 - 0.5 * W_34_Zg * (Z_orthogonal @ W_34.T)
            psi[m] = psi_i - np.transpose(psi_bar)

            # First-stage correction: Appendix B of Duarte-Magnolfi-Quint-Solvsten-Sullivan (2025).
            # Per observation i, the correction to psi[m][i,:] is:
            #   (1/2) W^{3/4} (W^+ Z_prec u_i lambda'_q + lambda_q u'_i Z_prec W^+) W^{3/4} g_m
            # where u_i = z^r_i * q^e_i, M_corr = W^{3/4} W^+ Z_prec.
            # Term 1 contracts to: M_corr @ u_i * (lambda_q . W^{3/4} g_m)
            # Term 2 contracts to: W^{3/4} lambda_q * (u_i . Z_prec W^+ W^{3/4} g_m)
            if endog_correction_data is not None:
                z_r, q_e, lambda_q, M_corr, Z_prec, W_plus = endog_correction_data
                W34_gm = W_34 @ g[m]                                       # (K,)
                v = Z_prec @ W_plus @ W34_gm                               # (K,) right-side contraction
                u = q_e * z_r                                              # (N, K)
                term1 = (u @ M_corr.T) * (lambda_q @ W34_gm)              # (N, K)
                term2 = (u @ v)[:, np.newaxis] * (W_34 @ lambda_q)[np.newaxis, :]  # (N, K)
                psi[m] = psi[m] + 0.5 * (term1 + term2)

            if demand_adjustment:
                G_k = -1 / N * np.transpose(Z_orthogonal) @ gradient_markups[m]
                # When endogenous_cost_component is set, account for d(gamma_m)/d(theta) in G_k.
                # The full gradient of omega w.r.t. theta includes a term from gamma changing,
                # which contributes -(1/N) Z' @ (d_gamma/d_theta * endog_resid) to G_k.
                if gradient_gamma is not None and endog_correction_data is not None:
                    endog_col_resid = endog_correction_data[1] + endog_correction_data[0]  # q^e + z^r... no
                    # Actually: the endogenous column residualized on w is needed. Reconstruct it.
                    endog_col_idx_local2 = next(
                        i for i, f in enumerate(self._w_formulation)
                        if str(f) == self.endogenous_cost_component
                    )
                    endog_col_raw = self.products.w[:, [endog_col_idx_local2]]
                    if self._absorb_cost_ids is not None:
                        endog_col_for_grad, _ = self._absorb_cost_ids(endog_col_raw)
                    else:
                        endog_col_for_grad = endog_col_raw
                    if w_absorbed.shape[1] > 0:
                        endog_col_for_grad = _qr_residualize(endog_col_for_grad, w_absorbed)
                    # gradient_gamma[m] is (n_theta,) — d gamma_m / d theta_k for each k
                    # G_k[:, k] -= (1/N) * Z' @ (d_gamma_m/d_theta_k * endog_resid)
                    G_k = G_k - 1 / N * np.transpose(Z_orthogonal) @ (endog_col_for_grad @ gradient_gamma[[m], :])
                adjustment_value[m] = W_12 @ G_k @ inv(H_prime_wd @ H) @ H_prime_wd
                psi[m] = psi[m] - (h_i - np.transpose(h)) @ np.transpose(adjustment_value[m])

        test_statistic_denominator = np.zeros((M, M))
        covariance_mc = np.zeros((M, M))
        psi_gram = self._compute_block_gram(N, clustering_adjustment, psi)
        for m in range(M):
            for i in range(m):
                vc_ii = self._extract_block(psi_gram, i, i, K)
                vc_mm = self._extract_block(psi_gram, m, m, K)
                vc_im = self._extract_block(psi_gram, i, m, K)
                weighted_variance = np.array([W_12 @ vc_ii @ W_12, W_12 @ vc_mm @ W_12, W_12 @ vc_im @ W_12])
                operations = np.array([1, 1, -2])
                moments = np.array([
                    g[i].T @ weighted_variance[0] @ g[i],
                    g[m].T @ weighted_variance[1] @ g[m],
                    g[i].T @ weighted_variance[2] @ g[m]
                ]).flatten()
                covariance_mc[i, m] = moments[2]
                covariance_mc[m, i] = moments[2]
                covariance_mc[m, m] = moments[1]
                covariance_mc[i, i] = moments[0]
                test_statistic_denominator[i, m] = math.sqrt(4 * (operations.T @ moments))

        # RV test statistic (upper triangle only; lower triangle and diagonal are NaN)
        rv_test_statistic = np.full((M, M), np.nan)
        for m in range(M):
            for i in range(m):
                rv_test_statistic[i, m] = test_statistic_numerator[i, m] / test_statistic_denominator[i, m]

        # F statistics — residualize omega on Z_orthogonal; precompute QR once for all models
        phi = np.zeros([M, N, K])
        Q_Z, _ = np.linalg.qr(Z_orthogonal, mode='reduced')
        for m in range(M):
            e = np.reshape(omega[m] - Q_Z @ (Q_Z.T @ omega[m]), [N, 1])
            phi[m] = (e * Z_orthogonal) @ weight_matrix
            if demand_adjustment:
                phi[m] = phi[m] - (h_i - np.transpose(h)) @ np.transpose(W_12 @ adjustment_value[m])

        unscaled_F = np.zeros((M, M))
        F = np.full((M, M), np.nan)
        rho = np.zeros((M, M))
        F_cv_size = np.empty((M, M), dtype=object)
        F_cv_power = np.empty((M, M), dtype=object)
        symbols_size = np.empty((M, M), dtype=object)
        symbols_power = np.empty((M, M), dtype=object)

        phi_gram = self._compute_block_gram(N, clustering_adjustment, phi)
        for (m, i) in itertools.product(range(M), range(M)):
            if i < m:
                variance = np.array([
                    self._extract_block(phi_gram, i, i, K),
                    self._extract_block(phi_gram, m, m, K),
                    self._extract_block(phi_gram, i, m, K)
                ])
                sigma = 1 / K_effective * np.array([
                    np.trace(variance[0] @ W_inverse),
                    np.trace(variance[1] @ W_inverse),
                    np.trace(variance[2] @ W_inverse)
                ])
                numerator_sqrt = sigma[0] - sigma[1]
                denominator_sqrt = np.sqrt((sigma[0] + sigma[1]) ** 2 - 4 * sigma[2] ** 2)
                rho[i, m] = numerator_sqrt / denominator_sqrt
                rho_squared = rho[i, m] ** 2

                operations = np.array([sigma[1], sigma[0], -2 * sigma[2]])
                moments = np.array([
                    g[i].T @ weight_matrix @ g[i],
                    g[m].T @ weight_matrix @ g[m],
                    g[i].T @ weight_matrix @ g[m]
                ]).flatten()
                F_numerator = operations @ moments
                F_denominator = sigma[0] * sigma[1] - sigma[2] ** 2
                unscaled_F[i, m] = N / (2 * K_effective) * F_numerator / F_denominator
                F[i, m] = (1 - rho_squared) * unscaled_F[i, m]

                rho_lookup = min(np.round(np.abs(rho[i, m]), 2), 0.99)
                K_lookup = K_effective if K_effective <= 30 else 30
                ind = np.where(
                    (critical_values_size['K'] == K_lookup) & (critical_values_size['rho'] == rho_lookup)
                )[0][0]
                F_cv_size[i, m] = np.array([
                    critical_values_size['r_125'][ind],
                    critical_values_size['r_10'][ind],
                    critical_values_size['r_075'][ind]
                ], dtype=object)
                F_cv_power[i, m] = np.array([
                    critical_values_power['r_50'][ind],
                    critical_values_power['r_75'][ind],
                    critical_values_power['r_95'][ind]
                ], dtype=object)

                symbols_size[i, m] = (
                    " " if F[i, m] < F_cv_size[i, m][0] else
                    "*" if F[i, m] < F_cv_size[i, m][1] else
                    "**" if F[i, m] < F_cv_size[i, m][2] else "***"
                )
                symbols_power[i, m] = (
                    " " if F[i, m] < F_cv_power[i, m][0] else
                    "^" if F[i, m] < F_cv_power[i, m][1] else
                    "^^" if F[i, m] < F_cv_power[i, m][2] else "^^^"
                )
            else:
                symbols_size[i, m] = ""
                symbols_power[i, m] = ""

        # model confidence set
        all_model_combinations = list(itertools.combinations(range(M), 2))
        n_combinations = len(all_model_combinations)
        model_confidence_set_variance = np.zeros([n_combinations, 1])
        sigma_mcs = np.zeros([n_combinations, n_combinations])
        for index_i, model_i in enumerate(all_model_combinations):
            model_confidence_set_variance[index_i] = test_statistic_denominator[model_i[0], model_i[1]] / 2
            for index_j, model_j in enumerate(all_model_combinations):
                term1 = covariance_mc[model_i[0], model_j[0]] - covariance_mc[model_i[1], model_j[0]]
                term2 = covariance_mc[model_i[0], model_j[1]] - covariance_mc[model_i[1], model_j[1]]
                sigma_mcs[index_j, index_i] = term1 - term2
        sigma_mcs = sigma_mcs / (model_confidence_set_variance @ model_confidence_set_variance.T)

        mcs_pvalues = self._compute_mcs(
            rv_test_statistic, sigma_mcs, model_confidence_set_variance, M, all_model_combinations
        )

        return {
            'g': g, 'Q': Q,
            'RV_numerator': test_statistic_numerator,
            'RV_denominator': test_statistic_denominator,
            'rv_test_statistic': rv_test_statistic,
            'F': F, 'unscaled_F': unscaled_F,
            'mcs_pvalues': mcs_pvalues,
            'rho': rho,
            'F_cv_size': F_cv_size,
            'F_cv_power': F_cv_power,
            'symbols_size': symbols_size,
            'symbols_power': symbols_power,
        }

    def _compute_mcs(
            self, rv_test_statistic: Array, sigma_mcs: Array,
            model_confidence_set_variance: Array, M: int, all_model_combinations: list
    ) -> Array:
        """Compute model confidence set p-values by iteratively eliminating the worst-fitting model."""
        rng = np.random.default_rng(options.random_seed)
        model_confidence_set = np.array(range(M))
        mcs_pvalues = np.ones([M, 1])
        converged = False
        while not converged:
            if np.shape(model_confidence_set)[0] == 2:
                max_test_statistic = rv_test_statistic[model_confidence_set[0], model_confidence_set[1]]
                if np.sign(max_test_statistic) >= 0:
                    worst_fit = model_confidence_set[0]
                    max_test_statistic = -max_test_statistic
                else:
                    worst_fit = model_confidence_set[1]
                mcs_pvalues[worst_fit] = 2 * norm.cdf(max_test_statistic)
                converged = True
            else:
                current_combinations = list(itertools.combinations(model_confidence_set, 2))
                sigma_index = np.array(
                    [all_model_combinations.index(c) for c in current_combinations], dtype=int
                )
                model_1 = [c[0] for c in current_combinations]
                model_2 = [c[1] for c in current_combinations]
                test_stats = rv_test_statistic[model_1, model_2]
                index = np.argmax(abs(test_stats))
                max_test_statistic = test_stats[index]

                if np.sign(max_test_statistic) >= 0:
                    worst_fit = model_1[index]
                else:
                    worst_fit = model_2[index]
                    max_test_statistic = -max_test_statistic

                cov = sigma_mcs[sigma_index[:, None], sigma_index]
                simulated = rng.multivariate_normal(np.zeros(len(current_combinations)), cov, options.ndraws)
                mcs_pvalues[worst_fit] = np.mean(np.amax(abs(simulated), 1) > max_test_statistic)
                model_confidence_set = np.delete(
                    model_confidence_set, np.where(model_confidence_set == worst_fit)
                )
        return mcs_pvalues

    def _compute_iv_correction(self, instrument: int, M: int, N: int, marginal_cost: list):
        """Run per-model 2SLS to estimate the coefficient on the endogenous cost component.

        For each model m the dependent variable is the implied marginal cost (price minus markup). The endogenous
        cost component (e.g. shares) is instrumented using the specified instrument set and the exogenous
        cost-shifters. The estimated coefficient gamma_m is used to form a marginal-cost correction:
        ``mc_correction[m] = -gamma_m * endogenous_variable``.

        Parameters
        ----------
        instrument : int
            Index of the instrument set (Z_l) to use for the first stage.

        Returns
        -------
        cost_param : list of ndarray
            Per-model 2SLS parameter vectors. Each vector contains coefficients for the exogenous cost-shifters
            followed by the coefficient on the endogenous component (gamma).
        mc_correction : list of ndarray
            Per-model (N, 1) correction arrays to be added to marginal cost.
        """
        # Identify the endogenous column in w
        endog_col_idx = next(
            i for i, f in enumerate(self._w_formulation)
            if str(f) == self.endogenous_cost_component
        )
        exog_col_indices = [i for i in range(self.products.w.shape[1]) if i != endog_col_idx]
        endog_col_raw = self.products.w[:, [endog_col_idx]]  # (N, 1) — raw, used for mc_correction
        exog_w = self.products.w[:, exog_col_indices]     # (N, K_w - 1)

        # Use only the instrument set for this test (keeps each instrument set's correction independent)
        Z_inst = self.products["Z{0}".format(instrument)]  # (N, K_l)

        # Absorb cost-side fixed effects before running 2SLS, so that gamma_m is estimated
        # within-group rather than in levels (consistent with _prepare_orthogonal_variables)
        endog_col = endog_col_raw
        if self._absorb_cost_ids is not None:
            exog_w, _ = self._absorb_cost_ids(exog_w)
            endog_col, _ = self._absorb_cost_ids(endog_col)
            Z_inst, _ = self._absorb_cost_ids(Z_inst)

        # First stage: project endogenous variable on [exog_w, Z_inst]
        first_stage_X = np.hstack([exog_w, Z_inst])
        Q_fs, _ = np.linalg.qr(first_stage_X, mode='reduced')
        endog_hat = (Q_fs @ (Q_fs.T @ endog_col)).reshape(-1, 1)

        # Second stage design matrix: replace endogenous column with its first-stage fitted values
        X_2sls = np.hstack([exog_w, endog_hat])  # (N, K_w)

        Q_2sls, R_2sls = np.linalg.qr(X_2sls, mode='reduced')
        cost_param = [None] * M
        mc_correction = [None] * M
        for m in range(M):
            y_m = marginal_cost[m]  # (N, 1)
            params = np.linalg.solve(R_2sls, Q_2sls.T @ y_m)
            gamma_m = params[-1]                    # coefficient on the endogenous component
            cost_param[m] = params                  # [tau_exog..., gamma]
            mc_correction[m] = -gamma_m * endog_col_raw  # (N, 1) — uses raw (un-absorbed) endog

        return cost_param, mc_correction, endog_hat

    def _compute_block_gram(self, N: int, clustering_adjustment: bool, var: Array) -> Array:
        """Compute the (M*K, M*K) block Gram matrix for all model pairs at once.

        Returns gram such that the (K, K) variance block for model pair (i, m) is:
            gram[i*K:(i+1)*K, m*K:(m+1)*K]

        With clustering, uses the Cameron-Gelbach-Miller cluster-sum formula:
            V_clustered = (1/N) * sum_c s_c s_c'
        where s_c is the sum of var within cluster c. This is mathematically equivalent
        to the per-pair roll-based computation but runs in O(M*N) numpy operations
        instead of O(C * S * M^2) Python loop iterations.
        """
        M, _, K = var.shape
        if clustering_adjustment:
            cluster_ids_flat = self.products.clustering_ids.flatten()
            unique_clusters = np.unique(cluster_ids_flat)
            C = len(unique_clusters)
            cluster_map = {c: idx for idx, c in enumerate(unique_clusters)}
            cluster_idx = np.array([cluster_map[c] for c in cluster_ids_flat])
            cluster_sums = np.zeros((M, C, K), dtype=var.dtype)
            for m in range(M):
                np.add.at(cluster_sums[m], cluster_idx, var[m])
            cs_flat = cluster_sums.transpose(1, 0, 2).reshape(C, M * K)
            return (1 / N) * cs_flat.T @ cs_flat
        else:
            var_flat = var.transpose(1, 0, 2).reshape(N, M * K)
            return (1 / N) * var_flat.T @ var_flat

    @staticmethod
    def _extract_block(gram: Array, i: int, m: int, K: int) -> Array:
        """Extract a (K, K) block from the block Gram matrix."""
        return gram[i * K:(i + 1) * K, m * K:(m + 1) * K]


