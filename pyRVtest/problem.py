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
            cls, model_formulations: Sequence[Optional[ModelFormulation]], product_data: Mapping) -> RecArray:
        """Structure model data. Data structures may be empty."""

        # validate the model formulations
        if not all(isinstance(f, ModelFormulation) or f is None for f in model_formulations):
            raise TypeError("Each formulation in model_formulations must be a ModelFormulation instance or None.")
        M = len(model_formulations)
        if M < 1:
            raise ValueError("At least one model formulation must be specified.")
        N = product_data.shape[0]

        # initialize model components
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

        # extract data for each model
        for m in range(M):
            model = model_formulations[m]._build_matrix(product_data)
            models_downstream[m] = model['model_downstream']
            if model['model_upstream'] is not None:
                models_upstream[m] = model['model_upstream']

            # define ownership matrices for downstream model
            model['firm_ids'] = model['ownership_downstream']
            if model['model_downstream'] == 'monopoly':
                ownership_matrices_downstream[m] = build_ownership(
                    product_data, model['ownership_downstream'], 'monopoly'
                )
                firm_ids_downstream[m] = 'monopoly'
            elif model['ownership_downstream'] is not None:
                ownership_matrices_downstream[m] = build_ownership(
                    product_data, model['ownership_downstream'], model['kappa_specification_downstream']
                )
                firm_ids_downstream[m] = model['ownership_downstream']

            # define ownership matrices for upstream model
            model['firm_ids'] = model['ownership_upstream']
            if model['model_upstream'] == 'monopoly':
                ownership_matrices_upstream[m] = build_ownership(product_data, model['ownership_upstream'], 'monopoly')
                firm_ids_upstream[m] = 'monopoly'
            elif model['ownership_upstream'] is not None:
                ownership_matrices_upstream[m] = build_ownership(
                    product_data, model['ownership_upstream'], model['kappa_specification_upstream']
                )
                firm_ids_upstream[m] = model['ownership_upstream']

            # define vertical integration related variables
            if model["vertical_integration"] is not None:
                vertical_integration[m] = extract_matrix(product_data, model["vertical_integration"])
                vertical_integration_index[m] = model["vertical_integration"]

            # define unit tax
            if model['unit_tax'] is not None:
                unit_tax[m] = extract_matrix(product_data, model['unit_tax'])
                unit_tax_name[m] = model['unit_tax']
            elif model['unit_tax'] is None:
                unit_tax[m] = np.zeros((N, 1))

            # define ad valorem tax
            if model['advalorem_tax'] is not None:
                advalorem_tax[m] = extract_matrix(product_data, model['advalorem_tax'])
                advalorem_tax_name[m] = model['advalorem_tax']
                advalorem_payer[m] = model['advalorem_payer']
                advalorem_payer[m] = advalorem_payer[m].replace('consumers', 'consumer').replace('firms', 'firm')
            elif model['advalorem_tax'] is None:
                advalorem_tax[m] = np.zeros((N, 1))

            # define cost scaling
            if model['cost_scaling'] is not None:
                cost_scaling_column[m] = model['cost_scaling']
                cost_scaling[m] = extract_matrix(product_data, model['cost_scaling'])
            elif model['cost_scaling'] is None:
                cost_scaling[m] = np.zeros((N, 1))

            # define custom markup model or user supplied markups
            custom_model[m] = model['custom_model_specification']
            if model["user_supplied_markups"] is not None:
                user_supplied_markups[m] = extract_matrix(product_data, model["user_supplied_markups"])
                user_supplied_markups_name[m] = model["user_supplied_markups"]

            # define mix_flag for mix_cournot_bertrand model
            if model.get("mix_flag") is not None:
                mix_flag[m] = extract_matrix(product_data, model["mix_flag"]).flatten().astype(bool)

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
            demand_params: Optional[dict] = None) -> None:
        """Initialize the underlying economy with product and agent data before absorbing fixed effects."""

        output("Initializing the problem ...")
        start_time = time.time()

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
            M = len(model_formulations)
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
            models = Models(model_formulations=model_formulations, product_data=product_data)
            markups = [None] * M
        else:
            models = None
            markups = markup_data

        super().__init__(products, models)

        self.cost_formulation = cost_formulation
        self.instrument_formulation = instrument_formulation
        self.model_formulations = model_formulations
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
        self.M = len(self.model_formulations) if self.markups[0] is None else np.shape(self.markups)[0]
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
            if self.model_formulations[m]._user_supplied_markups is not None:
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
          - `demand_results is not None` -> `PyBLPBackend(demand_results)`.
          - `demand_params is not None` with non-empty nonzero sigma -> `NestedLogitBackend`.
          - `demand_params is not None` with empty or all-zero sigma -> `LogitBackend`.

        Demand-adjustment state (beta, x_columns, demand_instrument_columns, W_demand)
        is forwarded when present so `SupportsDemandAdjustment` methods work. The raw
        `product_data` (not the structured `Products` recarray) is passed because the
        backends need access to arbitrary columns like `nesting_ids`, `x1`, etc.
        """
        if self.demand_results is not None:
            from .backends.pyblp import PyBLPBackend
            return PyBLPBackend(self.demand_results)

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

    def _perturb_and_build_markups(self):
        """Call _compute_markups with current demand_results and model specifications.

        v0.4 step 4e note: kept unchanged (NOT routed through `self._demand_backend`)
        so that inline ``_compute_demand_adjustment_gradient`` — which mutates
        ``self.demand_results._sigma`` / ``._pi`` / ``._beta`` / ``._rho`` directly
        and then calls this method — sees the perturbed state. If this method
        were to use the cached backend Jacobian, the inline finite-diff loop
        would return the cached (pre-mutation) value. The unified
        ``compute_demand_adjustment`` function has its own backend-aware
        ``_perturb_and_rebuild_markups`` that uses ``backend.perturbed(...)``
        context manager, which invalidates the cache safely.
        """
        demand_jacobian = None
        if self.demand_params is not None:
            from .demand_jacobian import compute_analytical_jacobian
            demand_jacobian = compute_analytical_jacobian(
                self.demand_params['alpha'],
                self.demand_params.get('sigma', []),
                self.products,
                self.demand_params.get('nesting_ids_columns', None)
            )
        dp = self.demand_params or {}
        return _compute_markups(
            self.products, self.demand_results, self.models["models_downstream"],
            self.models["ownership_downstream"], self.models["models_upstream"],
            self.models["ownership_upstream"], self.models["vertical_integration"],
            self.models["custom_model_specification"], self.models["user_supplied_markups"],
            self.models["mix_flag"], demand_jacobian=demand_jacobian,
            demand_alpha=dp.get('alpha'), demand_sigma=dp.get('sigma'),
        )

    def _compute_analytical_demand_adjustment(self, M: int, N: int, markups: list,
                                                  advalorem_tax_adj: list, cost_scaling: list):
        """Compute demand adjustment quantities analytically for logit/nested logit.

        Replaces _compute_demand_adjustment_gradient when demand_params is set. No finite
        differences, no PyBLP. Uses closed-form derivatives of the Berry (1994) demand system.

        Returns gradient_markups, H_prime_wd, H, h_i, h (same interface as the PyBLP path).
        """
        from .demand_jacobian import (compute_analytical_jacobian, _logit_jacobian,
                                      _nested_logit_jacobian, _nested_logit_jacobian_derivative)

        dp = self.demand_params
        alpha = dp['alpha']
        sigma = [s for s in dp.get('sigma', []) if s > 0]
        L = len(sigma)
        beta = np.asarray(dp['beta']).flatten()
        x_cols = dp['x_columns']
        z_d_cols = dp['demand_instrument_columns']

        # Build demand-side matrices from the original product_data (not the structured Products recarray,
        # which only stores cost/instrument columns)
        raw = self._product_data_raw
        X_D = np.column_stack([np.asarray(raw[col]).flatten() for col in x_cols])  # (N, K_x)
        Z_D = np.column_stack([np.asarray(raw[col]).flatten() for col in z_d_cols])  # (N, K_z)
        prices = np.asarray(self.products.prices).flatten()  # (N,)
        shares = np.asarray(self.products.shares).flatten()  # (N,)
        market_ids = np.asarray(self.products.market_ids).flatten()
        markets = np.unique(market_ids)

        # Compute outside good share and within-nest shares
        s0 = np.zeros(N)
        for t in markets:
            idx = market_ids == t
            s0[idx] = 1.0 - shares[idx].sum()

        # Within-nest conditional shares for each nesting level
        nesting_arrays = []
        log_within_nest_shares = []
        if L > 0:
            nesting_cols = dp.get('nesting_ids_columns', None)
            if nesting_cols is None:
                nesting_cols = ['nesting_ids'] if L == 1 else [f'nesting_ids_{l+1}' for l in range(L)]
            for l, col in enumerate(nesting_cols):
                nest_ids = np.asarray(raw[col]).flatten()
                nesting_arrays.append(nest_ids)
                # Compute within-nest shares per market
                within_share = np.zeros(N)
                for t in markets:
                    idx = np.where(market_ids == t)[0]
                    nest_ids_t = nest_ids[idx]
                    shares_t = shares[idx]
                    for g in np.unique(nest_ids_t):
                        mask = nest_ids_t == g
                        nest_share = shares_t[mask].sum()
                        within_share[idx[mask]] = shares_t[mask] / nest_share
                log_within_nest_shares.append(np.log(within_share))

        # Compute xi: demand residual
        # Berry (1994): log(s_j) - log(s_0) = x'beta + alpha*p + sum_l sigma_l * log(s_{j|g_l}) + xi
        log_share_ratio = np.log(shares) - np.log(s0)
        xi = log_share_ratio - X_D @ beta - alpha * prices
        for l in range(L):
            xi = xi - sigma[l] * log_within_nest_shares[l]

        # Demand moment condition: E[Z_D * xi] = 0
        # h = (1/N) Z_D' xi
        h = (1 / N) * Z_D.T @ xi  # (K_z,)
        h_i = Z_D * xi[:, np.newaxis]  # (N, K_z)

        # Demand weight matrix: the 2SLS (homoskedastic) weight W_D = (Z_D'Z_D/N)^{-1}.
        # Logit and nested logit demand are linear-in-parameters 2SLS (not iterative
        # GMM), so the weight matrix used in estimation is the 2SLS weight. Users can
        # override by passing demand_params['W_demand'] explicitly.
        W_D = dp.get('W_demand', None)
        if W_D is None:
            W_D = np.linalg.inv((1 / N) * Z_D.T @ Z_D)

        # Direct partials of xi w.r.t. theta = (alpha, sigma_1, ..., sigma_L)
        # d(xi)/d(alpha) = -prices
        # d(xi)/d(sigma_l) = -log(s_{j|g_l})
        n_theta = 1 + L
        dxi_dtheta = np.zeros((N, n_theta))
        dxi_dtheta[:, 0] = -prices
        for l in range(L):
            dxi_dtheta[:, 1 + l] = -log_within_nest_shares[l]

        # Concentration adjustment (matches PyBLP path _compute_demand_adjustment_gradient).
        #
        # In both PyBLP's estimation and this analytic path, the linear coefficients
        # beta (including the intercept and non-price coefficients) are concentrated
        # out via 2SLS at fixed non-linear parameters (alpha, rho). The asymptotic
        # expansion theta_hat - theta_0 = -Lambda h(theta_0) + o_p(n^{-1/2}) for the
        # DMSS (2024) Appendix C eq (77) correction is about the PROFILED moments,
        # which treat beta as a function of (alpha, rho) via the 2SLS FOC.
        #
        # The profiled gradient is:
        #    d(xi_profiled)/d(theta) = d(xi)/d(theta) - X_D @ (d(beta_hat)/d(theta))
        # where d(beta_hat)/d(theta) = (X_D' Z_D W_D Z_D' X_D)^{-1} X_D' Z_D W_D Z_D' @ d(xi)/d(theta).
        # This gives
        #    partial_xi_theta = dxi_dtheta - X_D @ inv(X_D' Z_D W_D Z_D' X_D) @ X_D' Z_D W_D Z_D' @ dxi_dtheta
        # which is the 2SLS residual of dxi_dtheta on X_D instrumented by Z_D.
        #
        # For pure logit this reduces to the standard 2SLS delta-method correction with
        # beta concentrated. For nested logit it correctly profiles beta out at fixed rho.
        # In both cases the result matches the PyBLP path to machine precision.
        if X_D.shape[1] > 0:
            XtZW = X_D.T @ Z_D @ W_D  # (K_x, K_z)
            M_xx = XtZW @ Z_D.T @ X_D  # (K_x, K_x)
            projection_coeffs = np.linalg.inv(M_xx) @ (XtZW @ Z_D.T @ dxi_dtheta)  # (K_x, n_theta)
            partial_xi_theta = dxi_dtheta - X_D @ projection_coeffs  # (N, n_theta)
        else:
            partial_xi_theta = dxi_dtheta

        # H = (1/N) Z_D' @ partial_xi_theta
        H = (1 / N) * Z_D.T @ partial_xi_theta  # (K_z, n_theta)
        H_prime_wd = H.T @ W_D  # (n_theta, K_z)

        # Gradient of markups w.r.t. theta
        # For each model m, d(markup_m)/d(theta_k) is computed analytically.
        # The markup is a function of the Jacobian D(alpha, sigma) and shares:
        #   markup = f(D, Omega, s) where D depends on alpha and sigma.
        # We compute d(markup)/d(theta) per market using the chain rule.

        # First, need to know which cost-shifter columns to residualize against
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
            w_absorbed, _ = self._absorb_cost_ids(w_for_ols)
        else:
            w_absorbed = w_for_ols
        Q_w = np.linalg.qr(w_absorbed, mode='reduced')[0] if w_absorbed.any() else None

        gradient_markups = np.zeros((M, N, n_theta), dtype=options.dtype)

        for t in markets:
            idx = np.where(market_ids == t)[0]
            J_t = len(idx)
            s_t = shares[idx]
            nesting_t = [arr[idx] for arr in nesting_arrays]

            for m in range(M):
                # Get the ownership matrix for this model and market
                model_type = self.models["models_downstream"][m]
                if model_type == 'perfect_competition':
                    # Markups are zero regardless of alpha/sigma, gradient is zero
                    continue

                ownership_m = self.models["ownership_downstream"][m]
                if ownership_m is not None:
                    O_t = ownership_m[idx]
                    O_t = O_t[:, ~np.isnan(O_t).all(axis=0)]
                else:
                    O_t = np.ones((J_t, J_t))

                mu_t = markups[m][idx].flatten()

                # d(markup)/d(alpha) = -markup / alpha (analytical, for standard models).
                #
                # Derivation: D = alpha * N(sigma, s), so markup = -(O*D^T)^{-1} s
                # = -alpha^{-1} (O*N^T)^{-1} s. Differentiating w.r.t. alpha with
                # shares held fixed:
                #     d(markup)/d(alpha) = alpha^{-2} (O*N^T)^{-1} s
                #                        = alpha^{-2} * (-alpha) * markup    [using (O*N^T)^{-1} s = -alpha * markup]
                #                        = -markup / alpha.
                # Equivalently: markup is homogeneous of degree -1 in D, and
                # D scales linearly in alpha, so Euler's theorem gives the same result.
                #
                # Sign check: with alpha < 0 and markup > 0, -markup/alpha > 0, matching
                # the expectation that less-elastic demand (alpha closer to 0) increases markups.
                # Custom models may not have this property, so they are handled by the
                # finite-difference block below.
                if self.models["custom_model_specification"][m] is None:
                    gradient_markups[m][idx, 0] = -mu_t / alpha

                # d(markup)/d(sigma_l): analytical derivative of Jacobian w.r.t. sigma_l,
                # then implicit differentiation of the FOC to get d(markup)/d(sigma_l).
                # Compute D at the actual sigma once for this market (shared across sigma_l's)
                D_actual = _nested_logit_jacobian(alpha, sigma, s_t, nesting_t) if L > 0 else None
                for l in range(L):
                    dD_dsigma = _nested_logit_jacobian_derivative(alpha, sigma, s_t, nesting_t, l)

                    # d(markup)/d(sigma_l) via implicit differentiation of FOC
                    # FOC: (Omega * D') markup + s = 0
                    # Differentiating: (Omega * d(D')/d(sigma)) markup + (Omega * D') d(markup)/d(sigma) = 0
                    # d(markup)/d(sigma) = -(Omega * D')^{-1} (Omega * d(D')/d(sigma)) markup
                    if model_type == 'bertrand':
                        A = O_t * D_actual.T
                        dA = O_t * dD_dsigma.T
                        d_mu = -np.linalg.solve(A, dA @ mu_t)
                    elif model_type == 'cournot':
                        D_inv = np.linalg.inv(D_actual)
                        dD_inv = -D_inv @ dD_dsigma @ D_inv
                        d_mu = -(O_t * dD_inv) @ s_t
                    elif model_type == 'monopoly':
                        dA = dD_dsigma.T
                        d_mu = -np.linalg.solve(D_actual.T, dA @ mu_t)
                    elif model_type == 'mix_cournot_bertrand':
                        mix_flag_m = self.models["mix_flag"][m]
                        b_t = mix_flag_m[idx].flatten().astype(bool)
                        c_t = ~b_t
                        if c_t.any() and b_t.any():
                            D_BB = D_actual[np.ix_(b_t, b_t)]
                            D_BC = D_actual[np.ix_(b_t, c_t)]
                            D_CB = D_actual[np.ix_(c_t, b_t)]
                            D_CC = D_actual[np.ix_(c_t, c_t)]
                            D_CC_inv = np.linalg.inv(D_CC)
                            O_BB = O_t[np.ix_(b_t, b_t)]
                            O_CC = O_t[np.ix_(c_t, c_t)]
                            dD_BB = dD_dsigma[np.ix_(b_t, b_t)]
                            dD_BC = dD_dsigma[np.ix_(b_t, c_t)]
                            dD_CB = dD_dsigma[np.ix_(c_t, b_t)]
                            dD_CC = dD_dsigma[np.ix_(c_t, c_t)]
                            # Cournot block: d(mu_C)/d(sigma)
                            dD_CC_inv = -D_CC_inv @ dD_CC @ D_CC_inv
                            d_mu_C = -(O_CC * dD_CC_inv) @ s_t[c_t]
                            # Bertrand block: implicit diff of A_B mu_B + s_B = 0
                            # A_B = O_BB * (D_BC D_CC^{-1} D_CB + D_BB)
                            Schur = D_BC @ D_CC_inv @ D_CB + D_BB
                            dSchur = (dD_BC @ D_CC_inv @ D_CB + D_BC @ dD_CC_inv @ D_CB
                                      + D_BC @ D_CC_inv @ dD_CB + dD_BB)
                            A_B = O_BB * Schur
                            dA_B = O_BB * dSchur
                            d_mu_B = -np.linalg.solve(A_B, dA_B @ mu_t[b_t])
                            d_mu = np.zeros(J_t)
                            d_mu[b_t] = d_mu_B.flatten()
                            d_mu[c_t] = d_mu_C.flatten()
                        else:
                            d_mu = np.zeros(J_t)
                    else:
                        d_mu = np.zeros(J_t)

                    gradient_markups[m][idx, 1 + l] = d_mu

        # For vertical and custom models, use finite differences of the entire markup
        # computation w.r.t. each demand parameter. This is cheap (only re-evaluates
        # closed-form Jacobian and Hessian, no BLP inversion).
        # - Vertical: the per-market analytical sigma derivative only handles the downstream
        #   FOC; the upstream depends on the Hessian and passthrough matrix.
        # - Custom: the user's markup function may not be homogeneous in alpha or have a
        #   known relationship to sigma, so all derivatives must be finite-differenced.
        eps_fd = 1e-7
        alpha_orig = dp['alpha']
        sigma_orig = list(dp.get('sigma', []))
        for m in range(M):
            needs_fd = (self.models["models_upstream"][m] is not None
                        or self.models["custom_model_specification"][m] is not None)
            if not needs_fd:
                continue
            try:
                # d(markup)/d(alpha) via finite difference
                self.demand_params['alpha'] = alpha + eps_fd / 2
                markups_up, _, _ = self._perturb_and_build_markups()
                self.demand_params['alpha'] = alpha - eps_fd / 2
                markups_dn, _, _ = self._perturb_and_build_markups()
                gradient_markups[m][:, 0] = (
                    markups_up[m].flatten() - markups_dn[m].flatten()
                ) / eps_fd
                # d(markup)/d(sigma_l) via finite difference
                for l in range(L):
                    sigma_plus = list(sigma_orig)
                    sigma_minus = list(sigma_orig)
                    sigma_plus[l] = sigma_orig[l] + eps_fd / 2
                    sigma_minus[l] = sigma_orig[l] - eps_fd / 2
                    self.demand_params['sigma'] = sigma_plus
                    markups_up, _, _ = self._perturb_and_build_markups()
                    self.demand_params['sigma'] = sigma_minus
                    markups_dn, _, _ = self._perturb_and_build_markups()
                    gradient_markups[m][:, 1 + l] = (
                        markups_up[m].flatten() - markups_dn[m].flatten()
                    ) / eps_fd
            finally:
                self.demand_params['alpha'] = alpha_orig
                self.demand_params['sigma'] = sigma_orig

        # Residualize gradient_markups on cost shifters (same basis as omega)
        for m in range(M):
            for k in range(n_theta):
                col = gradient_markups[m][:, k]
                if self._absorb_cost_ids is not None:
                    col, _ = self._absorb_cost_ids(col.reshape(-1, 1))
                    col = col.flatten()
                if Q_w is not None:
                    col = col - Q_w @ (Q_w.T @ col)
                gradient_markups[m][:, k] = col

        return gradient_markups, H_prime_wd, H, h_i, h.reshape(-1, 1)

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

    def _compute_demand_adjustment_gradient(self, N: int, advalorem_tax_adj: list, cost_scaling: list,
                                               marginal_cost_base: Optional[list] = None):
        """Compute the finite-difference gradient of markups w.r.t. demand parameters.

        When endogenous_cost_component is set and marginal_cost_base is provided, also computes the
        per-instrument gradient of gamma (the scale-economies coefficient) w.r.t. demand parameters.
        This accounts for the dependence of gamma on theta through the markup-implied marginal costs.

        Returns gradient_markups, H_prime_wd, H, h_i, h, and optionally gradient_gamma_per_instrument.
        """
        M = self.M
        ZD = self.demand_results.problem.products.ZD
        # Choose the pyblp weight matrix per options.demand_adjustment_weight.
        # Default 'W' is the weight actually used in estimation (correct per DMSS
        # Appendix C). 'updated_W' reproduces the pre-v0.3.3 behavior for
        # validation/replication against older results.
        weight_choice = getattr(options, 'demand_adjustment_weight', 'W')
        if weight_choice == 'W':
            WD = self.demand_results.W
        elif weight_choice == 'updated_W':
            WD = self.demand_results.updated_W
        else:
            raise ValueError(
                f"options.demand_adjustment_weight must be 'W' or 'updated_W', "
                f"got {weight_choice!r}."
            )
        h = self.demand_results.moments
        h_i = ZD * self.demand_results.xi
        K2 = self.demand_results.problem.K2
        D = self.demand_results.problem.D

        XD = self.demand_results.problem.products.X1
        XD_column_names = self.demand_results.problem.products.dtype.fields['X1'][2]
        price_in_linear_parameters = 'prices' in XD_column_names
        if price_in_linear_parameters:
            XD = np.delete(XD, XD_column_names.index('prices'), 1)

        partial_y_theta = (
            np.append(self.demand_results.xi_by_theta_jacobian, -self.demand_results.problem.products.prices, axis=1)
            if price_in_linear_parameters else self.demand_results.xi_by_theta_jacobian
        )

        if self.demand_results.problem.ED > 0:
            partial_y_theta = self.demand_results.problem._absorb_demand_ids(partial_y_theta)
            partial_y_theta = np.reshape(
                partial_y_theta[0], [N, len(self.demand_results.theta) + int(price_in_linear_parameters)]
            )

        if not XD.shape[1]:
            partial_xi_theta = partial_y_theta
        else:
            try:
                product = XD @ inv(XD.T @ ZD @ WD @ ZD.T @ XD) @ (XD.T @ ZD @ WD @ ZD.T @ partial_y_theta)
                partial_xi_theta = partial_y_theta - product
            except np.linalg.LinAlgError:
                output(
                    "Dimension mismatch occurred. This can happen if you specify a supply side in the demand "
                    "estimation."
                )
                raise

        H = 1 / N * (np.transpose(ZD) @ partial_xi_theta)
        H_prime_wd = np.transpose(H) @ WD

        epsilon = options.finite_differences_epsilon
        n_theta = len(self.demand_results.theta) + int(price_in_linear_parameters)
        gradient_markups = np.zeros((M, N, n_theta), dtype=options.dtype)

        # When endogenous_cost_component is set, also track how gamma changes with theta
        L = self.L
        compute_gamma_gradient = (self.endogenous_cost_component is not None and marginal_cost_base is not None)
        gradient_gamma_per_instrument = None
        if compute_gamma_gradient:
            gradient_gamma_per_instrument = [np.zeros((M, n_theta), dtype=options.dtype) for _ in range(L)]

        theta_index = 0
        delta_estimate = self.demand_results.delta

        def apply_tax_adjustment(markups_list):
            for m in range(M):
                markups_list[m] = (advalorem_tax_adj[m] * markups_list[m]) / (1 + cost_scaling[m])
            return markups_list

        def _record_gamma_gradient(markups_u, markups_l, theta_idx):
            """Finite-difference gamma w.r.t. the current demand parameter perturbation."""
            if not compute_gamma_gradient:
                return
            mc_u = [self.products.prices - markups_u[m] for m in range(M)]
            mc_l = [self.products.prices - markups_l[m] for m in range(M)]
            for l in range(L):
                cp_u, _, _ = self._compute_iv_correction(l, M, N, mc_u)
                cp_l, _, _ = self._compute_iv_correction(l, M, N, mc_l)
                for m in range(M):
                    gamma_u = float(cp_u[m][-1])
                    gamma_l = float(cp_l[m][-1])
                    gradient_gamma_per_instrument[l][m, theta_idx] = (gamma_u - gamma_l) / epsilon

        # Save all demand parameter state before perturbation so it can be restored on any exit path
        sigma_saved = self.demand_results.sigma.copy()
        pi_saved = self.demand_results.pi.copy()
        beta_saved = self.demand_results.beta.copy()
        rho_saved = self.demand_results.rho.copy() if len(self.demand_results.rho) != 0 else None

        try:
            # perturb sigma parameters
            for (i, j) in itertools.product(range(K2), range(K2)):
                if not self.demand_results.sigma[i, j] == 0:
                    sigma_initial = self.demand_results.sigma[i, j]
                    self.demand_results._sigma[i, j] = sigma_initial - epsilon / 2
                    with contextlib.redirect_stdout(open(os.devnull, 'w')):
                        self.demand_results._delta = self.demand_results.compute_delta()
                    markups_l, _, _ = self._perturb_and_build_markups()
                    self.demand_results._sigma[i, j] = sigma_initial + epsilon / 2
                    with contextlib.redirect_stdout(open(os.devnull, 'w')):
                        self.demand_results._delta = self.demand_results.compute_delta()
                    markups_u, _, _ = self._perturb_and_build_markups()
                    markups_u_adj = apply_tax_adjustment(markups_u)
                    markups_l_adj = apply_tax_adjustment(markups_l)
                    gradient_markups = self._compute_first_difference_markups(
                        markups_u_adj, markups_l_adj, epsilon, theta_index, gradient_markups
                    )
                    _record_gamma_gradient(markups_u_adj, markups_l_adj, theta_index)
                    self.demand_results._sigma[i, j] = sigma_initial
                    theta_index += 1

            # perturb pi parameters
            for (i, j) in itertools.product(range(K2), range(D)):
                if not self.demand_results.pi[i, j] == 0:
                    pi_initial = self.demand_results.pi[i, j]
                    markups_l, _, _ = self._compute_perturbation(i, j, pi_initial - epsilon / 2)
                    markups_u, _, _ = self._compute_perturbation(i, j, pi_initial + epsilon / 2)
                    markups_u_adj = apply_tax_adjustment(markups_u)
                    markups_l_adj = apply_tax_adjustment(markups_l)
                    gradient_markups = self._compute_first_difference_markups(
                        markups_u_adj, markups_l_adj, epsilon, theta_index, gradient_markups
                    )
                    _record_gamma_gradient(markups_u_adj, markups_l_adj, theta_index)
                    self.demand_results._pi[i, j] = pi_initial
                    theta_index += 1

            # perturb alpha (price coefficient in linear parameters)
            price_index = [idx for idx, v in enumerate(self.demand_results.beta_labels) if v == 'prices']
            if price_index:
                alpha_initial = self.demand_results.beta[price_index].copy()
                self.demand_results._beta[price_index] = alpha_initial - epsilon / 2
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    self.demand_results._delta = self.demand_results.compute_delta()
                markups_l, _, _ = self._perturb_and_build_markups()
                self.demand_results._beta[price_index] = alpha_initial + epsilon / 2
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    self.demand_results._delta = self.demand_results.compute_delta()
                markups_u, _, _ = self._perturb_and_build_markups()
                markups_u_adj = apply_tax_adjustment(markups_u)
                markups_l_adj = apply_tax_adjustment(markups_l)
                gradient_markups = self._compute_first_difference_markups(
                    markups_u_adj, markups_l_adj, epsilon, theta_index, gradient_markups
                )
                _record_gamma_gradient(markups_u_adj, markups_l_adj, theta_index)
                self.demand_results._beta[price_index] = alpha_initial
                theta_index += 1

            # perturb rho (nesting parameter)
            if rho_saved is not None:
                rho_initial = rho_saved.copy()
                self.demand_results._rho = rho_initial - epsilon / 2
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    self.demand_results._delta = self.demand_results.compute_delta()
                markups_l, _, _ = self._perturb_and_build_markups()
                self.demand_results._rho = rho_initial + epsilon / 2
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    self.demand_results._delta = self.demand_results.compute_delta()
                markups_u, _, _ = self._perturb_and_build_markups()
                markups_u_adj = apply_tax_adjustment(markups_u)
                markups_l_adj = apply_tax_adjustment(markups_l)
                gradient_markups = self._compute_first_difference_markups(
                    markups_u_adj, markups_l_adj, epsilon, theta_index, gradient_markups
                )
                _record_gamma_gradient(markups_u_adj, markups_l_adj, theta_index)

        finally:
            # Restore all demand parameter state regardless of success or failure
            self.demand_results._sigma[:] = sigma_saved
            self.demand_results._pi[:] = pi_saved
            self.demand_results._beta[:] = beta_saved
            if rho_saved is not None:
                self.demand_results._rho[:] = rho_saved
            self.demand_results._delta = delta_estimate

        return gradient_markups, H_prime_wd, H, h_i, h, gradient_gamma_per_instrument

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

    def _compute_first_difference_markups(
            self, markups_u: list, markups_l: list, epsilon: float, theta_index: int, gradient_markups: Array
    ) -> Array:
        """Compute finite-difference approximation of markup gradient w.r.t. a single demand parameter."""
        if self.endogenous_cost_component is not None:
            endog_col_idx = next(
                i for i, f in enumerate(self._w_formulation)
                if str(f) == self.endogenous_cost_component
            )
            exog_col_indices = [i for i in range(self.products.w.shape[1]) if i != endog_col_idx]
            w_for_ols = self.products.w[:, exog_col_indices]
        else:
            w_for_ols = self.products.w
        # Absorb w before QR so that the projection is in the same basis as the absorbed diff_markups
        if self._absorb_cost_ids is not None:
            w_absorbed, _ = self._absorb_cost_ids(w_for_ols)
        else:
            w_absorbed = w_for_ols
        Q_w = np.linalg.qr(w_absorbed, mode='reduced')[0] if w_absorbed.any() else None
        for m in range(self.M):
            diff_markups = (markups_u[m] - markups_l[m]) / epsilon
            if self._absorb_cost_ids is not None:
                diff_markups, _ = self._absorb_cost_ids(diff_markups)
            if Q_w is not None:
                resid = diff_markups - Q_w @ (Q_w.T @ diff_markups)
                gradient_markups[m][:, theta_index] = np.squeeze(resid)
            else:
                gradient_markups[m][:, theta_index] = np.squeeze(diff_markups)
        return gradient_markups

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

    def _compute_perturbation(self, i: int, j: int, perturbation: float):
        """Perturb pi[i, j] to the given value, recompute delta, and return new markups."""
        self.demand_results._pi[i, j] = perturbation
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            self.demand_results._delta = self.demand_results.compute_delta()
        return self._perturb_and_build_markups()

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


