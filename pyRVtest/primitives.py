"""Primitive data structures that constitute the foundation of the BLP model."""

import abc
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from pyblp.utilities.basics import Array, Data, Groups, RecArray, extract_matrix, structure_matrices
from pyblp.configurations.formulation import ColumnFormulation

from . import options
from . construction import build_ownership
from .configurations.formulation import Formulation, ModelFormulation
from .data import F_CRITICAL_VALUES_POWER_RHO, F_CRITICAL_VALUES_SIZE_RHO


class Products(object):
    r"""Product data structured as a record array.

    Attributes
    ----------
     market_ids : `ndarray`
        IDs that associate product_data with markets.
    cost_ids : `ndarray`
        IDs that associate product_data with cost-side fixed effects.
    nesting_ids : `ndarray`
        IDs that associate product_data with nesting groups.
    product_ids : `ndarray`
        IDs that identify product_data within markets.
    clustering_ids : `ndarray`
        IDs used to compute clustered standard errors.
    shares : `ndarray`
        Market shares, :math:`s`.
    prices : `ndarray`
        Product prices, :math:`p`.
    Z : `ndarray`
        Instruments, :math: `Z`.
    w : `ndarray`
        Cost-shifters, :math: `w`.
    """

    market_ids: Array
    cost_ids: Array
    nesting_ids: Array
    product_ids: Array
    clustering_ids: Array
    shares: Array
    prices: Array
    Z: Array
    w: Array

    def __new__(
            cls, cost_formulation: Formulation, instrument_formulation: Sequence[Optional[Formulation]],
            product_data: Mapping) -> RecArray:
        """Structure product data."""

        # validate the cost formulations
        if not isinstance(cost_formulation, Formulation):
            raise TypeError("cost_formulation must be a Formulation instance or None.")
        if cost_formulation is None:
            raise ValueError("The formulation for marginal cost must be specified.")

        # build w
        w, w_formulation, w_data = cost_formulation._build_matrix(product_data)
        if 'shares' in w_data:
            raise NameError("shares cannot be included in the formulation for marginal cost.")

        # check that prices are not in X1
        if 'prices' in w_data:
            raise NameError("prices cannot be included in the formulation for marginal cost.")

        # validate the instrument formulation
        if instrument_formulation is None:
            raise ValueError("The formulation for instruments for testing must be specified.")
        if hasattr(instrument_formulation, '__len__'):
            L = len(instrument_formulation)
            if not all(isinstance(f, Formulation) for f in instrument_formulation):
                raise TypeError("Each formulation in instrument_formulation must be a Formulation.")
        else:
            L = 1
            if not isinstance(instrument_formulation, Formulation):
                raise TypeError("instrument_formulation must be a single Formulation instance.")

        # build Z
        Z: Data = {}
        if L == 1:
            Z_l, Z_formulation_l, Z_data_l = instrument_formulation._build_matrix(product_data)
            for z in Z_formulation_l:
                if z in w_data:
                    raise NameError("Z must be excluded from marginal cost.")
            Z["Z0"] = Z_l
            Z["Z0_formulation"] = Z_formulation_l
            Z["Z0_data"] = Z_data_l
        elif L > 1:
            for l in range(L):
                Z_l, Z_formulation_l, Z_data_l = instrument_formulation[l]._build_matrix(product_data)
                for z in Z_formulation_l:
                    if z in w_data:
                        raise NameError("Z must be excluded from marginal cost.")
                Z["Z{0}".format(l)] = Z_l
                Z["Z{0}_formulation".format(l)] = Z_formulation_l
                Z["Z{0}_data".format(l)] = Z_data_l

        # load fixed effect IDs
        cost_ids = None
        if cost_formulation._absorbed_terms:
            cost_ids = cost_formulation._build_ids(product_data)

        # load other IDs
        market_ids = extract_matrix(product_data, 'market_ids')
        nesting_ids = extract_matrix(product_data, 'nesting_ids')
        product_ids = extract_matrix(product_data, 'product_ids')
        clustering_ids = extract_matrix(product_data, 'clustering_ids')
        if market_ids is None:
            raise KeyError("product_data must have a market_ids field.")
        if market_ids.shape[1] > 1:
            raise ValueError("The market_ids field of product_data must be one-dimensional.")
        if nesting_ids is not None and nesting_ids.shape[1] > 1:
            raise ValueError("The nesting_ids field of product_data must be one-dimensional.")
        if product_ids is not None and product_ids.shape[1] > 1:
            raise ValueError("The product_ids field of product_data must be one-dimensional.")
        if clustering_ids is not None:
            if clustering_ids.shape[1] > 1:
                raise ValueError("The clustering_ids field of product_data must be one-dimensional.")
            if np.unique(clustering_ids).size == 1:
                raise ValueError("The clustering_ids field of product_data must have at least two distinct IDs.")

        # load shares
        shares = extract_matrix(product_data, 'shares')
        if shares is None:
            raise KeyError("product_data must have a shares field.")
        if shares.shape[1] > 1:
            raise ValueError("The shares field of product_data must be one-dimensional.")
        if (shares <= 0).any() or (shares >= 1).any():
            raise ValueError(
                "The shares field of product_data must consist of values between zero and one, exclusive.")

        # verify that shares sum to less than one in each market
        market_groups = Groups(market_ids)
        bad_shares_index = market_groups.sum(shares) >= 1
        if np.any(bad_shares_index):
            bad_market_ids = market_groups.unique[bad_shares_index.flat]
            raise ValueError(f"Shares in these markets do not sum to less than 1: {bad_market_ids}.")

        # load prices
        prices = extract_matrix(product_data, 'prices')
        if prices is None:
            raise KeyError("product_data must have a prices field.")
        if prices.shape[1] > 1:
            raise ValueError("The prices field of product_data must be one-dimensional.")
        if (prices < 0).any():
            raise ValueError(
                "The prices field of product_data must consist of values >= zero, exclusive.")

        # structure product fields as a mapping
        product_mapping: Dict[Union[str, tuple], Tuple[Optional[Array], Any]] = {}
        product_mapping.update({
            'market_ids': (market_ids, np.object_),
            'cost_ids': (cost_ids, np.object_),
            'nesting_ids': (nesting_ids, np.object_),
            'product_ids': (product_ids, np.object_),
            'clustering_ids': (clustering_ids, np.object_),
            'shares': (shares, options.dtype),
            'prices': (prices, options.dtype)
        })
        product_mapping.update({(tuple(w_formulation), 'w'): (w, options.dtype), })
        for l in range(L):
            key = (tuple(Z["Z{0}_formulation".format(l)]), 'Z{0}'.format(l))
            product_mapping.update({key: (Z["Z{0}".format(l)], options.dtype)})

        # structure and validate variables underlying instruments
        underlying_data = {k: (v, options.dtype) for k, v in {**w_data}.items() if k != 'shares'}
        for l in range(L):
            underlying_data.update({
                k: (v, options.dtype) for k, v in {**Z["Z{0}_data".format(l)]}.items() if k != 'shares'
            })
        invalid_names = set(underlying_data) & {k if isinstance(k, str) else k[1] for k in product_mapping}
        if invalid_names:
            raise NameError(f"These reserved names in product_formulations are invalid: {list(invalid_names)}.")

        return structure_matrices({**product_mapping, **underlying_data})


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
        if M < 2:
            raise ValueError("At least two model formulations must be specified.")
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
            'user_supplied_markups_name': user_supplied_markups_name
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


def read_critical_values_tables():
    """Read in the critical values for size and power from the corresponding csv file. These will be used to evaluate
    the strength of the instruments."""

    # read in data for critical values for size as a structured array
    critical_values_size = np.genfromtxt(
        F_CRITICAL_VALUES_SIZE_RHO,
        delimiter=',',
        skip_header=1,
        dtype=[('K', 'i4'), ('rho', 'f8'), ('r_075', 'f8'), ('r_10', 'f8'), ('r_125', 'f8')]
    )

    # read in data for critical values for power as a structured array
    critical_values_power = np.genfromtxt(
        F_CRITICAL_VALUES_POWER_RHO,
        delimiter=',',
        skip_header=1,
        dtype=[('K', 'i4'), ('rho', 'f8'), ('r_50', 'f8'), ('r_75', 'f8'), ('r_95', 'f8')]
    )

    return critical_values_power, critical_values_size
