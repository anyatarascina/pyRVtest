"""Primitive data structures that constitute the foundation of the BLP model."""

import abc
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from . import options
from .configurations.formulation import ColumnFormulation, Formulation, ModelFormulation
from .utilities.basics import Array, Groups, RecArray, extract_matrix, structure_matrices
from . import construction


class Products(object):
    r"""Product data structured as a record array.

    Attributes
    ----------

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
            cls, cost_formulation: Formulation,
            instrument_formulation: Sequence[Optional[Formulation]], product_data: Mapping) -> RecArray:
        """Structure product data."""

        # validate the cost formulation
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
        if hasattr(instrument_formulation, '__len__'):
            L = len(instrument_formulation)
        else:
            L = 1

        if L == 1:
            if not isinstance(instrument_formulation, Formulation):
                raise TypeError("instrument_formulation must be a single Formulation instance.")
        elif L > 1:
            if not all(isinstance(f, Formulation) for f in instrument_formulation):
                raise TypeError("Each formulation in instrument_formulation must be a Formulation.")

        # if not isinstance(instrument_formulation, Formulation):
        #    raise TypeError("instrument_formulation must be a Formulation instance or None.")
        if instrument_formulation is None:
            raise ValueError("The formulation for instruments for testing must be specified.")
        
        # build Z
        Instr = {}
        if L == 1:
            tmp_Z, tmp_Z_formulation, tmp_Z_data = instrument_formulation._build_matrix(product_data)
            for kk in tmp_Z_formulation:
                if kk in w_data:
                    raise NameError("Z must be excluded from marginal cost.")

            Instr["Z0"] = tmp_Z
            Instr["Z0_formulation"] = tmp_Z_formulation
            Instr["Z0_data"] = tmp_Z_data
        elif L > 1:
            for zz in range(L):
                tmp_Z, tmp_Z_formulation, tmp_Z_data = instrument_formulation[zz]._build_matrix(product_data)

                for kk in tmp_Z_formulation:
                    if kk in w_data:
                        raise NameError("Z must be excluded from marginal cost.")

                Instr["Z{0}".format(zz)] = tmp_Z
                Instr["Z{0}_formulation".format(zz)] = tmp_Z_formulation
                Instr["Z{0}_data".format(zz)] = tmp_Z_data

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
            'market_ids': (market_ids, np.object),
            'cost_ids': (cost_ids, np.object),
            'nesting_ids': (nesting_ids, np.object),
            'product_ids': (product_ids, np.object),
            'clustering_ids': (clustering_ids, np.object),
            'shares': (shares, options.dtype),
            'prices': (prices, options.dtype)
        })
        product_mapping.update({
            (tuple(w_formulation), 'w'): (w, options.dtype),
        })
        
        for zz in range(L):
            # TODO: format this line
            product_mapping.update({(tuple(Instr["Z{0}_formulation".format(zz)]), 'Z{0}'.format(zz)): (Instr["Z{0}".format(zz)], options.dtype)})

        # structure and validate variables underlying X1, X2, and X3
        underlying_data = {k: (v, options.dtype) for k, v in {**w_data}.items() if k != 'shares'}
        for zz in range(L):
            underlying_data.update(
                {k: (v, options.dtype) for k, v in {**Instr["Z{0}_data".format(zz)]}.items() if k != 'shares'}
            )
        invalid_names = set(underlying_data) & {k if isinstance(k, str) else k[1] for k in product_mapping}
        if invalid_names:
            raise NameError(f"These reserved names in product_formulations are invalid: {list(invalid_names)}.")

        return structure_matrices({**product_mapping, **underlying_data})


class Models(object):
    r"""Models structured as a record array.

    Attributes
    ----------

    """

    def __new__(
            cls, model_formulations: Sequence[Optional[ModelFormulation]],
            product_data: Mapping) -> RecArray:
        """Structure agent data."""

        # data structures may be empty

        # validate the model formulations
        if not all(isinstance(f, ModelFormulation) or f is None for f in model_formulations):
            raise TypeError("Each formulation in model_formulations must be a ModelFormulation instance or None.")
        M = len(model_formulations)
        if M < 2:
            raise ValueError("At least two model formulations must be specified.")

        omatrices_downstream = [None]*M
        omatrices_upstream = [None]*M
        firmids_downstream = [None]*M
        firmids_upstream = [None]*M
        VI = [None]*M
        VI_ind = [None]*M
        models_upstream = [None]*M
        models_downstream = [None]*M
        custom_model = [None] * M

        # make ownership matrices and extract vertical integration
        for kk in range(M):
            model = model_formulations[kk]._build_matrix(product_data)

            # TODO: allow for different way to enter data
            models_downstream[kk] = model["model_downstream"]
            if model["model_upstream"] is not None:
                models_upstream[kk] = model["model_upstream"]  
            
            if model["model_downstream"] == "monopoly":
                omatrices_downstream[kk] = construction.build_ownership_testing(
                    product_data, model["ownership_downstream"], 'monopoly'
                )
                firmids_downstream[kk] = "monopoly"
            else:
                omatrices_downstream[kk] = construction.build_ownership_testing(
                    product_data, model["ownership_downstream"], model["kappa_specification_downstream"]
                )
                firmids_downstream[kk] = model["ownership_downstream"]
            
            if model["model_upstream"] == "monopoly":
                omatrices_upstream[kk] = construction.build_ownership_testing(
                    product_data, model["ownership_upstream"], 'monopoly'
                )
                firmids_upstream[kk] = "monopoly"
            elif model["ownership_upstream"] is not None:
                omatrices_upstream[kk] = construction.build_ownership_testing(
                    product_data, model["ownership_upstream"], model["kappa_specification_upstream"]
                )
                firmids_upstream[kk] = model["ownership_upstream"]
            if model["vertical_integration"] is not None:
                VI[kk] = extract_matrix(product_data, model["vertical_integration"])
                VI_ind[kk] = model["vertical_integration"]
            custom_model[kk] = model["custom_model_specification"]

        models_mapping = pd.Series({
            'models_downstream': models_downstream,
            'models_upstream': models_upstream,
            'firmids_downstream': firmids_downstream,
            'firmids_upstream': firmids_upstream,
            'ownership_downstream': omatrices_downstream,
            'ownership_upstream': omatrices_upstream,
            'VI': VI,
            'VI_ind': VI_ind,
            'custom_model_specification': custom_model
        })

        return models_mapping


class Container(abc.ABC):
    """An abstract container for structured product and agent data."""

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
        stop = 0
        zz = 0
        
        while stop == 0:
            if 'Z{0}'.format(zz) in self.products.dtype.fields:
                self._Z_formulation = self.products.dtype.fields['Z{0}'.format(zz)][2]
                self.Dict_Z_formulation.update({"_Z{0}_formulation".format(zz): self._Z_formulation})
                zz = zz + 1 
            else:
                stop = 1
