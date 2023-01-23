"""Economy underlying the firm conduct testing model."""

import abc
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from pyblp.utilities.basics import Array, RecArray, StringRepresentation, format_table, get_indices

from ..configurations.formulation import Formulation, Absorb, ModelFormulation
from ..primitives import Container


class Economy(Container, StringRepresentation):
    """An abstract economy underlying the firm conduct testing model."""

    model_formulations: Sequence[Optional[ModelFormulation]]
    cost_formulation: Formulation
    instrument_formulation: Formulation
    markups: RecArray
    unique_market_ids: Array
    unique_nesting_ids: Array
    unique_product_ids: Array
    T: int
    N: int
    Dict_K: Dict[Union[str, tuple], Tuple[Optional[Array], Any]] = {}
    M: int
    EC: int
    H: int
    L: int
    _market_indices: Dict[Hashable, int]
    _product_market_indices: Dict[Hashable, Array]
    _max_J: int
    _absorb_cost_ids: Optional[Absorb]

    @abc.abstractmethod
    def __init__(
            self, cost_formulation: Formulation, instrument_formulation: Formulation,
            model_formulations: Sequence[Optional[ModelFormulation]],
            products: RecArray, models: RecArray, demand_results: Mapping, markups: RecArray) -> None:
        """Store information about formulations and data. Any fixed effects should be absorbed after initialization."""

        # store data and formulations
        super().__init__(products, models)
        self.cost_formulation = cost_formulation
        self.instrument_formulation = instrument_formulation
        self.model_formulations = model_formulations
        self.demand_results = demand_results
        self.markups = markups

        # identify unique markets, nests, products, and agents
        self.unique_market_ids = np.unique(self.products.market_ids.flatten())
        self.unique_nesting_ids = np.unique(self.products.nesting_ids.flatten())
        self.unique_product_ids = np.unique(self.products.product_ids.flatten())

        # count dimensions
        self.N = self.products.shape[0]
        self.T = self.unique_market_ids.size
        self.L = len(self.instrument_formulation) if hasattr(self.instrument_formulation, '__len__') else 1
        for instrument in range(self.L):
            self.Dict_K.update({"K{0}".format(instrument): self.products["Z{0}".format(instrument)].shape[1]})
        self.M = len(self.model_formulations) if self.markups[0] is None else np.shape(self.markups)[0]
        self.EC = self.products.cost_ids.shape[1]
        self.H = self.unique_nesting_ids.size

        # identify market indices
        self._market_indices = {t: i for i, t in enumerate(self.unique_market_ids)}
        self._product_market_indices = get_indices(self.products.market_ids)

        # identify the largest number of products and agents in a market
        self._max_J = max(i.size for i in self._product_market_indices.values())

        # construct fixed effect absorption functions
        self._absorb_cost_ids = None
        if self.EC > 0:
            assert cost_formulation is not None
            self._absorb_cost_ids = cost_formulation._build_absorb(self.products.cost_ids)

    def __str__(self) -> str:
        """Format economy information as a string."""
        return "\n\n".join([self._format_dimensions(), self._format_formulations(), self._format_model_formulations()])

    def _format_dimensions(self) -> str:
        """Format information about the nonzero dimensions of the economy as a string."""
        header: List[str] = []
        values: List[str] = []
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

        # construct the data
        named_formulations = [(self._w_formulation, "w: Marginal Cost")]
        for instruments in range(self.L):
            named_formulations.append((
                self.Dict_Z_formulation["_Z{0}_formulation".format(instruments)],
                "z{0}: Instruments".format(instruments)
            ))
        data: List[List[str]] = []
        for formulations, name in named_formulations:
            if any(formulations):
                data.append([name] + [str(f) for f in formulations])

        # construct the header
        max_formulations = max(len(r[1:]) for r in data)
        header = ["Column Indices:"] + [f" {i} " for i in range(max_formulations)]
        return format_table(header, *data, title="Formulations")

    def _format_model_formulations(self) -> str:
        """Format information about the model formulations as a string."""

        # construct the data
        data: List[List[str]] = []
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
