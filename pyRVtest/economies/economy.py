"""Economy underlying the BLP model."""

import abc
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from pyblp.utilities.algebra import precisely_identify_collinearity, precisely_identify_psd
from pyblp.utilities.basics import (
    Array, Error, RecArray, StringRepresentation, format_table, get_indices, output
)

from .. import exceptions, options
from ..configurations.formulation import Formulation, Absorb, ModelFormulation
from ..primitives import Container


class Economy(Container, StringRepresentation):
    """An abstract economy underlying the BLP model."""

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
            header.append("K{0}".format(instrument))
            values.append(str(self.Dict_K["K{0}".format(instrument)]))

        return format_table(header, values, title="Dimensions")

    def _format_formulations(self) -> str:
        """Formation information about the formulations of the economy as a string."""

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
        """Formation information about the formulations of the economy as a string."""

        # construct the data
        data: List[List[str]] = []
        if self.markups[0] is None:
            data.append(["Model - Downstream"] + [self.models.models_downstream[i] for i in range(self.M)])
            data.append(["Model - Upstream"] + [self.models.models_upstream[i] for i in range(self.M)])
            data.append(["Firm id - Downstream"] + [self.models.firm_ids_downstream[i] for i in range(self.M)])
            data.append(["Firm id - Upstream"] + [self.models.firm_ids_upstream[i] for i in range(self.M)])
            data.append(["VI ind"] + [self.models.vertical_integration_index[i] for i in range(self.M)])
            data.append(["Cost Scaling Column"] + [self.models.cost_scaling_column[i] for i in range(self.M)])
            data.append(["Unit Tax"] + [self.models.unit_tax[i] for i in range(self.M)])
            data.append(["Advalorem Tax"] + [self.models.advalorem_tax[i] for i in range(self.M)])
            data.append(["Advalorem Payer"] + [self.models.advalorem_payer[i] for i in range(self.M)])
            data.append(
                ["User Supplied Markups"] + [self.models.user_supplied_markups_name[i] for i in range(self.M)]
            )
            header = [" "] + [f" {i} " for i in range(self.M)]
        else:
            data.append(["Markups Supplied by User"])    
            # TODO: is this header correct?
            header = [" "]

        return format_table(header, *data, title="Models")    

    def _detect_collinearity(self) -> None:
        """Detect any collinearity issues in product data matrices."""

        # skip collinearity checking when it is disabled via zero tolerances
        if max(options.collinear_atol, options.collinear_rtol) <= 0:
            return

        # collect labels for columns of matrices that will be checked for collinearity issues
        matrix_labels = {'w': [str(f) for f in self._w_formulation]}
        for zz in range(len(self.Dict_Z_formulation)):
            matrix_labels.update(
                {"Z{0}".format(zz): [str(f) for f in self.Dict_Z_formulation["_Z{0}_formulation".format(zz)]]}
            )
            matrix_labels.update(
                {"Z{0}".format(zz): [str(f) for f in self._w_formulation] + matrix_labels["Z{0}".format(zz)]}
            )

        # check each matrix for collinearity
        for name, labels in matrix_labels.items():
            collinear, successful = precisely_identify_collinearity(self.products[name])
            common_message = "To disable collinearity checks, set options.collinear_atol = options.collinear_rtol = 0."
            for zz in range(len(self.Dict_Z_formulation)):
                if name in {'w', 'Z{0}'.format(zz)}:
                    common_message = f"Absorbed fixed effects may be creating collinearity problems. {common_message}"
            if not successful:
                raise ValueError(
                    f"Failed to compute the QR decomposition of {name} while checking for collinearity issues. "
                    f"{common_message}"
                )
            if collinear.any():
                collinear_labels = ", ".join(l for l, c in zip(labels, collinear) if c)
                raise ValueError(
                    f"Detected collinearity issues with [{collinear_labels}] and at least one other column in {name}. "
                    f"{common_message}"
                )

    @staticmethod
    def _detect_psd(matrix: Array, name: str) -> None:
        """Detect whether a matrix is PSD."""
        psd, successful = precisely_identify_psd(matrix)
        common_message = "To disable PSD checks, set options.psd_atol = options.psd_rtol = numpy.inf."
        if not successful:
            raise ValueError(f"Failed to compute the SVD of {name} while checking that it is PSD. {common_message}")
        if not psd:
            raise ValueError(f"{name} must be a PSD matrix. {common_message}")

    @staticmethod
    def _handle_errors(errors: List[Error], error_behavior: str = 'raise') -> None:
        """Either raise or output information about any errors."""
        if errors:
            if error_behavior == 'raise':
                raise exceptions.MultipleErrors(errors)
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

    def _validate_product_ids(self, product_ids: Sequence[Any], market_ids: Optional[Array] = None) -> None:
        """Validate that product IDs either contain None (denoting the outside option) or contain at least one product
        ID for each market in the data (or in specific markets if specified). Also verify that each product ID appears
        only once in each relevant market.
        """
        if self.unique_product_ids.size == 0 and any(i is not None for i in product_ids):
            raise ValueError("Product IDs must have been specified.")

        if market_ids is None:
            market_ids = self.unique_market_ids

        for t in market_ids:
            counts = []
            for product_id in product_ids:
                count = 1
                if product_id is not None:
                    count = (self.products.product_ids[self._product_market_indices[t]] == product_id).sum()
                    if count > 1:
                        raise ValueError(
                            f"Product IDs should be unique within markets, but ID '{product_id}' shows up {count} "
                            f"times in market '{t}'."
                        )
                counts.append(count)

            if all(c == 0 for c in counts):
                raise ValueError(
                    f"None of the product_ids {sorted(list(product_ids))} show up in market '{t}' with IDs: "
                    f"{list(sorted(self.products.product_ids[self._product_market_indices[t]]))}."
                )

    def _coerce_optional_firm_ids(self, firm_ids: Optional[Any], market_ids: Optional[Array] = None) -> Array:
        """Coerce optional array-like firm IDs into a column vector and validate it. By default, assume that firm IDs
        are for all markets.
        """
        if firm_ids is None:
            return None
        firm_ids = np.c_[np.asarray(firm_ids, options.dtype)]
        rows = self.N
        if market_ids is not None:
            rows = sum(i.size for t, i in self._product_market_indices.items() if t in market_ids)
        if firm_ids.shape != (rows, 1):
            raise ValueError(f"firm_ids must be None or a {rows}-vector.")
        return firm_ids

    def _coerce_optional_ownership(self, ownership: Optional[Any], market_ids: Optional[Array] = None) -> Array:
        """Coerce optional array-like ownership matrices into a stacked matrix and validate it. By default, assume that
        ownership matrices are for all markets.
        """
        if ownership is None:
            return None
        ownership = np.c_[np.asarray(ownership, options.dtype)]
        rows = self.N
        columns = self._max_J
        if market_ids is not None:
            rows = sum(i.size for t, i in self._product_market_indices.items() if t in market_ids)
            columns = max(i.size for t, i in self._product_market_indices.items() if t in market_ids)
        if ownership.shape != (rows, columns):
            raise ValueError(f"ownership must be None or a {rows} by {columns} matrix.")
        return ownership
