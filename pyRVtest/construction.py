"""Data construction."""

import contextlib
import os
from typing import Any, Callable, Dict, Mapping, Optional, Union

import numpy as np
from numpy.linalg import inv
from pyblp.utilities.basics import Array, RecArray, extract_matrix, get_indices
from pyblp import exceptions

from . import options
from .configurations.formulation import Formulation


def build_ownership_testing(
        product_data: Mapping, firm_col: str,
        kappa_specification: Optional[Union[str, Callable[[Any, Any], float]]] = None) -> Array:
    r"""Build ownership matrices, :math:`O`.

    Ownership or product holding matrices are defined by their cooperation matrix counterparts, :math:`\kappa`. For each
    market :math:`t`, :math:`\mathscr{H}_{jk} = \kappa_{fg}` where :math:`j \in J_{ft}`, the set of products
    produced by firm :math:`f` in the market, and similarly, :math:`g \in J_{gt}`.

    Parameters
    ----------
    product_data : `structured array-like`
        Each row corresponds to a product. Markets can have differing numbers of products. The following fields are
        required (except for ``firm_ids`` when ``kappa_specification`` is one of the special cases):

            - **market_ids** : (`object`) - IDs that associate products with markets.

    firm_col: column in product_data with firm idsthat associate products with firms. This field is ignored if
              ``kappa_specification`` is one of the special cases and not a function.

    kappa_specification : `str or callable, optional`
        Specification for each market's cooperation matrix, :math:`\kappa`, which can either be a general function or a
        string that implements a special case. The general function is is of the following form::

            kappa(f, g) -> value

        where ``value`` is :math:`\mathscr{H}_{jk}` and both ``f`` and ``g`` are firm IDs from the ``firm_ids`` field of
        ``product_data``.

        The default specification, ``lambda: f, g: int(f == g)``, constructs traditional ownership matrices. That is,
        :math:`\kappa = I`, the identity matrix, implies that :math:`\mathscr{H}_{jk}` is :math:`1` if the same firm
        produces products :math:`j` and :math:`k`, and is :math:`0` otherwise.

        If ``firm_ids`` happen to be indices for an actual :math:`\kappa` matrix, ``lambda f, g: kappa[f, g]`` will
        build ownership matrices according to the matrix ``kappa``.

        When one of the special cases is specified, ``firm_ids`` in ``product_data`` are not required and if specified
        will be ignored:

            - ``'monopoly'`` - Monopoly ownership matrices are all ones: :math:`\mathscr{H}_{jk} = 1` for all :math:`j`
              and :math:`k`.

            - ``'single'`` - Single product firm ownership matrices are identity matrices: :math:`\mathscr{H}_{jk} = 1`
              if :math:`j = k` and :math:`0` otherwise.

    Returns
    -------
    `ndarray`
        Stacked :math:`J_t \times J_t` ownership matrices, :math:`\mathscr{H}`, for each market :math:`t`. If a market
        has fewer products than others, extra columns will contain ``numpy.nan``.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/build_ownership.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """

    # validate or use the default kappa specification
    if kappa_specification is None:
        kappa_specification = lambda f, g: np.where(f == g, 1, 0).astype(options.dtype)
    elif callable(kappa_specification):
        kappa_specification = np.vectorize(kappa_specification, [options.dtype])
    elif kappa_specification not in {'monopoly', 'single'}:
        raise ValueError("kappa_specification must be None, callable, 'monopoly', or 'single'.")

    # extract and validate IDs
    market_ids = extract_matrix(product_data, 'market_ids')
    firm_ids = extract_matrix(product_data, firm_col)
    if market_ids is None:
        raise KeyError("product_data must have a market_ids field.")
    if market_ids.shape[1] > 1:
        raise ValueError("The market_ids field of product_data must be one-dimensional.")
    if callable(kappa_specification):
        if firm_ids is None:
            raise KeyError(
                "product_data must have a field named firm_col when kappa_specification is not a special case."
            )
        if firm_ids.shape[1] > 1:
            raise ValueError("The firm_ids field of product_data must be one-dimensional.")

    # determine the overall number of products and the maximum number in a market
    market_indices = get_indices(market_ids)
    N = market_ids.size
    max_J = max(i.size for i in market_indices.values())

    # construct the ownership matrices
    ownership = np.full((N, max_J), np.nan, options.dtype)
    for indices_t in market_indices.values():
        if kappa_specification == 'monopoly':
            ownership[indices_t, :indices_t.size] = 1
        elif kappa_specification == 'single':
            ownership[indices_t, :indices_t.size] = np.eye(indices_t.size)
        else:
            assert callable(kappa_specification) and firm_ids is not None
            ids_t = firm_ids[indices_t]
            tiled_ids_t = np.tile(np.c_[ids_t], ids_t.size)
            ownership[indices_t, :indices_t.size] = kappa_specification(tiled_ids_t, tiled_ids_t.T)

    return ownership


def build_matrix(formulation: Formulation, data: Mapping) -> Array:
    r"""Construct a matrix according to a formulation.

    Parameters
    ----------
    formulation : `Formulation`
        :class:`Formulation` configuration for the matrix. Variable names should correspond to fields in ``data``. The
        ``absorb`` argument of :class:`Formulation` can be used to absorb fixed effects after the matrix has been
        constructed.
    data : `structured array-like`
        Fields can be used as variables in ``formulation``.

    Returns
    -------
    `ndarray`
        The built matrix.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/build_matrix.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """
    if not isinstance(formulation, Formulation):
        raise TypeError("formulation must be a Formulation instance.")
    matrix = formulation._build_matrix(data)[0]
    if formulation._absorbed_terms:
        absorb = formulation._build_absorb(formulation._build_ids(data))
        matrix, errors = absorb(matrix)
        if errors:
            raise exceptions.MultipleErrors(errors)

    return matrix


def data_to_dict(data: RecArray, ignore_empty: bool = True) -> Dict[str, Array]:
    r"""Convert a NumPy record array into a dictionary.

    Most data in PyBLP are structured as NumPy record arrays (e.g., :attr:`Problem.products` and
    :attr:`SimulationResults.product_data`) which can be cumbersome to work with when working with data types that can't
    represent matrices, such as the :class:`pandas.DataFrame`.

    This function converts record arrays created by PyBLP into dictionaries that map field names to one-dimensional
    arrays. Matrices in the original record array (e.g., ``demand_instruments``) are split into as many fields as there
    are columns (e.g., ``demand_instruments0``, ``demand_instruments1``, and so on).

    Parameters
    ----------
    data : `recarray`
        Record array created by PyBLP.
    ignore_empty : `bool, optional`
        Whether to ignore matrices with zero size. By default, these are ignored.

    Returns
    -------
    `dict`
        The data re-structured as a dictionary.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/data_to_dict.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """
    if not isinstance(data, np.recarray):
        raise TypeError("data must be a NumPy record array.")

    mapping: Dict[str, Array] = {}
    for key in data.dtype.names:
        if len(data[key].shape) > 2:
            raise ValueError("Arrays with more than two dimensions are not supported.")
        if ignore_empty and data[key].size == 0:
            continue
        if len(data[key].shape) == 1 or data[key].shape[1] == 1 or data[key].size == 0:
            mapping[key] = data[key].flatten()
            continue
        for index in range(data[key].shape[1]):
            new_key = f'{key}{index}'
            if new_key in data.dtype.names:
                raise KeyError(f"'{key}' cannot be split into columns because '{new_key}' is already a field.")
            mapping[new_key] = data[key][:, index].flatten()

    return mapping


def build_markups(
        products: RecArray, demand_results: Mapping, model_downstream: Array, ownership_downstream: Array,
        model_upstream: Optional[Array] = None, ownership_upstream: Optional[Array] = None,
        vertical_integration: Optional[Array] = None, custom_model_specification: Optional[dict] = None,
        user_supplied_markups: Optional[Array] = None) -> Array:
    r"""This function computes markups for a large set of standard models. These include:
            - standard bertrand with ownership matrix based on firm id
            - price setting with arbitrary ownership matrix (e.g. profit weight model)
            - standard cournot with ownership matrix based on firm id
            - quantity setting with arbitrary ownership matrix (e.g. profit weight model)
            - monopoly
            - bilateral oligopoly with any combination of the above models upstream and downstream
            - bilateral oligopoly as above but with subset of products vertically integrated
            - any of the above with consumer surplus weights (maybe)
        Parameters
        ----------
        products : `RecArray`
            product_data used for pytBLP demand estimation
        demand_results : `Mapping`
            results structure from pyBLP demand estimation
        model_downstream: Array
            Can be one of ['bertrand', 'cournot', 'monopoly', 'perfect_competition']. If model_upstream not specified,
            this is a model without vertical integration.
        ownership_downstream: Array
            (optional, default is standard ownership) ownership matrix for price or quantity setting
        model_upstream: Optional[Array]
            Can be one of ['non'' (default), bertrand', 'cournot', 'monopoly'].  Upstream firm's model.
        ownership_upstream: Optional[Array]
            (optional, default is standard ownership) ownership matrix for price or quantity setting of upstream firms
        vertical_integration: Optional[Array]
            (optional, default is no vertical integration) vector indicating which product_ids are vertically integrated
            (ie store brands)
        Returns
        -------
        `ndarray`
            The built matrix.
        Notes
        _____
        For models without vertical integration, firm_ids must be defined in product_data for vi models, and
        firm_ids_upstream and firm_ids (=firm_ids_downstream) must be defined.
    """
    # TODO: add error if model is other custom model and custom markup can't be None

    # initialize
    N = np.size(products.prices)
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        ds_dp = demand_results.compute_demand_jacobians()
    number_models = len(model_downstream)
    markets = np.unique(products.market_ids)

    # TODO: is there a better way to initialize these?
    # initialize markups
    markups = [None] * number_models
    markups_upstream = [None] * number_models
    markups_downstream = [None] * number_models
    for i in range(number_models):
        markups_downstream[i] = np.zeros((N, 1), dtype=options.dtype)
        markups_upstream[i] = np.zeros((N, 1), dtype=options.dtype)


    # compute markups market-by-market
    for i in range(number_models):
        if user_supplied_markups[i] is not None:
            markups[i] = user_supplied_markups[i]
            markups_downstream[i] = user_supplied_markups[i]
        else:
            for t in markets:
                # TODO: all of these are giving me iteritems is deprecated warning
                index_t = np.where(demand_results.problem.products['market_ids'] == t)[0]
                shares_t = products.shares[index_t]
                retailer_response_matrix = ds_dp[index_t]
                retailer_response_matrix = retailer_response_matrix[:, ~np.isnan(retailer_response_matrix).all(axis=0)]

                if not (model_upstream[i] is None):
                    with contextlib.redirect_stdout(open(os.devnull, 'w')):
                        d2s_dp2_t = demand_results.compute_demand_hessians(market_id=t)


                # compute downstream markups for model i market t
                markups_downstream[i], retailer_ownership_matrix = compute_markups(
                    index_t, model_downstream[i], ownership_downstream[i], retailer_response_matrix, shares_t,
                    markups_downstream[i], custom_model_specification[i], markup_type='downstream'
                )
                markups_t = markups_downstream[i][index_t]

                # compute upstream markups (if applicable) following formula in Villas-Boas (2007)
                if not (model_upstream[i] is None):

                    # construct the matrix of derivatives with respect to prices for other manufacturers
                    J = len(shares_t)
                    g = np.zeros((J, J))
                    for j in range(J):
                        g[j] = np.transpose(markups_t) @ (retailer_ownership_matrix * d2s_dp2_t[:, j, :])

                    # solve for derivatives of all prices with respect to the wholesale prices
                    H = np.transpose(retailer_ownership_matrix * retailer_response_matrix)
                    G = retailer_response_matrix + H + g
                    delta_p = inv(G) @ H

                    # solve for matrix of cross-price elasticities of derived demand and the effects of cost
                    #   pass-through
                    manufacturer_response_matrix = np.transpose(delta_p) @ retailer_response_matrix

                    # compute upstream markups
                    markups_upstream[i], manufacturer_ownership_matrix = compute_markups(
                        index_t, model_upstream[i], ownership_upstream[i], manufacturer_response_matrix, shares_t,
                        markups_upstream[i], custom_model_specification[i], markup_type='upstream'
                    )

    # compute total markups as sum of upstream and downstream markups, taking into account vertical integration
    for i in range(number_models):
        if user_supplied_markups[i] is None:
            if vertical_integration[i] is None:
                vi = np.ones((N, 1))
            else:
                vi = (vertical_integration[i] - 1) ** 2
            markups[i] = markups_downstream[i] + vi * markups_upstream[i]

    return markups, markups_downstream, markups_upstream


def compute_markups(
        index, model_type, type_ownership_matrix, response_matrix, shares, markups, custom_model_specification,
        markup_type):
    """ Compute markups for some standard models including Bertrand, Cournot, and Monopoly. Allow user to pass in their
    own markup function as well.
    """
    if (markup_type == 'downstream') or (markup_type == 'upstream' and model_type is not None):

        # construct ownership matrix
        ownership_matrix = type_ownership_matrix[index]
        ownership_matrix = ownership_matrix[:, ~np.isnan(ownership_matrix).all(axis=0)]

        # compute markups based on specified model
        if model_type == 'bertrand':
            markups[index] = -inv(ownership_matrix * response_matrix) @ shares
        elif model_type == 'cournot':
            markups[index] = -(ownership_matrix * inv(response_matrix)) @ shares
        elif model_type == 'monopoly':
            markups[index] = -inv(response_matrix) @ shares
        elif model_type == 'perfect_competition':
            markups[index] = np.zeros((len(shares), 1))
        else:
            if custom_model_specification is not None:
                custom_model, custom_model_formula = next(iter(custom_model_specification.items()))
                markups[index] = eval(custom_model_formula)
                model_type = custom_model  # TODO: have custom model name in table output

    return markups, ownership_matrix
