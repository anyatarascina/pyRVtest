"""Data construction."""

import contextlib
import os
from pathlib import Path
import pickle
from typing import Any, Callable, Mapping, Optional, Union

import numpy as np
from numpy.linalg import inv
import pyblp
from pyblp.utilities.basics import Array, RecArray

from . import options


def build_ownership(
        product_data: Mapping, firm_ids_column_name: str,
        kappa_specification: Optional[Union[str, Callable[[Any, Any], float]]] = None) -> Array:
    r"""Build ownership matrices, :math:`O`.

    Thin wrapper around :func:`pyblp.build_ownership` that allows specifying which column contains firm IDs, rather
    than requiring a column named ``firm_ids``. This supports separate downstream and upstream ownership structures.

    Parameters
    ----------
    product_data : `structured array-like`
        Each row corresponds to a product. Must have a ``market_ids`` field.
    firm_ids_column_name : `str`
        Column in ``product_data`` with firm IDs. Ignored for special-case ``kappa_specification`` values.
    kappa_specification : `str or callable, optional`
        Passed directly to :func:`pyblp.build_ownership`. See that function for full documentation.

    Returns
    -------
    `ndarray`
        Stacked :math:`J_t \times J_t` ownership matrices for each market :math:`t`.

    """
    names = product_data.dtype.names if hasattr(product_data, 'dtype') and product_data.dtype.names else product_data.columns
    modified_data = {name: product_data[name] for name in names}
    if firm_ids_column_name is not None:
        modified_data['firm_ids'] = product_data[firm_ids_column_name]
    return pyblp.build_ownership(modified_data, kappa_specification)


def build_markups(
        model_formulations: Any, product_data: RecArray, pyblp_results: Mapping) -> Array:
    r"""Compute markups for one or more candidate conduct models.

    This is the main user-facing markup function. It accepts :class:`ModelFormulation` objects together with product
    data and a PyBLP demand results object, and returns implied markups for each model.

    Supported models include standard Bertrand, Cournot, monopoly, perfect competition, mixed Cournot/Bertrand,
    bilateral oligopoly (with optional vertical integration), and custom user-specified markup formulas.

    Parameters
    ----------
    model_formulations : `sequence of ModelFormulation`
        One or more :class:`ModelFormulation` instances that specify the conduct model(s) for which markups should be
        computed. At least one formulation must be provided.
    product_data : `structured array-like`
        Product-level data used for demand estimation. Must contain ``market_ids``, ``shares``, and ``prices``
        fields.
    pyblp_results : `structured array-like`
        Results object returned by ``pyblp.Problem.solve``. Used to compute demand Jacobians and Hessians.

    Returns
    -------
    `tuple[list, list, list]`
        Computed total markups, downstream markups, and upstream markups for each model. Each element of the
        returned lists is an ``(N, 1)`` array of markups stacked across markets.

    Examples
    --------
    Compute markups for a single Bertrand model::

        model = pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids')
        markups, markups_down, markups_up = pyRVtest.build_markups([model], product_data, pyblp_results)

    """
    from .primitives import Models  # local import avoids circular dependency
    if not hasattr(model_formulations, '__len__'):
        model_formulations = [model_formulations]
    models = Models(model_formulations, product_data)
    return _compute_markups(
        product_data, pyblp_results, models["models_downstream"], models["ownership_downstream"],
        models["models_upstream"], models["ownership_upstream"], models["vertical_integration"],
        models["custom_model_specification"], models["user_supplied_markups"], models["mix_flag"]
    )


def _compute_markups(
        product_data: RecArray, pyblp_results: Mapping, model_downstream: Optional[Array],
        ownership_downstream: Optional[Array], model_upstream: Optional[Array] = None,
        ownership_upstream: Optional[Array] = None, vertical_integration: Optional[Array] = None,
        custom_model_specification: Optional[dict] = None, user_supplied_markups: Optional[Array] = None,
        mix_flag: Optional[Array] = None) -> Array:
    r"""Compute markups given pre-processed model arrays.

    Internal function called by :func:`build_markups` and :meth:`Problem.solve`. Accepts the raw arrays produced by
    :class:`Models` rather than :class:`ModelFormulation` objects.

    Parameters
    ----------
    product_data : `recarray`
        The `product_data` containing information on markets and product characteristics. This should be the same as
        the data used for demand estimation. To compute markups, this data must include `prices`, `market_ids`, and
        `shares`.
    pyblp_results : `structured array-like`
        The results object obtained from using the pyBLP demand estimation procedure. We use built-in PyBLP
        functions to return the demand Jacobians and Hessians (first and second derivatives of shares with respect
        to prices).
    model_downstream: `ndarray`
        The model of conduct for downstream firms. Can be one of [`bertrand`, `cournot`, `monopoly`,
        `perfect_competition`, `other`]. Only specify option `other` if supplying a custom markup formula.
    ownership_downstream: `ndarray`
        The ownership matrix for price or quantity setting (optional, default is standard ownership).
    model_upstream: `ndarray, optional`
        Upstream firm model of conduct. Only specify option `other` if supplying a custom markup formula. Can be one
        of ['none' (default), `bertrand`, `cournot`, `monopoly`, `perfect_competition`, `other`].
    ownership_upstream: `ndarray, optional`
        Ownership matrix for price or quantity setting of upstream firms (optional, default is None).
    vertical_integration: `ndarray, optional`
        Vector indicating which `product_ids` are vertically integrated (ie store brands) (optional, default is
        None).
    custom_model_specification: `dict, optional`
        Dictionary containing a custom markup formula and the name of the formula (optional, default is None).
    user_supplied_markups: `ndarray, optional`
        Vector containing user-computed markups (optional, default is None). If user supplied own markups, this
        function simply returns them.

    Returns
    -------
    `tuple[list, list, list]`
        . Computed markups, downstream markups, and upstream markups for each model.

    Notes
    _____
    For models without vertical integration, firm_ids must be defined in product_data for vi models, and
    firm_ids_upstream and firm_ids (=firm_ids_downstream) must be defined.

    """

    # initialize market characteristics
    N = product_data.shape[0]
    number_models = len(model_downstream)
    markets = np.unique(product_data.market_ids)

    # initialize markups
    markups = [None] * number_models
    markups_upstream = [None] * number_models
    markups_downstream = [None] * number_models
    for i in range(number_models):
        markups_downstream[i] = np.zeros((N, 1), dtype=options.dtype)
        markups_upstream[i] = np.zeros((N, 1), dtype=options.dtype)
        
    # Transform absent input into list
    if user_supplied_markups is None:
        user_supplied_markups = [None] * number_models
    if custom_model_specification is None:
        custom_model_specification = [None] * number_models
    if model_upstream is None:
        model_upstream=[None]*number_models
    if vertical_integration is None:
        vertical_integration = [None] * number_models
    if mix_flag is None:
        mix_flag = [None] * number_models
    
    # precompute demand jacobians
    if pyblp_results is not None:
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            ds_dp = pyblp_results.compute_demand_jacobians()

    # compute markups market-by-market
    for i in range(number_models):
        if user_supplied_markups[i] is not None:
            markups[i] = user_supplied_markups[i]
            markups_downstream[i] = user_supplied_markups[i]
        else:
            for t in markets:
                index_t = np.where(product_data.market_ids == t)[0]
                shares_t = product_data.shares[index_t]
                retailer_response_matrix = ds_dp[index_t]
                retailer_response_matrix = retailer_response_matrix[:, ~np.isnan(retailer_response_matrix).all(axis=0)]

                # compute downstream markups for model i market t
                markups_downstream[i], retailer_ownership_matrix = evaluate_first_order_conditions(
                    index_t, model_downstream[i], ownership_downstream[i], retailer_response_matrix, shares_t,
                    markups_downstream[i], custom_model_specification[i], markup_type='downstream', type_mix_flag=mix_flag[i])

                # compute upstream markups (if applicable) following formula in Villas-Boas (2007)
                if not (model_upstream[i] is None):

                    # construct the matrix of derivatives with respect to prices for other manufacturers
                    markups_t = markups_downstream[i][index_t]
                    passthrough_matrix = construct_passthrough_matrix(
                        pyblp_results, t, retailer_response_matrix, retailer_ownership_matrix, markups_t
                    )

                    # solve for matrix of cross-price elasticities of derived demand and the effects of cost
                    #   pass-through
                    manufacturer_response_matrix = np.transpose(passthrough_matrix) @ retailer_response_matrix

                    # compute upstream markups
                    markups_upstream[i], manufacturer_ownership_matrix = evaluate_first_order_conditions(
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


def construct_passthrough_matrix(
        pyblp_results, market_id, retailer_response_matrix, retailer_ownership_matrix, markups_t):
    """Construct the passthrough matrix using the formula from Villas-Boas (2007). This matrix contains the derivatives
    of all retail prices with respect to all wholesale prices."""

    # compute demand hessians
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        d2s_dp2_t = pyblp_results.compute_demand_hessians(market_id=market_id)

    # compute the product of demand hessians and markups
    J = len(markups_t)
    g = np.zeros((J, J))
    for j in range(J):
        g[:, [j]] = (retailer_ownership_matrix * d2s_dp2_t[:, :, j]) @ markups_t

    # solve for derivatives of all prices with respect to the wholesale prices
    H = np.transpose(retailer_ownership_matrix * retailer_response_matrix)
    G = retailer_response_matrix + H + g
    return inv(G) @ H


def evaluate_first_order_conditions(
        index, model_type, type_ownership_matrix, response_matrix, shares, markups, custom_model_specification,
        markup_type, type_mix_flag=None):
    """Compute markups for some standard models including Bertrand, Cournot, monopoly, and perfect competition using
    the first order conditions corresponding to each model. Allow user to pass in their own markup function as well.
    """
    if len(shares.shape)==1:
        shares=np.expand_dims(shares, axis=1)
    ownership_matrix = None
    if (markup_type == 'downstream') or (markup_type == 'upstream' and model_type is not None):

        # construct ownership matrix and mix_flag vector
        if type_ownership_matrix is not None:
            ownership_matrix = type_ownership_matrix[index]
            ownership_matrix = ownership_matrix[:, ~np.isnan(ownership_matrix).all(axis=0)]
        if type_mix_flag is not None:
          mix_flag=type_mix_flag[index]
          
        # compute markups based on specified model first order condition
        if model_type == 'bertrand':
            markups[index,:] = -inv(ownership_matrix * response_matrix) @ shares
        elif model_type == 'cournot':
            markups[index,:] = -(ownership_matrix * inv(response_matrix)) @ shares
        elif model_type == 'monopoly':
            markups[index,:] = -inv(response_matrix) @ shares
        elif model_type == 'perfect_competition':
            markups[index,:] = np.zeros((len(shares), 1))
        elif model_type == 'mix_cournot_bertrand':
            markups[index,:]=MixMkup(ownership_matrix,response_matrix,mix_flag,shares)
        else:
            if custom_model_specification is not None:
                custom_model, custom_model_formula = next(iter(custom_model_specification.items()))
                markups[index] = eval(custom_model_formula)

    return markups, ownership_matrix

def MixMkup(ownership_matrix,response_matrix,mix_flag,shares):
        sharesB=shares[mix_flag]
        sharesC=shares[~mix_flag]
        ownB = ownership_matrix[mix_flag,:][:,mix_flag]
        ownC = ownership_matrix[~mix_flag,:][:,~mix_flag]
        D_BB = response_matrix[:,mix_flag][mix_flag,:]
        D_CB = response_matrix[:,mix_flag][~mix_flag,:]
        D_CC = response_matrix[:,~mix_flag][~mix_flag,:]
        D_BC = response_matrix[:,~mix_flag][mix_flag,:]
        mkups_C = -(ownC * inv(D_CC)) @ sharesC
        mkups_B = -inv(ownB * (D_BC @ inv(D_CC) @ D_CB +D_BB)) @ sharesB
        mkups=np.zeros((len(mix_flag),1))
        mkups[mix_flag]=mkups_B
        mkups[~mix_flag]=mkups_C
        return(mkups)
      
def read_pickle(path: Union[str, Path]) -> object:
    """Load a pickled object into memory.
    This is a simple wrapper around `pickle.load`.

    Parameters
    ----------
    path : `str or Path`
        File path of a pickled object.
    Returns
    -------
    `object`
        The unpickled object.

    """
    with open(path, 'rb') as handle:
        return pickle.load(handle)
