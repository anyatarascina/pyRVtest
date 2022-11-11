"""Data construction."""

import contextlib
import os
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Union

import numpy as np
from numpy.linalg import inv

from . import exceptions, options
from .configurations.formulation import Formulation
from .utilities.algebra import precisely_invert
from .utilities.basics import Array, Groups, RecArray, extract_matrix, interact_ids, get_indices


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


def build_blp_instruments(formulation: Formulation, product_data: Mapping) -> Array:
    r"""Construct "sums of characteristics" excluded BLP instruments.

    Traditional "sums of characteristics" BLP instruments are

    .. math:: Z^\text{BLP}(X) = [Z^\text{BLP,Other}(X), Z^\text{BLP,Rival}(X)],

    in which :math:`X` is a matrix of product characteristics, :math:`Z^\text{BLP,Other}(X)` is a second matrix that
    consists of sums over characteristics of non-rival goods, and :math:`Z^\text{BLP,Rival}(X)` is a third matrix that
    consists of sums over rival goods. All three matrices have the same dimensions.

    .. note::

       To construct simpler, firm-agnostic instruments that are sums over characteristics of other goods, specify a
       constant column of firm IDs and keep only the first half of the instrument columns.

    Let :math:`x_{jt}` be the vector of characteristics in :math:`X` for product :math:`j` in market :math:`t`, which is
    produced by firm :math:`f`. That is, :math:`j \in J_{ft}`. Then,

    .. math::

       Z_{jt}^\text{BLP,Other}(X) = \sum_{k \in J_{ft} \setminus \{j\}} x_{kt}, \\
       Z_{jt}^\text{BLP,Rival}(X) = \sum_{k \notin J_{ft}} x_{kt}.

    .. note::

       Usually, any supply or demand shifters are added to these excluded instruments, depending on whether they are
       meant to be used for demand- or supply-side estimation.

    Parameters
    ----------
    formulation : `Formulation`
        :class:`Formulation` configuration for :math:`X`, the matrix of product characteristics used to build excluded
        instruments. Variable names should correspond to fields in ``product_data``.
    product_data : `structured array-like`
        Each row corresponds to a product. Markets can have differing numbers of products. The following fields are
        required:

            - **market_ids** : (`object`) - IDs that associate products with markets.

            - **firm_ids** : (`object`) - IDs that associate products with firms.

        Along with ``market_ids`` and ``firm_ids``, the names of any additional fields can be used as variables in
        ``formulation``.

    Returns
    -------
    `ndarray`
        Traditional "sums of characteristics" BLP instruments, :math:`Z^\text{BLP}(X)`.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/build_blp_instruments.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """

    # load IDs
    market_ids = extract_matrix(product_data, 'market_ids')
    firm_ids = extract_matrix(product_data, 'firm_ids')
    if market_ids is None or firm_ids is None:
        raise KeyError("product_data must have market_ids and firm_ids fields.")
    if market_ids.shape[1] > 1:
        raise ValueError("The market_ids field of product_data must be one-dimensional.")
    if firm_ids.shape[1] > 1:
        raise ValueError("The firm_ids field of product_data must be one-dimensional.")

    # initialize grouping objects
    market_groups = Groups(market_ids)
    paired_groups = Groups(interact_ids(market_ids, firm_ids))

    # build the instruments
    X = build_matrix(formulation, product_data)
    other = paired_groups.expand(paired_groups.sum(X)) - X
    rival = market_groups.expand(market_groups.sum(X)) - X - other
    return np.ascontiguousarray(np.c_[other, rival])


def build_differentiation_instruments(
        formulation: Formulation, product_data: Mapping, version: str = 'local', interact: bool = False) -> Array:
    r"""Construct excluded differentiation instruments.

    Differentiation instruments in the spirit of :ref:`references:Gandhi and Houde (2017)` are

    .. math:: Z^\text{Diff}(X) = [Z^\text{Diff,Other}(X), Z^\text{Diff,Rival}(X)],

    in which :math:`X` is a matrix of product characteristics, :math:`Z^\text{Diff,Other}(X)` is a second matrix that
    consists of sums over functions of differences between non-rival goods, and :math:`Z^\text{Diff,Rival}(X)` is a
    third matrix that consists of sums over rival goods. Without optional interaction terms, all three matrices have the
    same dimensions.

    .. note::

       To construct simpler, firm-agnostic instruments that are sums over functions of differences between all different
       goods, specify a constant column of firm IDs and keep only the first half of the instrument columns.

    Let :math:`x_{jt\ell}` be characteristic :math:`\ell` in :math:`X` for product :math:`j` in market :math:`t`, which
    is produced by firm :math:`f`. That is, :math:`j \in J_{ft}`. Then in the "local" version of
    :math:`Z^\text{Diff}(X)`,

    .. math::
       :label: local_instruments

       Z_{jt\ell}^\text{Local,Other}(X) =
       \sum_{k \in J_{ft} \setminus \{j\}} 1(|d_{jkt\ell}| < \text{SD}_\ell), \\
       Z_{jt\ell}^\text{Local,Rival}(X) =
       \sum_{k \notin J_{ft}} 1(|d_{jkt\ell}| < \text{SD}_\ell),

    where :math:`d_{jkt\ell} = x_{kt\ell} - x_{jt\ell}` is the difference between products :math:`j` and :math:`k` in
    terms of characteristic :math:`\ell`, :math:`\text{SD}_\ell` is the standard deviation of these pairwise differences
    computed across all markets, and :math:`1(|d_{jkt\ell}| < \text{SD}_\ell)` indicates that products :math:`j` and
    :math:`k` are close to each other in terms of characteristic :math:`\ell`.

    The intuition behind this "local" version is that demand for products is often most influenced by a small number of
    other goods that are very similar. For the "quadratic" version of :math:`Z^\text{Diff}(X)`, which uses a more
    continuous measure of the distance between goods,

    .. math::
       :label: quadratic_instruments

       Z_{jtk}^\text{Quad,Other}(X) = \sum_{k \in J_{ft} \setminus\{j\}} d_{jkt\ell}^2, \\
       Z_{jtk}^\text{Quad,Rival}(X) = \sum_{k \notin J_{ft}} d_{jkt\ell}^2.

    With interaction terms, which reflect covariances between different characteristics, the summands for the "local"
    versions are :math:`1(|d_{jkt\ell}| < \text{SD}_\ell) \times d_{jkt\ell'}` for all characteristics :math:`\ell'`,
    and the summands for the "quadratic" versions are :math:`d_{jkt\ell} \times d_{jkt\ell'}` for all
    :math:`\ell' \geq \ell`.

    .. note::

       Usually, any supply or demand shifters are added to these excluded instruments, depending on whether they are
       meant to be used for demand- or supply-side estimation.

    Parameters
    ----------
    formulation : `Formulation`
        :class:`Formulation` configuration for :math:`X`, the matrix of product characteristics used to build excluded
        instruments. Variable names should correspond to fields in ``product_data``.
    product_data : `structured array-like`
        Each row corresponds to a product. Markets can have differing numbers of products. The following fields are
        required:

            - **market_ids** : (`object`) - IDs that associate products with markets.

            - **firm_ids** : (`object`) - IDs that associate products with firms.

        Along with ``market_ids`` and ``firm_ids``, the names of any additional fields can be used as variables in
        ``formulation``.

    version : `str, optional`
        The version of differentiation instruments to construct:

            - ``'local'`` (default) - Construct the instruments in :eq:`local_instruments` that consider only the
              characteristics of "close" products in each market.

            - ``'quadratic'`` - Construct the more continuous instruments in :eq:`quadratic_instruments` that consider
              all products in each market.

    interact : `bool, optional`
        Whether to include interaction terms between different product characteristics, which can help capture
        covariances between product characteristics.

    Returns
    -------
    `ndarray`
        Excluded differentiation instruments, :math:`Z^\text{Diff}(X)`.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/build_differentiation_instruments.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """

    # load ids
    market_ids = extract_matrix(product_data, 'market_ids')
    firm_ids = extract_matrix(product_data, 'firm_ids')
    if market_ids is None or firm_ids is None:
        raise KeyError("product_data must have market_ids and firm_ids fields.")
    if market_ids.shape[1] > 1:
        raise ValueError("The market_ids field of product_data must be one-dimensional.")
    if firm_ids.shape[1] > 1:
        raise ValueError("The firm_ids field of product_data must be one-dimensional.")

    # identify markets
    market_indices = get_indices(market_ids)

    # build the matrix and count its dimensions
    X = build_matrix(formulation, product_data)
    N, K = X.shape

    # for the local version, do a first pass to compute standard deviations of pairwise differences across all markets
    sd_mapping: Dict[int, Array] = {}
    if version == 'local':
        for k in range(K):
            distances_count = distances_sum = squared_distances_sum = 0
            for t, indices_t in market_indices.items():
                x = X[indices_t][:, [k]]
                distances = x - x.T
                np.fill_diagonal(distances, 0)
                distances_count += distances.size - x.size
                distances_sum += np.sum(distances)
                squared_distances_sum += np.sum(distances**2)
            sd_mapping[k] = np.sqrt(squared_distances_sum / distances_count - (distances_sum / distances_count)**2)

    # build instruments market-by-market to conserve memory
    other_blocks: List[List[Array]] = []
    rival_blocks: List[List[Array]] = []
    for t, indices_t in market_indices.items():
        # build distance matrices for all characteristics
        distances_mapping: Dict[int, Array] = {}
        for k in range(K):
            x = X[indices_t][:, [k]]
            distances_mapping[k] = x - x.T
            np.fill_diagonal(distances_mapping[k], 0 if version == 'quadratic' else np.inf)

        def generate_instrument_terms() -> Iterator[Array]:
            """Generate terms that will be summed to create instruments."""
            for k1 in range(K):
                if version == 'quadratic':
                    for k2 in range(k1, K if interact else k1 + 1):
                        yield distances_mapping[k1] * distances_mapping[k2]
                elif version == 'local':
                    with np.errstate(invalid='ignore'):
                        close = (np.abs(distances_mapping[k1]) < sd_mapping[k1]).astype(np.float64)
                    if not interact:
                        yield close
                    else:
                        for k2 in range(K):
                            yield close * np.nan_to_num(distances_mapping[k2])
                else:
                    raise ValueError("version must be 'local' or 'quadratic'.")

        # append instrument blocks
        other_blocks.append([])
        rival_blocks.append([])
        ownership = (firm_ids[indices_t] == firm_ids[indices_t].T).astype(np.float64)
        nonownership = 1 - ownership
        for term in generate_instrument_terms():
            other_blocks[-1].append((ownership * term).sum(axis=1, keepdims=True))
            rival_blocks[-1].append((nonownership * term).sum(axis=1, keepdims=True))

    return np.c_[np.block(other_blocks), np.block(rival_blocks)]


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


def build_markups_all(
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
            Can be one of ['bertrand', 'cournot', 'monopoly']. If model_upstream not specified, this is a model without
            vertical integration.
        ownership_downstream: Array
            (optional, default is standard ownership) ownership matrix for price or quantity setting
        model_upstream: Optional[Array]
            Can be one of ['non'' (default), bertrand', 'cournot', 'monopoly'].  Upstream firm's model.
        ownership_upstream: Optional[Array]
            (optional, default is standard ownership) ownership matrix for price or quantity setting of upstream firms
        vertical_integration: Optional[Array]
        TODO: ask about this variable
        store_prod_ids:
            vector indicating which product_ids are vertically integrated (ie store brands). Default is
        missing and no vertical integration.
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
    N = np.size(products.prices)  # 2256
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        ds_dp = demand_results.compute_demand_jacobians()  # (2256, 24)
    number_models = len(model_downstream)
    markets = np.unique(products.market_ids)  # (94,)

    # TODO: is there a better way to initialize these?
    # initialize markups
    markups = [None] * number_models
    markups_upstream = [None] * number_models
    markups_downstream = [None] * number_models
    for i in range(number_models):
        markups_downstream[i] = np.zeros((N, 1), dtype=options.dtype)  # (2256, 1)
        markups_upstream[i] = np.zeros((N, 1), dtype=options.dtype)

    # if there is an upstream mode, get demand hessians
    if model_upstream is not None:
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            d2s_dp2 = demand_results.compute_demand_hessians()  # (2256, 24, 24)

    # compute markups market-by-market
    for i in range(number_models):
        if user_supplied_markups[i] is not None:
            markups[i] = user_supplied_markups[i]
            markups_downstream[i] = user_supplied_markups[i]
        else:
            for t in markets:
                # TODO: all of these are giving me iteritems is depracated warning
                index_t = np.where(demand_results.problem.products['market_ids'] == t)[0]  # (24, )
                shares_t = products.shares[index_t]  # (24, )
                retailer_response_matrix = ds_dp[index_t]  # (24, 24)
                # TODO: check this one
                retailer_response_matrix = retailer_response_matrix[:, ~np.isnan(retailer_response_matrix).all(axis=0)]  # (24, 24) - since nothing empty

                # compute downstream markups for model i market t
                markups_downstream[i], retailer_ownership_matrix = compute_markups(
                    index_t, model_downstream[i], ownership_downstream[i], retailer_response_matrix, shares_t,
                    markups_downstream[i], custom_model_specification[i], markup_type='downstream'
                )
                markups_t = markups_downstream[i][index_t]

                # compute upstream markups (if applicable) following formula in Villas-Boas (2007)
                if not (model_upstream[i] is None):
                    d2s_dp2_t = d2s_dp2[index_t]  # (24, 24, 24)
                    d2s_dp2_t = d2s_dp2_t[~np.isnan(d2s_dp2_t).any(axis=2)]  # TODO: why is this reshaped?
                    d2s_dp2_t = d2s_dp2_t.reshape(d2s_dp2_t.shape[1], d2s_dp2_t.shape[1], d2s_dp2_t.shape[1])  # TODO: this needs to be done better
                    J = len(shares_t)

                    # construct the matrix of derivatives with respect to prices for other manufacturers
                    g = np.zeros((J, J))
                    for j in range(J):
                        g[j] = np.transpose(markups_t) @ (retailer_ownership_matrix * d2s_dp2_t[:, j, :])

                    # solve for derivatives of all prices with respect to the wholesale prices
                    H = np.transpose(retailer_ownership_matrix * retailer_response_matrix)
                    G = retailer_response_matrix + H + g
                    delta_p = inv(G) @ H

                    # solve for matrix of cross-price elasticities of derived demand and the effects of cost pass-through
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
        markup_type
):
    """Compute markups for some standard models including Bertrand, Cournot, and Monopoly. Allow user to pass in their
    own markup function as well.
    """
    if (markup_type == 'downstream') or (markup_type == 'upstream' and model_type is not None):

        # pull out custom model information
        if custom_model_specification is not None:
            custom_model, custom_model_formula = next(iter(custom_model_specification.items()))

        # construct ownership matrix
        ownership_matrix = type_ownership_matrix[index]
        ownership_matrix = ownership_matrix[:, ~np.isnan(ownership_matrix).all(axis=0)]

        # maps each model to its corresponding markup formula, with option for own formula
        model_markup_formula = {
            'bertrand': -inv(ownership_matrix * response_matrix) @ shares,
            'cournot': -(ownership_matrix * inv(response_matrix)) @ shares,
            'monopoly': -inv(response_matrix) @ shares,
            'perfect_competition': np.zeros((len(shares), 1))
        }

        # compute markup for desired model
        if custom_model_specification is not None:
            model_markup_formula[custom_model] = eval(custom_model_formula)
            model_type = custom_model
        markups[index] = model_markup_formula[model_type]

    return markups, ownership_matrix
