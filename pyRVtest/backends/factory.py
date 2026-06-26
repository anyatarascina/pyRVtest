"""Demand-backend construction factory.

Hosts :func:`make_demand_backend`, the single routing point that turns raw
``product_data`` plus a fitted demand object (``demand_results`` from pyblp or a
``demand_params`` dict) into a concrete backend implementing the
``DemandBackend`` / ``SupportsDemandAdjustment`` protocols.

This logic previously lived inside ``Problem._construct_demand_backend`` and its
helpers. It was extracted here so the standalone precompute helpers
(:func:`pyRVtest.build_phi_matrix`, :func:`pyRVtest.build_markup_derivative`)
can build a backend without constructing a full :class:`~pyRVtest.Problem`
(which would require a ``cost_formulation`` and ``instrument_formulation`` that
the demand-side correction does not need). ``Problem._construct_demand_backend``
now delegates here, so there is one source of truth for the routing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

import numpy as np

from .. import options


__all__ = ['make_demand_backend']


def make_demand_backend(
        product_data: Any,
        *,
        demand_results: Any = None,
        demand_params: Optional[Dict[str, Any]] = None,
        demand_backend: Any = None,
) -> Any:
    """Build a demand backend from raw inputs, mirroring Problem routing.

    Returns ``None`` when none of ``demand_backend`` / ``demand_results`` /
    ``demand_params`` is supplied (the user-supplied-markups-only path).
    Otherwise, in priority order:

    - ``demand_backend is not None``: return the caller's pre-built backend.
    - ``demand_results is not None``: route to ``NestedLogitBackend`` when
      single-scalar-rho nested logit is detected (analytical ``d(D)/d(rho)`` is
      exact), else ``PyBLPBackend``.
    - ``demand_params is not None`` with nonzero rho -> ``NestedLogitBackend``;
      otherwise ``LogitBackend``.

    Parameters
    ----------
    product_data
        Raw product data (DataFrame / structured array / dict). The analytical
        backends read arbitrary columns (``nesting_ids``, ``x1``, instruments)
        directly from it.
    demand_results
        A fitted ``pyblp.ProblemResults`` object, or ``None``.
    demand_params
        A demand-parameter dict (``alpha`` required; optional ``rho``, ``beta``,
        ``x_columns``, ``demand_instrument_columns``, ``nesting_ids_columns``,
        ``W_demand``), or ``None``.
    demand_backend
        A pre-built backend object to return as-is, or ``None``.
    """
    if demand_backend is not None:
        return demand_backend
    if demand_results is not None:
        return _backend_from_demand_results(demand_results, product_data)

    if demand_params is not None:
        dp = demand_params
        rho_nonzero = [s for s in dp.get('rho', []) if s > 0]
        # Auto-augment product_data with a literal '1' column if any
        # x_columns / demand_instrument_columns reference it but the raw data
        # does not (mirrors _backend_from_demand_results for the
        # Formulation('1 + ...') path so the LogitEstimator output works
        # without users hand-adding an intercept column).
        raw = product_data
        referenced = list(dp.get('x_columns') or []) + list(
            dp.get('demand_instrument_columns') or []
        )
        if '1' in referenced and '1' not in _raw_product_data_columns(raw):
            raw = _augment_with_intercept_column(raw)
        shared_kwargs = dict(
            alpha=dp['alpha'],
            product_data=raw,
            beta=dp.get('beta'),
            x_columns=dp.get('x_columns'),
            demand_instrument_columns=dp.get('demand_instrument_columns'),
            W_demand=dp.get('W_demand'),
        )
        if rho_nonzero:
            from .nested_logit import NestedLogitBackend
            return NestedLogitBackend(
                rho=dp.get('rho', []),
                nesting_ids_columns=dp.get('nesting_ids_columns'),
                **shared_kwargs,
            )
        from .logit import LogitBackend
        return LogitBackend(**shared_kwargs)

    return None


def _backend_from_demand_results(r: Any, product_data: Any) -> Any:
    """Route pyblp ProblemResults to the best-matched backend.

    * **Plain logit** (``K2==0`` and ``r.rho.size==0``): try routing to
      ``LogitBackend``.
    * **Single-scalar-rho nested logit** (``K2==0`` and ``r.rho.size==1``): try
      routing to ``NestedLogitBackend(rho=[rho])`` (analytical ``d(D)/d(rho)``
      is exact; pyblp's finite-diff has genuine O(eps^2) truncation error).
    * **Per-nest rho** (``K2==0`` and ``r.rho.size>1``): stay on
      ``PyBLPBackend``.
    * **BLP** (``K2>0``): stay on ``PyBLPBackend``.

    Falls back to ``PyBLPBackend`` if the raw product_data doesn't carry the
    columns the analytical backend needs (e.g., pyblp-generated FE dummies).
    """
    from .pyblp import PyBLPBackend
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

    # pyblp's ZD dtype does not carry sub-column names as a 3-tuple; we need
    # names to feed `demand_instrument_columns`. Infer from the raw
    # product_data: `demand_instrumentsN` columns + non-price X1 columns,
    # matching pyblp's default ZD construction. If the inferred count does not
    # match the actual ZD width, the user passed a custom ZD formulation and we
    # should bail.
    raw = product_data
    raw_cols = _raw_product_data_columns(raw)
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
        raw = _augment_with_intercept_column(raw)
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
        from .nested_logit import NestedLogitBackend
        return NestedLogitBackend(
            rho=[float(rho_arr[0])],
            nesting_ids_columns=None,  # backend infers from product_data
            **shared_kwargs,
        )
    from .logit import LogitBackend
    return LogitBackend(**shared_kwargs)


def _raw_product_data_columns(raw: Any) -> Set[str]:
    """Return the column names of raw product_data regardless of container type.

    Handles a pandas DataFrame, a structured recarray, or a dict-like mapping.
    """
    if hasattr(raw, 'columns'):
        return set(map(str, raw.columns))
    if hasattr(raw, 'dtype') and raw.dtype.names:
        return set(raw.dtype.names)
    if hasattr(raw, 'keys'):
        return set(raw.keys())
    return set()


def _augment_with_intercept_column(raw: Any) -> Any:
    """Return a copy of raw product_data with a '1' column (all ones).

    pyblp's `Formulation('1 + ...')` packs a literal '1' column into X1/ZD.
    Users typically don't include a column literally named '1' in their
    product_data; synthesize it for the analytical-backend path. Never mutates
    the input.
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
        f"pyRVtest internal error: expected product_data to be a pandas "
        f"DataFrame, a structured numpy array, or a dict-like mapping for "
        f"intercept augmentation. "
        f"Received {type(raw).__name__}."
    )
