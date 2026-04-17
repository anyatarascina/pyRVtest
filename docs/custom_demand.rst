Custom demand systems with ``UserSuppliedBackend``
===================================================

When to use ``UserSuppliedBackend``
-----------------------------------

If your demand system isn't logit / nested logit / BLP, or if you estimated it
outside ``pyblp`` (for example a labor-supply system, a discrete-choice model
with a non-logit link function, or a reduced-form share regression), you can
still run conduct tests by wrapping your own demand Jacobian in
:class:`pyRVtest.backends.UserSuppliedBackend`. The backend presents the same
interface pyRVtest uses internally for logit / nested logit / BLP, so
downstream markup computation code works unchanged.

What you need
-------------

At minimum you supply:

* **A stacked demand Jacobian.** Shape ``(N, J_max)``, NaN-padded so that
  market ``t`` occupies rows ``idx_t`` and columns ``0 .. J_t - 1`` with the
  ``J_t x J_t`` matrix :math:`\partial s / \partial p` for that market.
  Remaining columns in the block are NaN. This is the same layout
  ``pyblp.ProblemResults.compute_demand_jacobians()`` produces, so if you
  are partially using ``pyblp`` you can reuse its output.

* **Per-product market ids.** A length-``N`` array aligning each Jacobian row
  to a market. Required for the per-market slicing that
  ``compute_jacobian(market_id=...)`` performs.

Optional:

* **A per-market Hessian callable** ``hessian_fn(market_id) ->
  (J_t, J_t, J_t) ndarray``. Required only for vertical-integration models
  (upstream markups via Villas-Boas (2007) passthrough).

* **Parameter names and a perturb callback.** If you want to run finite-
  difference demand-adjustment corrections, supply ``theta_names=[...]`` and
  ``perturb_callback=lambda idx, delta: <new backend>``. Without these,
  :meth:`UserSuppliedBackend.perturbed` raises ``NotImplementedError``.

Limitations
-----------

``UserSuppliedBackend`` implements the core ``DemandBackend`` protocol
(``compute_jacobian``, ``compute_hessian``, ``perturbed``, ``n_parameters``,
``theta_names``) but **does not** implement ``SupportsDemandAdjustment``.
Consequences:

* The first-stage demand-adjustment correction
  (``demand_adjustment=True`` on :meth:`pyRVtest.Problem.solve`) is not
  available out of the box. If you need it, supply your own
  ``SupportsDemandAdjustment`` implementation.

* Vertical models (``model_upstream`` set, non-zero vertical_integration) raise
  a clear error unless you pass ``hessian_fn``.

* The public :class:`pyRVtest.Problem` constructor currently wires the demand
  backend from either ``demand_results=`` (a ``pyblp.ProblemResults``) or
  ``demand_params=`` (a dict). A pre-constructed backend cannot be passed
  through ``Problem`` directly in v0.4. The example below uses the
  lower-level :func:`pyRVtest.markups._compute_markups` entry point instead,
  which accepts ``demand_backend=`` and returns the per-model markup arrays
  you would otherwise hand to ``Problem(markup_data=...)``. A future release
  will add ``Problem(demand_backend=...)`` so the high-level API is fully
  connected.

Worked example
--------------

We demonstrate with a stylized linear demand system. A product's share in
market ``t`` is

.. math::

    s_{jt} = a_j - b\, p_{jt} + \frac{c}{J_t - 1} \sum_{k \ne j} p_{kt},

so the demand Jacobian is

.. math::

    \frac{\partial s_j}{\partial p_k}
    = \begin{cases} -b & k = j \\ c / (J_t - 1) & k \ne j. \end{cases}

The system is not a logit — own-price responses are linear in price rather
than quadratic in shares — and the cross-price structure is chosen by fiat.
The pattern (write the Jacobian, wrap in ``UserSuppliedBackend``, feed into
markup computation) is the same one a researcher with a real custom demand
system would follow; just swap in your own Jacobian code.

We simulate ``T = 30`` markets with ``J = 3`` single-product firms each.
Prices are set to the Bertrand-Nash equilibrium under the true DGP, so
Bertrand is the "correct" conduct model and perfect competition is not.

.. code-block:: python

    import numpy as np
    import pandas as pd
    import pyRVtest
    from pyRVtest.backends import UserSuppliedBackend
    from pyRVtest.markups import _compute_markups
    from pyRVtest.problem import Models

    rng = np.random.default_rng(seed=42)
    T, J = 30, 3
    N = T * J

    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)

    # Demand parameters.
    b, c = 0.8, 0.15
    a_j = np.array([1.8, 1.6, 1.4])

    def demand_jacobian_block(J_t):
        D = np.full((J_t, J_t), c / (J_t - 1))
        np.fill_diagonal(D, -b)
        return D

    # Simulate Bertrand-Nash prices and shares (true DGP).
    mc = 0.5 + 0.1 * rng.normal(size=N)
    prices = np.empty(N)
    shares = np.empty(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        p_t = mc[idx] + 0.5
        for _ in range(5000):
            s_t = a_j - b * p_t + (c / (J - 1)) * (p_t.sum() - p_t)
            p_new = mc[idx] + s_t / b    # single-product-firm Bertrand FOC
            if np.max(np.abs(p_new - p_t)) < 1e-14:
                break
            p_t = 0.5 * p_t + 0.5 * p_new
        s_t = a_j - b * p_t + (c / (J - 1)) * (p_t.sum() - p_t)
        prices[idx] = p_t
        shares[idx] = s_t

    # Rival-level cost shifter to use as a testing instrument.
    rival_mean_mc = np.zeros(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        for i in idx:
            rival_mean_mc[i] = np.mean([mc[k] for k in idx if k != i])

    product_data = pd.DataFrame({
        'market_ids': market_ids,
        'firm_ids': firm_ids,
        'prices': prices,
        'shares': shares,
        'w_cost': mc,
        'z_rival_mc': rival_mean_mc,
    }).to_records(index=False)

    # ---- Build the user-supplied Jacobian ----
    J_max = J
    jacobian = np.full((N, J_max), np.nan)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        J_t = len(idx)
        jacobian[idx[:, None], np.arange(J_t)[None, :]] = demand_jacobian_block(J_t)

    backend = UserSuppliedBackend(jacobian=jacobian, market_ids=market_ids)

    # ---- Compute markups for two candidate conduct models ----
    models_list = [
        pyRVtest.Bertrand(ownership='firm_ids'),
        pyRVtest.PerfectCompetition(),
    ]
    models = Models(models_list, product_data)

    markups, markups_down, markups_up = _compute_markups(
        product_data=product_data,
        pyblp_results=None,
        model_downstream=models['models_downstream'],
        ownership_downstream=models['ownership_downstream'],
        model_upstream=models['models_upstream'],
        ownership_upstream=models['ownership_upstream'],
        vertical_integration=models['vertical_integration'],
        custom_model_specification=models['custom_model_specification'],
        user_supplied_markups=models['user_supplied_markups'],
        mix_flag=models['mix_flag'],
        demand_backend=backend,
    )

    bertrand_markups = markups[0].flatten()
    pc_markups = markups[1].flatten()
    implied_mc = prices - bertrand_markups

    print(f"Bertrand markups: mean={bertrand_markups.mean():.4f}, "
          f"min={bertrand_markups.min():.4f}, max={bertrand_markups.max():.4f}")
    print(f"Perfect-competition markups should be 0: max|.|={np.abs(pc_markups).max():.2e}")
    print(f"max|implied_mc - true_mc| = {np.abs(implied_mc - mc).max():.2e}")

Expected output (values up to RNG)::

    Bertrand markups: mean=0.8789, min=0.7032, max=1.0820
    Perfect-competition markups should be 0: max|.|=0.00e+00
    max|implied_mc - true_mc| = 3.33e-15

Interpretation
~~~~~~~~~~~~~~

* The perfect-competition markup is identically zero by construction. This
  is the degenerate case: under PC, :math:`p = mc` so the implied marginal
  cost is :math:`p` itself.

* The Bertrand markup is positive for every product-market. Because we
  generated the data from Bertrand-Nash pricing with the same demand
  Jacobian, :math:`p - \text{markup}^{\text{Bertrand}}` recovers the
  true ``mc`` to machine precision. That is the "sanity check" an RV
  test would then sharpen statistically: with a cost-shifter instrument
  ``z_rival_mc``, the Bertrand model's residual cost regression should
  fit far better than the PC model's.

Running the full RV test would pass ``(markups, markups_down, markups_up)``
as ``markup_data=`` to :class:`pyRVtest.Problem` along with a cost formulation
and instrument formulation, then call ``.solve()``. That is the standard
pyRVtest pipeline; see :doc:`tutorial` for an end-to-end walkthrough. The
only thing a custom-demand workflow changes is the markup step — everything
downstream of that is identical.

Disclaimer
----------

This example is deliberately stylized. Real research problems will typically
have a more complex demand system (richer heterogeneity, non-linear price
responses, multiple parameters). The key pattern is the same:

1. Estimate / specify your demand system.
2. Compute the :math:`\partial s / \partial p` Jacobian in the stacked
   NaN-padded format.
3. Wrap in ``UserSuppliedBackend``.
4. Pass to :func:`pyRVtest.markups._compute_markups` (or, in a future
   release, to :class:`pyRVtest.Problem` directly).
