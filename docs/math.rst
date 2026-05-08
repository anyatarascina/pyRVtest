Mathematical reference
======================

A condensed reference for the core formulas pyRVtest computes. See the
methodology papers in :doc:`references` for full derivations and proofs.

Notation
--------

* :math:`j \in \{1, \ldots, J_t\}` indexes products in market :math:`t`.
* :math:`s_{jt}, p_{jt}` are observed share and price; :math:`s_{0t}`
  is the outside-good share.
* :math:`x_{jt}, w_{jt}` are observed demand-side characteristics and
  cost shifters.
* :math:`z_{jt}` denotes a testing instrument.
* :math:`\eta_{mjt}` is the markup implied by candidate conduct model
  :math:`m`; :math:`\eta_{0jt}` is the true markup.
* :math:`c_{mjt}` is the marginal cost implied by model :math:`m`,
  recovered from the FOC at the observed prices and shares.


First-order condition
---------------------

Each candidate conduct model implies a first-order condition that maps
prices, markups, and (if present) taxes and cost-scaling to marginal
cost. With unit tax :math:`\tau_t` and ad-valorem retention factor
:math:`\nu_t`, the FOC of :ref:`references: Dearing, Magnolfi, Quint, Sullivan, and Waldfogel (2026)`
equation (3) is

.. math::

   \nu_t \, p_{jt} - \tau_t = \nu_t \, \eta_{0jt} + c_{0jt}.

Rearranging for marginal cost:

.. math::

   c_{0jt} = \nu_t (p_{jt} - \eta_{0jt}) - \tau_t.

For a pure rule-of-thumb / cost-scaling model where firms price at
:math:`\phi` times marginal cost (so :math:`\eta_{mjt} = (\phi - 1) c_{mjt}`),
substitution yields

.. math::

   c_{mjt}
       = \frac{\nu_t \, p_{jt} - \tau_t}{1 + \nu_t \cdot \mathrm{cost\_scaling}_m},
   \quad \mathrm{cost\_scaling}_m = \phi - 1.

This is the formula pyRVtest uses internally; it is consistent with
DMQSW eq (3) at the no-cost-scaling baseline (``cost_scaling=0``)
where it reduces to :math:`c = \nu(p - \eta) - \tau`.


Rivers-Vuong test statistic
---------------------------

For each pair of candidate models :math:`(m, m')` and each instrument
set, define the GMM moments :math:`\bar g_m = (1/N) \sum_i z_i \omega_{mi}`
where :math:`\omega_{mi}` is the implied cost-side residual for model
:math:`m` at observation :math:`i`. The Rivers-Vuong statistic is

.. math::

   T_{RV}(m, m') = \sqrt{N} \cdot
       \frac{\bar g_{m}' W \bar g_{m} - \bar g_{m'}' W \bar g_{m'}}
            {\hat\sigma(m, m')},

where :math:`W` is the GMM weighting matrix and :math:`\hat\sigma(m, m')`
is the standard error of the moment-difference. Under the null of equal
fit, :math:`T_{RV}` is asymptotically standard normal. Pairwise
significance is read at :math:`|T_{RV}| > 1.64,\ 1.96,\ 2.58` for the
10%, 5%, 1% levels.

When demand parameters are estimated rather than fixed,
:math:`\hat\sigma(m, m')` includes a first-stage correction (DMSS
Appendix C; DMQSS Appendix B for the non-linear-cost case). The
``Problem.solve(demand_adjustment=True)`` flag activates this
correction. Clustering corrections multiply the variance estimate by
the standard cluster-robust factor.


F-statistic diagnostic
----------------------

The DMSS scaled F-statistic diagnoses whether the RV test itself has
acceptable size and power for the given instrument set and pair of
models. Per :ref:`references: Duarte, Magnolfi, Sølvsten, and Sullivan (2023)`,

.. math::

   F(m, m') = \frac{2 N}{K_{\text{eff}}} \cdot
       \frac{F_{\text{num}}(m, m')}{D_{\rho}(m, m')},

where :math:`K_{\text{eff}}` is the rank-adjusted instrument count
(equal to the raw :math:`K` unless ``endogenous_cost_component`` is
set, which residualizes one degree of freedom out of the instrument
set), and :math:`D_\rho` is the variance-difference denominator built
from per-model trace contractions
:math:`\hat\sigma_0, \hat\sigma_1, \hat\sigma_2` against the
demand-side weight matrix.

Critical values from the tables in :py:mod:`pyRVtest.data` are
indexed by :math:`K_{\text{eff}}` and the estimated
:math:`\hat\rho^2 = (\hat\sigma_0 - \hat\sigma_1)^2 / D_\rho`. The
package reports critical values for worst-case size in
:math:`\{0.075, 0.10, 0.125\}` and best-case power in
:math:`\{0.50, 0.75, 0.95\}`.

Numerical-fragility safeguard. When :math:`\hat\rho^2` is close to 1
(the Cauchy-Schwarz boundary), float64 loses precision in computing
:math:`D_\rho`. The package detects this case (see
``LAMBDA_PRECISION_THRESHOLD`` in
``pyRVtest/solve/test_engine.py``) and recomputes :math:`F` and
:math:`\hat\rho^2` using mpmath at higher precision. The footer of
the printed results table notes ``recomputed with extra precision``
when this happens.


Model Confidence Set
--------------------

The Hansen-Lunde-Nason MCS is computed by sequential elimination on
the matrix of pairwise RV statistics. Initialize the candidate set
:math:`\mathcal{M}_0` to all models. At each step :math:`k`:

1. Compute the test statistic
   :math:`T_k = \max_{m \in \mathcal{M}_k} t_m`,
   where :math:`t_m = \max_{m' \in \mathcal{M}_k} T_{RV}(m, m')`
   measures how badly model :math:`m` is dominated by the best
   competitor in the current set.
2. The MCS p-value of model :math:`m_k^* = \arg\max_m t_m` is
   :math:`P(T \geq T_k)` under the asymptotic distribution of the
   maximum.
3. Eliminate :math:`m_k^*` and repeat until one model remains.

The reported MCS p-value of model :math:`m` is the largest
:math:`\alpha` at which :math:`m` survives.


Villas-Boas passthrough matrix
------------------------------

For models in the :class:`pyRVtest.Vertical` family (downstream firms
plus an upstream supplier), pyRVtest computes upstream markups using
the :ref:`references: Villas-Boas (2007)` passthrough formula. Let

.. math::

   \mathcal{P}_t = \frac{\partial p_t}{\partial w_t}

be the matrix of partial derivatives of equilibrium prices with
respect to a marginal-cost shifter. Under Bertrand-Nash downstream
conduct,

.. math::

   \mathcal{P}_t = (I - \Omega_t H_t)^{-1} \Omega_t H_t,

where :math:`\Omega_t` is the ownership matrix and :math:`H_t` is the
matrix of demand derivatives. The implementation is in
:py:func:`pyRVtest.construct_passthrough_matrix`. The diagnostic
version exposed on :class:`~pyRVtest.ProblemResults`
(:meth:`~pyRVtest.ProblemResults.passthrough_matrix`) returns
:math:`\mathcal{P}_t` per (model, market) for inspection.


Pass-through by conduct class
-----------------------------

For diagnostic purposes (DMQSW pass-through framework, instrument
relevance, channel decomposition), pyRVtest computes the per-candidate
pass-through matrix
:math:`P_m = (I - \partial \Delta_m / \partial p)^{-1}`
*numerically* via central-difference perturbation of prices through
the candidate's first-order condition; see
:py:func:`pyRVtest.solve.passthrough.compute_passthrough_numerical`.

Per-conduct analytical references
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The numerics converge to the closed-form expressions below. We list
them for reference; the package does not evaluate them analytically
(numerics handle every conduct uniformly), but they are what the
numerical core approximates. Notation: :math:`D = \partial s / \partial p`,
:math:`H = \partial^2 s / \partial p \partial p`, :math:`\Omega` the
ownership matrix, :math:`\odot` the Hadamard product.

* **Perfect competition** :math:`(p = mc)`:
  :math:`\Delta_{PC} \equiv 0`, so :math:`P_{PC} = I`.

* **Constant markup** :math:`(\Delta = \zeta)`:
  :math:`\Delta_{CM}` is independent of :math:`p`, so
  :math:`P_{CM} = I`.

* **User-supplied markups** (precomputed column):
  treated as exogenous, so :math:`P_{USM} = I`.

* **Rule-of-thumb** :math:`(p = \varphi \cdot mc, \; \varphi \geq 1)`:
  :math:`\Delta_\varphi = (\varphi - 1)/\varphi \cdot p`, so
  :math:`P_\varphi = \varphi I`.

* **Bertrand-Nash**:
  :math:`\Delta_B = -(\Omega \odot D)^{-1} s`. Differentiating
  implicitly,
  :math:`P_B = \big(I + (\Omega \odot D)^{-1} (\Omega \odot H \cdot s)\big)^{-1}`,
  evaluated row-by-row via :math:`H[:, :, k] s` for each price index.

* **Cournot quantity setting**:
  :math:`\Delta_C = -(\Omega \odot D^{-1}) s`. Pass-through is the
  analogue using the inverse demand Jacobian.

* **Monopoly / joint profit maximization**: special case of Bertrand
  with :math:`\Omega = J` (all-ones ownership for joint maximization)
  or the relevant ownership structure for monopoly over the candidate
  product set.

* **Partial collusion**: Bertrand-style with
  :math:`\Omega \to \Omega \odot \kappa` for a profit-weight matrix
  :math:`\kappa`.

* **Mix Cournot/Bertrand**: per-product flag selects Cournot vs.
  Bertrand FOC; pass-through assembles row-by-row from the two
  closed forms.

* **Vertical (downstream-Bertrand, upstream-anything)**: Villas-Boas
  (2007) — see the dedicated section above.
  :py:func:`pyRVtest.solve.passthrough.build_passthrough` keeps the
  existing analytical fast path for this case rather than routing
  through the numerical core.

* **CustomConductModel**: no closed form by construction. The
  numerical core perturbs the user-supplied markup callable directly.


Where the formulas live in code
-------------------------------

* RV statistic and F-stat diagnostic:
  :py:func:`pyRVtest.solve.test_engine.compute_instrument_results`.
* MCS elimination loop:
  :py:func:`pyRVtest.solve.test_engine.compute_mcs`.
* Marginal-cost FOC including taxes and cost-scaling:
  :py:func:`pyRVtest.markups.evaluate_first_order_conditions`.
* Demand-adjustment correction (DMSS Appendix C / DMQSS Appendix B):
  :py:func:`pyRVtest.solve.demand_adjustment.compute_demand_adjustment`.
* Villas-Boas passthrough matrix (Vertical analytical fast path):
  :py:func:`pyRVtest.construct_passthrough_matrix`.
* Numerical pass-through for all conducts (DMQSW Phase 1):
  :py:func:`pyRVtest.solve.passthrough.compute_passthrough_numerical`.
* Pass-through dispatch entry point:
  :py:func:`pyRVtest.solve.passthrough.build_passthrough`.
* mpmath high-precision recompute trigger:
  :py:func:`pyRVtest.solve.test_engine._recompute_F_high_precision`.
