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

For each :math:`(\alpha, K_{\text{eff}})` pair, the *worst-rho* CV is
the maximum over :math:`\rho^2 \in [0, 0.99]` of the rho-indexed CV:

.. math::

   \mathrm{CV}^{\text{worst}}_{\alpha, K_{\text{eff}}}
       = \max_{\rho^2 \in [0, 0.99]}
         \mathrm{CV}_{\alpha, K_{\text{eff}}, \rho^2}.

This is the conservative robustness bound :math:`F` must clear to
support the claim "worst-case size :math:`\le \alpha`" without
relying on the (noisy) plug-in :math:`\hat\rho^2`. The *empirical-rho*
CV is the same table evaluated at the cell's
:math:`\hat\rho^2`; sharper but plug-in-dependent. The package
reports both side-by-side in
:meth:`~pyRVtest.ProblemResults.reliability_summary` (columns
``size_cv_*`` worst-rho vs. ``size_cv_*_emp`` empirical-rho;
analogous power columns). The printed-output verdict uses the
empirical-rho CVs.

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


Pass-through framework for instrument relevance
-----------------------------------------------

The Dearing, Magnolfi, Quint, Sullivan, and Waldfogel (2026, "DMQSW")
framework derives instrument relevance for the conduct test from
candidate models' implied pass-through matrices. Two diagnostic
methods surface the framework view in the package:

* :meth:`~pyRVtest.Problem.passthrough_summary` — pre-solve γ-free
  pair-by-pair structural feature distances.
* :meth:`~pyRVtest.Problem.instrument_channels` — post-solve channel
  decomposition for one chosen IV column, separating the
  pass-through-mediated indirect channel from the markup-derivative
  direct channel.

Decomposition: indirect and direct channels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`z` denote a testing-instrument column and let
:math:`\partial p_m / \partial z` be candidate :math:`m`'s implied
causal effect of :math:`z` on equilibrium prices. Differentiating the
candidate's first-order condition yields the chain-rule decomposition

.. math::

   \frac{\partial p_m}{\partial z}
       = P_m \cdot \Big(\underbrace{\frac{\partial \Delta_m}{\partial z}}_{\text{direct}}
                          + \underbrace{\frac{\partial \bar c}{\partial z}}_{\text{cost-side}}\Big),

where :math:`P_m = (I - \partial \Delta_m / \partial p)^{-1}` is the
candidate pass-through matrix and :math:`\bar c` is the cost
formulation. For two candidates :math:`(m, m')` the per-pair
*difference* in causal effects splits into

.. math::

   \frac{\partial p_m}{\partial z} - \frac{\partial p_{m'}}{\partial z}
       = \underbrace{(P_m - P_{m'}) \cdot \frac{\partial \bar c}{\partial z}}_{\text{indirect channel}}
       + \underbrace{P_m \cdot \frac{\partial \Delta_m}{\partial z}
                       - P_{m'} \cdot \frac{\partial \Delta_{m'}}{\partial z}}_{\text{direct channel}}.

A *cost-shifter* instrument enters only :math:`\partial \bar c /
\partial z`, so :math:`\partial \Delta_m / \partial z = 0` for every
candidate and the direct channel vanishes structurally. A *product-
characteristic* instrument enters :math:`\partial \Delta_m / \partial
z` (through the demand index), so the direct channel is nonzero in
general. A *tax* instrument enters :math:`\partial \bar c / \partial
z` for unit taxes, or both channels via the FOC level-adjustment for
ad-valorem taxes (see DMQSW Section 3.3).

The pass-through-mediated indirect channel is what
:meth:`~pyRVtest.Problem.passthrough_summary` characterizes ex-ante:
the four pass-through-feature distances each correspond to a
projection of :math:`P_m - P_{m'}` against the cost-side response of
a particular instrument type.

Per-instrument-type targeting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DMQSW Remarks 1, 2, 4, and 5 each identify a γ-free pass-through
feature whose distance vanishes if and only if the candidate pair is
structurally indistinguishable under the corresponding instrument
type:

* **Rival cost shifter** (Remark 1). The relevant projection is the
  off-diagonal column of :math:`P_m`: rival cost shifter :math:`w_\ell`
  enters product :math:`j`'s price only through the
  :math:`(j, \ell)` cell. The package reports the column-normalized
  off-diagonal Frobenius norm

  .. math::

     \mathrm{offdiag\_ratio}(m, m')
         = \Big\| \widetilde{P}_m^{\text{off}} - \widetilde{P}_{m'}^{\text{off}} \Big\|_F,
     \qquad \widetilde{P}_{m, j\ell} = P_{m, j\ell} / P_{m, \ell\ell},

  per market and aggregates by median across markets. Zero ⇒ Cournot-
  vs-PerfectCompetition-style degeneracy under logit demand.

* **Own and rival cost shifter** / **product characteristic** under
  linear-index demand (Remark 2 / Remark 4). The relevant projection
  is the *full* matrix :math:`P_m`. The package reports

  .. math::

     \mathrm{full\_pass}(m, m')
         = \big\| P_m - P_{m'} \big\|_F.

  γ-free under rival exclusion; for product-characteristic
  instruments under linear-index demand, the difference of full
  pass-through matrices is the relevant feature.

* **Per-unit tax** (Remark 5, unit). The relevant projection is the
  *row sum* of :math:`P_m`: a uniform :math:`\tau` shifts every
  product by the row sum. The package reports

  .. math::

     \mathrm{row\_sum}(m, m')
         = \Big\| P_m \mathbf{1} - P_{m'} \mathbf{1} \Big\|_2.

  Fully computable: requires only the observed retention factor
  :math:`\nu`.

* **Ad-valorem tax** (Remark 5, ad valorem). The relevant projection
  is the level-adjusted pass-through :math:`P_m (p - \Delta_m)`,
  reflecting the way an ad-valorem rate scales the firm's net
  margin in the FOC. The package reports

  .. math::

     \mathrm{level\_adj}(m, m')
         = \Big\| P_m (p - \Delta_m) - P_{m'} (p - \Delta_{m'}) \Big\|_2.

  Requires observed prices and the candidate's implied markup.

Each feature distance is a *structural* magnitude, not a power
prediction: zero distance under an instrument type rules out the
pair ex-ante; nonzero distance is a necessary but not sufficient
condition for the empirical test to have power. Cross-read against
:meth:`~pyRVtest.ProblemResults.reliability_summary` post-solve.

Direct channel via conditional regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For composite or non-primitive instruments — and as a finite-sample
diagnostic for primitive cost shifters — the package identifies the
direct channel empirically via a Frisch-Waugh-Lovell (FWL) conditional
regression. Let :math:`\Delta_{mjt}` be the model-implied markup of
product :math:`j` in market :math:`t` under candidate :math:`m`,
:math:`p_{jt}` the observed price, and :math:`z_{jt}` the chosen IV
column. The direct-channel coefficient

.. math::

   \beta_m = \frac{\mathrm{Cov}(\Delta_m, z \mid p)}
                  {\mathrm{Var}(z \mid p)}

is the OLS slope of :math:`\Delta_m` on :math:`z` after partialling
:math:`p` out of both. Implementation:

1. Regress :math:`\Delta_{m,\,jt}` on :math:`p_{jt}` and a constant;
   take the residual :math:`\widetilde\Delta_{m,\,jt}`.
2. Regress :math:`z_{jt}` on :math:`p_{jt}` and a constant; take the
   residual :math:`\widetilde z_{jt}`.
3. :math:`\beta_m = \mathrm{Cov}(\widetilde\Delta_m, \widetilde z) /
   \mathrm{Var}(\widetilde z)`.

For a cost-shifter instrument :math:`z = w_\ell` that does not enter
the markup function, :math:`\beta_m \to 0` in population. A small-
but-nonzero estimate in finite samples reflects empirical correlation
between :math:`z` and other markup-function inputs (e.g. product
characteristics) rather than a structural direct channel; the
methodology footer of
:meth:`~pyRVtest.Problem.instrument_channels` documents this. For a
product-characteristic instrument that enters the demand index,
:math:`\beta_m` identifies the structural markup response.

The data-side magnitude :math:`\| \mathrm{d} p_0 / \mathrm{d} z \|_{\mathrm{obs}}`
in :meth:`~pyRVtest.Problem.instrument_channels` is similarly
estimated by OLS regression of observed prices on :math:`z` with
cost-formulation controls; the structural-side magnitude
:math:`\| P_m^{-1} - P_{m'}^{-1} \|_F` aggregates the inverse
pass-through difference by median across markets.


Pass-through diagnostic under non-constant marginal cost
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When marginal cost is *non-constant* — depending on endogenous
variables :math:`q_1, \ldots, q_{K_{\text{endog}}}` (own production,
rival production, polynomials, etc.) — the framework requires both
the Dearing pass-through condition AND the DMQSS Appendix A.4
"economic distinctness" rank condition. The package's
:meth:`~pyRVtest.Problem.instrument_channels` collapses the two into
a single diagnostic via the residualization argument below.

**The DMQSS rank condition.** With :math:`K_{\text{endog}}`
endogenous variables in the cost regression, falsification of
candidate :math:`m` against the truth requires that the
:math:`(K_{\text{endog}} + 1) \times d_z` matrix

.. math::

   M(z, m) = \begin{pmatrix}
       \mathrm{Cov}(z, q_1) \\
       \vdots \\
       \mathrm{Cov}(z, q_{K_{\text{endog}}}) \\
       \mathrm{Cov}(z, f(p - \Delta_m) - f(p - \Delta_0))
   \end{pmatrix}

have rank :math:`K_{\text{endog}} + 1`, where :math:`f` is the cost
transform (identity for ``costs_type='linear'``, log for
``'log'``). For :math:`K_{\text{endog}} = 1` this is the paper's
Equation (7) ratio condition.

**Decomposition.** Standard linear-projection arithmetic decomposes
the rank-:math:`(K_{\text{endog}}+1)` condition into two equivalent
pieces:

1. **First-stage rank** :math:`K_{\text{endog}}`: the first
   :math:`K_{\text{endog}}` rows of :math:`M(z, m)` are linearly
   independent, i.e. the instruments :math:`z` jointly identify the
   endogenous cost variables. The :math:`K_{\text{inst}} >
   K_{\text{endog}}` gate in
   :meth:`~pyRVtest.Problem.solve` enforces the necessary dimension
   count; the residualization-based collapse below tests the
   stronger linear-independence condition empirically.

2. **Residualized Dearing condition**:
   :math:`\mathrm{Cov}(z^e, f(p - \Delta_m) - f(p - \Delta_0))
   \neq 0`, where :math:`z^e = z - \widehat\Lambda_q
   \widetilde q - \widehat\Lambda_w w` is :math:`z` with the
   first-stage prediction of the endogenous variables and the
   exogenous shifters projected out (DMQSS Appendix B notation).

The two-piece decomposition is exact under the parametric
linear-in-basis-columns cost regression that pyRVtest implements:
the cost-coefficient differences :math:`(\gamma_{m,k} - \gamma_{0,k})`
are constants that come out of the moment-level covariance, leaving
only the residualized term :math:`\mathrm{Cov}(z^e, f(p-\Delta_m) -
f(p-\Delta_0))`.

**Implementation.** When ``endogenous_cost_component`` is set, the
data-side regression in
:meth:`~pyRVtest.Problem.instrument_channels` projects on
:math:`(\widetilde q, w_{\text{exog}})` rather than on raw
:math:`(q, w_{\text{exog}})`. This reproduces the :math:`z^e`
residualization the inferential machinery already uses internally
in :py:func:`pyRVtest.solve.test_engine.compute_instrument_results`,
so the pre-solve diagnostic and the post-solve test agree on which
moment direction carries identifying variation.

The constant-MC case (``endogenous_cost_component is None``) is the
:math:`K_{\text{endog}} = 0` special case: the residualization
collapses to projection on :math:`w` only, recovering the standard
Dearing-decomposition behavior.


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
* Numerical pass-through for all conducts:
  :py:func:`pyRVtest.solve.passthrough.compute_passthrough_numerical`.
* Pass-through dispatch entry point:
  :py:func:`pyRVtest.solve.passthrough.build_passthrough`.
* Pre-solve framework diagnostic
  (:meth:`~pyRVtest.Problem.passthrough_summary`):
  :py:func:`pyRVtest.solve.passthrough.compute_passthrough_summary`.
* Post-solve channel decomposition
  (:meth:`~pyRVtest.Problem.instrument_channels`):
  :py:func:`pyRVtest.solve.passthrough.compute_instrument_channels`.
* mpmath high-precision recompute trigger:
  :py:func:`pyRVtest.solve.test_engine._recompute_F_high_precision`.
