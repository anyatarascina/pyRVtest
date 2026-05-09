Migrating to v0.4
=================

pyRVtest v0.4 introduces a class-based API for specifying conduct models. The
legacy string-based ``ModelFormulation`` API still works through v0.5; it is
deprecated and scheduled for removal in v0.6.

This page documents the migration for every common pattern. If your case is
not covered, open an issue.

Quick summary
-------------

**Before (v0.3):**

.. code-block:: python

    problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
        product_data=data,
        demand_results=pyblp_results,
        model_formulations=(
            pyRVtest.ModelFormulation(
                model_downstream='bertrand',
                ownership_downstream='firm_ids',
            ),
            pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
        ),
    )

**After (v0.4):**

.. code-block:: python

    problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
        product_data=data,
        demand_results=pyblp_results,
        models=[
            pyRVtest.Bertrand(ownership='firm_ids'),
            pyRVtest.PerfectCompetition(),
        ],
    )

The migration rules
-------------------

#. ``model_formulations=(X, Y, ...)`` becomes ``models=[X, Y, ...]``.
#. ``ModelFormulation(model_downstream='bertrand', ...)`` becomes ``Bertrand(...)``;
   similarly for ``Cournot``, ``Monopoly``, ``PerfectCompetition``,
   ``MixCournotBertrand``.
#. ``ownership_downstream='firm_ids'`` becomes ``ownership='firm_ids'``.
#. Vertical (both ``model_downstream`` and ``model_upstream`` set) →
   ``Vertical(downstream=..., upstream=..., ...)``; see :ref:`vertical-migration`.
#. ``kappa_specification_downstream=...`` → ``kappa_specification=...`` on the
   relevant class. Use :class:`~pyRVtest.PartialCollusion` for collusion-flavored
   ownership.
#. ``custom_model_specification={name: callable}`` + ``model_downstream='other'``
   → ``CustomConductModel(markup_fn=callable, name=name, ownership=...)``.
#. ``cost_scaling`` and ``user_supplied_markups`` keep their names. Per-unit and
   ad-valorem taxes (``unit_tax``, ``advalorem_tax``, ``advalorem_payer``) move
   from the model to :class:`~pyRVtest.Problem`; see :ref:`tax-migration`.

Each case in detail
-------------------

Standard oligopoly (Bertrand, Cournot, Monopoly, PerfectCompetition)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Class name replaces the string; ``ownership_downstream`` becomes ``ownership``.
``PerfectCompetition`` takes no ownership argument.

.. code-block:: python

    # Before
    pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids')
    pyRVtest.ModelFormulation(model_downstream='cournot', ownership_downstream='firm_ids')
    pyRVtest.ModelFormulation(model_downstream='monopoly', ownership_downstream='firm_ids')
    pyRVtest.ModelFormulation(model_downstream='perfect_competition')

    # After
    pyRVtest.Bertrand(ownership='firm_ids')
    pyRVtest.Cournot(ownership='firm_ids')
    pyRVtest.Monopoly(ownership='firm_ids')
    pyRVtest.PerfectCompetition()

Partial collusion
^^^^^^^^^^^^^^^^^

Dedicated class; ``kappa_specification_downstream`` → ``kappa_specification``.

.. code-block:: python

    # Before
    pyRVtest.ModelFormulation(
        model_downstream='bertrand', ownership_downstream='firm_ids',
        kappa_specification_downstream='collusion_row',
    )

    # After
    pyRVtest.PartialCollusion(ownership='firm_ids', kappa_specification='collusion_row')

Mix Cournot/Bertrand
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Before
    pyRVtest.ModelFormulation(
        model_downstream='mix_cournot_bertrand', ownership_downstream='firm_ids',
        mix_flag='bertrand_products',
    )

    # After
    pyRVtest.MixCournotBertrand(ownership='firm_ids', mix_flag='bertrand_products')

Custom conduct (user-supplied markup formula)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def my_markup(ownership, response_matrix, shares):
        ...
        return markups

    # Before
    pyRVtest.ModelFormulation(
        model_downstream='other', ownership_downstream='firm_ids',
        custom_model_specification={'my_model': my_markup},
    )

    # After
    pyRVtest.CustomConductModel(markup_fn=my_markup, ownership='firm_ids', name='my_model')

User-supplied markups
^^^^^^^^^^^^^^^^^^^^^

The ``user_supplied_markups`` kwarg keeps its name. Supply on the conduct
class; :class:`~pyRVtest.PerfectCompetition` is a natural choice when you do
not want the test to depend on the model's own FOC.

.. code-block:: python

    # Before
    pyRVtest.ModelFormulation(
        model_downstream='bertrand', ownership_downstream='firm_ids',
        user_supplied_markups='precomputed_markup_col',
    )

    # After
    pyRVtest.Bertrand(
        ownership='firm_ids',
        user_supplied_markups='precomputed_markup_col',
    )

.. _vertical-migration:

Vertical (bilateral oligopoly)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The biggest restructuring. Old API set both downstream and upstream conduct on
a single ``ModelFormulation``. New API uses :class:`~pyRVtest.Vertical` as a
wrapper with each side specified as its own class instance.

.. code-block:: python

    # Before
    pyRVtest.ModelFormulation(
        model_downstream='bertrand', ownership_downstream='firm_ids',
        model_upstream='monopoly', ownership_upstream='manufacturer_ids',
        vertical_integration='vi_col',
    )

    # After
    pyRVtest.Vertical(
        downstream=pyRVtest.Bertrand(ownership='firm_ids'),
        upstream=pyRVtest.Monopoly(ownership='manufacturer_ids'),
        vertical_integration='vi_col',
    )

Each inner conduct carries its own ``ownership`` (and, where applicable,
``kappa_specification``). Tax, vertical-integration, and cost-scaling kwargs
go on the wrapper, not the inner conducts.

.. _tax-migration:

Taxes and cost scaling
^^^^^^^^^^^^^^^^^^^^^^

Tax kwargs (``unit_tax``, ``advalorem_tax``, ``advalorem_payer``) move from
the per-model level to :class:`~pyRVtest.Problem`. Per-unit and ad-valorem
taxes describe the data-generating process, not a behavioral choice, so the
DGP is the right home. Individual models can opt out via
``unit_tax_salient=False`` and ``advalorem_tax_salient=False`` flags
(both default to ``True``) for salience tests.

The legacy per-model spelling continues to work through v0.6 with a
``DeprecationWarning``. ``cost_scaling`` stays on the conduct model
(or on :class:`~pyRVtest.Vertical`); it is a behavioral primitive.

.. code-block:: python

    # Before (v0.3 — tax repeated on every model)
    pyRVtest.ModelFormulation(
        model_downstream='bertrand', ownership_downstream='firm_ids',
        advalorem_tax='tax_rate_col', advalorem_payer='firm',
        cost_scaling='scale_col',
    )

    # After (v0.4 preferred — tax on Problem, salience flag for opt-outs)
    pyRVtest.Problem(
        cost_formulation=...,
        instrument_formulation=...,
        product_data=product_data,
        advalorem_tax='tax_rate_col',
        advalorem_payer='firm',
        models=[
            pyRVtest.Bertrand(ownership='firm_ids', cost_scaling='scale_col'),
            pyRVtest.Cournot(ownership='firm_ids', cost_scaling='scale_col'),
            # Salience test: same Bertrand, but tax-blind:
            pyRVtest.Bertrand(
                ownership='firm_ids', cost_scaling='scale_col',
                advalorem_tax_salient=False,
            ),
        ],
    )

.. _tax-precedence-tiebreaker:

If a tax is set on **both** :class:`~pyRVtest.Problem` and a conduct model,
the model-level value wins for that specific model and two
``DeprecationWarning`` messages fire (the standard per-model deprecation, plus
a conflict warning naming both columns). Recommended migration: drop the
per-model ``unit_tax='...'`` and decide per model whether to keep the
Problem-level tax (do nothing) or opt out (``unit_tax_salient=False``).

Known-coefficient cost shifters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

v0.4 adds a ``known_coefficients`` kwarg on :class:`~pyRVtest.Formulation` for
cost shifters with researcher-supplied (non-estimated) coefficients:

.. math::

    \texttt{prices\_effective} = \frac{\tau_{\text{av}} \cdot p}{1 + \lambda}
    - \tau_{\text{unit}} - \sum_k \gamma_k \cdot x_k

where :math:`(x_k, \gamma_k)` are the known-coefficient column and value.
Per-unit taxes are the leading special case; Dearing et al. (2026) work with
broader classes.

.. code-block:: python

    cost_formulation = pyRVtest.Formulation(
        '0 + z1',
        known_coefficients={'input_price': 0.75, 'union_wage': 1.0},
    )

Each column must be in ``product_data`` and must NOT appear in the formula
string. Known-coefficient shifters apply uniformly to every model and have no
salience opt-out (they are a DGP primitive, not a behavioral input).

Dearing simple-markup models (RuleOfThumb, ConstantMarkup)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

v0.4 adds two simple-markup models from Dearing, Magnolfi, Quint, Sullivan,
and Waldfogel (2026):

* :class:`~pyRVtest.RuleOfThumb` — :math:`p = \varphi \cdot mc` for a fixed
  multiple :math:`\varphi \geq 1`. Ergonomic wrapper over ``cost_scaling``,
  which now also accepts a numeric scalar.
* :class:`~pyRVtest.ConstantMarkup` — fixed per-product dollar markup
  :math:`\Delta_{jt} = \zeta_j`. Scalar (uniform) or column name.

.. code-block:: python

    # v0.3 idiom: per-product lambda column of ones for phi=2 uniformly
    product_data['lmbda_col'] = 1.0
    pyRVtest.ModelFormulation(model_downstream='perfect_competition', cost_scaling='lmbda_col')

    # v0.4 shorthand:
    pyRVtest.RuleOfThumb(phi=2.0)
    pyRVtest.ConstantMarkup(markup=0.5)

The v0.3 ``PerfectCompetition(cost_scaling='col')`` form still works.

``demand_params['rho']`` vs ``demand_params['sigma']``
------------------------------------------------------

The nested-logit correlation parameter inside ``demand_params`` is now
canonically named ``rho`` (matching ``pyblp``'s nomenclature).
``demand_params['sigma']`` is a deprecated alias with a once-per-session
``DeprecationWarning``. Supplying both raises ``TypeError``.

.. code-block:: python

    # Before
    demand_params = {'alpha': alpha_hat, 'sigma': [0.3], 'beta': beta}
    # After
    demand_params = {'alpha': alpha_hat, 'rho': [0.3], 'beta': beta}

The :class:`~pyRVtest.backends.NestedLogitBackend` constructor (if you build
one directly) still uses ``sigma=[...]`` — its internal math follows the
AFSSZ L-level convention. Only the user-facing ``demand_params`` dict
changed.

Labor-side conduct testing (new in v0.4)
----------------------------------------

v0.4 adds ``Problem(market_side='labor')`` for labor-supply conduct testing.
Default ``market_side='product'`` is unchanged. Labor classes:
:class:`~pyRVtest.Monopsony`, :class:`~pyRVtest.BertrandWages`,
:class:`~pyRVtest.CournotEmployment`, :class:`~pyRVtest.NashBargaining`
(``NotImplementedError`` placeholder). Default column names are ``'wages'``
and ``'employment_share'``; override via ``column_names=`` if needed.

.. code-block:: python

    pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation('0 + iv0'),
        product_data=labor_df,
        models=[
            pyRVtest.Monopsony(user_supplied_markups='markdown_m1'),
            pyRVtest.PerfectCompetition(user_supplied_markups='markdown_m2'),
        ],
        market_side='labor',
    )

The :class:`~pyRVtest.backends.LaborSupplyBackend` ships as a v0.4 skeleton —
``compute_jacobian`` / ``compute_hessian`` raise
:class:`NotImplementedError`, and ``demand_params=`` paired with
``market_side='labor'`` similarly raises. Use ``user_supplied_markups`` per
model (or wrap your own Jacobian via
:class:`~pyRVtest.backends.UserSuppliedBackend`) until the analytical labor
backend lands in v0.5. See :doc:`faq` for additional caveats.

In-package demand estimation (new in v0.4)
------------------------------------------

If you ran PyBLP only as a preprocessing step for plain-logit or one-level
nested-logit demand, you can collapse the pipeline:

.. code-block:: python

    # v0.3 / v0.4 PyBLP path (still supported)
    pyblp_results = pyblp.Problem(
        product_formulations=(pyblp.Formulation('1 + prices + x1'),),
        product_data=data,
    ).solve(method='1s')

    rv_problem = pyRVtest.Problem(..., demand_results=pyblp_results)

    # v0.4 in-package equivalent
    rv_problem = pyRVtest.Problem(
        ...,
        demand_params={
            'estimate': 'logit',
            'formulation_X': pyRVtest.Formulation('1 + x1'),
            'formulation_Z': pyRVtest.Formulation('0 + z1'),
        },
    )

See :doc:`in_package_demand` for the standalone path
(``pyRVtest.LogitEstimator(...).solve()`` returning a ``demand_params`` dict),
the nested-logit variant, and trade-offs between forms. Continue using
PyBLP + ``demand_results=`` for random coefficients, micro-moments, and
multi-level nesting.

``models=`` vs ``model_formulations=``
--------------------------------------

:class:`~pyRVtest.Problem` accepts either keyword, not both (mixing raises
``TypeError``). During the v0.4-v0.5 deprecation window, both produce
byte-identical results for equivalent inputs.

Deprecation timeline
--------------------

Different deprecations have different runways. The schedule is cumulative:

* **v0.4 (current):** class-based API lands. ``ModelFormulation`` emits
  ``DeprecationWarning`` once per session on first construction.
  ``model_formulations=``, ``demand_params=dict(sigma=...)``,
  ``pyRVtest.output.output()``, and per-model ``unit_tax`` / ``advalorem_tax`` /
  ``advalorem_payer`` also emit deprecation warnings.
* **v0.5:** continued migration window for everything.
* **v0.6:** ``ModelFormulation``, ``model_formulations=``,
  ``demand_params=dict(sigma=...)``, and ``pyRVtest.output.output()`` removed.
  Per-model tax kwargs keep working one more release.
* **v0.7:** per-model ``unit_tax`` / ``advalorem_tax`` / ``advalorem_payer``
  removed. Only ``Problem``-level tax kwargs with per-model salience opt-outs
  remain.

Suppressing the warning
-----------------------

If you are not ready to migrate yet:

.. code-block:: python

    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='pyRVtest')

We recommend keeping the warning visible.

rc1 → final changes
-------------------

The 0.4.0rc1 tag was an internal / coauthor-only release. Several
F-stat reliability and pass-through diagnostic names changed between rc1
and v0.4 final, and there is no deprecation alias; rc1 callers must
update their code.

* :meth:`ProblemResults.F_reliability_summary` is renamed to
  :meth:`~ProblemResults.reliability_summary`. No alias; rc1 calls
  raise :class:`AttributeError`.
* The ``F_se`` / ``F_ci_low`` / ``F_ci_high`` per-cell array attributes
  on :class:`ProblemResults` are removed entirely, along with their
  columns in :meth:`~ProblemResults.reliability_summary`. The plug-in
  CV check the verdict already runs is the principled robustness signal;
  the asymptotic SE was redundant inspection-only metadata.
* The ``verdict`` column was removed from
  :meth:`~ProblemResults.reliability_summary`'s returned DataFrame. The
  internal classification still drives the ``⚠`` warning glyph in the
  printed output and remains accessible as
  ``ProblemResults.verdict[instrument_set]`` (an ``(M, M)`` object
  array per instrument set), but the diagnostic frame no longer exposes
  it as a column.
* The ``worst_case_cv_size`` / ``worst_case_cv_power`` columns of
  :meth:`~ProblemResults.reliability_summary` (which were object-array
  cells holding length-3 vectors) were replaced by six scalar columns
  reporting the worst-rho CVs at the published-table levels:

  * Size: ``size_cv_075`` / ``size_cv_100`` / ``size_cv_125`` for the
    7.5% / 10% / 12.5% size levels.
  * Power: ``power_cv_050`` / ``power_cv_075`` / ``power_cv_095`` for
    the 50% / 75% / 95% power levels.

  Six matching empirical-rho columns (``size_cv_075_emp``,
  ``size_cv_100_emp``, ``size_cv_125_emp``, ``power_cv_050_emp``,
  ``power_cv_075_emp``, ``power_cv_095_emp``) report the plug-in CVs at
  the cell's empirical ρ̂² — the same CVs the verdict uses internally.
  The rc1 ``strongest_claim_size`` / ``strongest_claim_power`` columns
  are unchanged; the strongest claim string is equivalent to the F
  clearing the relevant ``size_cv_075`` / ``power_cv_095`` column.
* :meth:`ProblemResults.passthrough_comparison` is removed entirely.
  The Dearing et al. (2026) pass-through diagnostic suite in v0.4 final
  is provided by :meth:`Problem.passthrough_summary` (γ-free pair-by-
  pair structural-feature distance, callable pre- or post-solve) and
  :meth:`Problem.instrument_channels` (post-solve per-pair channel
  decomposition for one IV column); see :doc:`advanced_features` for
  the three-method walkthrough. The rc1 method's ``offdiag_frobenius``
  metric was also paper-renumbered: the off-diagonal-to-diagonal
  feature is Remark 1 in the current DMQSW draft, not Remark 4.

New diagnostic methods in v0.4 final
------------------------------------

v0.4 final ships the full DMQSW (2026) pass-through framework as
first-class diagnostic methods on :class:`~pyRVtest.Problem` and
:class:`~pyRVtest.ProblemResults`:

* :meth:`Problem.passthrough_summary` — pre-solve γ-free pair-by-pair
  structural-feature distances. Four metrics correspond to DMQSW
  Remarks 1, 2, 4, and 5: ``offdiag_ratio`` (rival cost shifters),
  ``full_pass`` (own+rival cost / product chars under linear-index
  demand), ``row_sum`` (per-unit tax), ``level_adj`` (ad-valorem tax).
  Optional ``with_models=True`` adds a per-model structural block;
  ``detail='full'`` returns one row per ``(pair, market)``. Also
  callable on :class:`~pyRVtest.ProblemResults` post-solve.
* :meth:`Problem.instrument_channels` — post-solve per-pair channel
  decomposition for one chosen IV column. Reports the data-side
  empirical magnitude :math:`\| \mathrm{d} p_0 / \mathrm{d} z \|`, the
  per-candidate direct-channel coefficient :math:`\beta_m` (FWL
  partialling), the structural-side magnitude
  :math:`\| P_m^{-1} - P_{m'}^{-1} \|_F`, and the per-pair
  ``|β_m − β_m'|`` direct difference. Also accessible on
  :class:`~pyRVtest.ProblemResults`.
* The pre-existing :meth:`ProblemResults.passthrough_matrix` is now
  general — it returns :math:`P_m` for *every* conduct class via
  numerical central-difference perturbation (with the existing
  Villas-Boas analytical fast path for :class:`~pyRVtest.Vertical`
  preserved). The ``Vertical``-only restriction in v0.4 rc1 is
  removed.

The v0.4 final pass-through machinery is computed *numerically* by
default (one routine handles every conduct uniformly), with
analytical fast paths for ``Vertical`` and short-circuit identity /
:math:`\varphi I` for trivial conducts (``PerfectCompetition``,
``ConstantMarkup``, ``UserSuppliedMarkups``, ``RuleOfThumb``). Per-
conduct closed-form expressions are documented in :doc:`math` as
reference. See the methodology footer in each diagnostic's printed
output for which paths fired on your candidate set.

Multi-endogenous-variable cost regressions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``endogenous_cost_component`` now accepts either a single column name
(the original single-endogenous case, unchanged) or a list of column
names for multi-endogenous-variable cost regressions per Dearing,
Magnolfi, Quint, Sølvsten, and Sullivan (2026, "DMQSS") Appendix A.4.

.. code-block:: python

   # v0.4 single endogenous variable (unchanged):
   pyRVtest.Problem(
       ...,
       endogenous_cost_component='log_quantity',
   )

   # v0.4 final: multi-endogenous variables (DMQSS A.4 examples)
   pyRVtest.Problem(  # quadratic cost: c = γ_1 q + γ_2 q² + w'τ + ω
       ...,
       endogenous_cost_component=['q', 'q_sq'],
   )
   pyRVtest.Problem(  # scale + scope: log(c) = γ_1 log(q) + γ_2 log(Q⁻) + w'τ + ω
       ...,
       endogenous_cost_component=['log_q', 'log_Q_minus'],
   )

The instrument bundle must satisfy ``K_inst > K_endog`` per testing-
IV set (paper Remark 1); ``Problem.solve`` raises a
:class:`ValueError` listing every offending instrument set if not.
The pre-existing single-string API path remains bit-identical.

``costs_type='log' + demand_adjustment=True``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The combination is now fully supported. Pre-v0.4-final the package
silently fell back to ``costs_type='linear'`` with a UserWarning;
the chain rule is now correctly applied (``gradient_markups``
rescaled by :math:`f'(p - \Delta_m) = 1/(p - \Delta_m)`). No code
change required from users — just the surprise behavior is gone.

Reporting issues
----------------

If you have a configuration that does not translate cleanly, open an issue.
We can either add an example here or extend the API.
