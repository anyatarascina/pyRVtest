Migrating to v0.4
=================

pyRVtest v0.4 introduces a class-based API for specifying conduct models. The
legacy string-based ``ModelFormulation`` API still works and will continue to
work through v0.5; it is deprecated and scheduled for removal in v0.6.

This page documents the migration for every common usage pattern. If you have a
case that isn't covered here, open an issue — we'll add an example.

Quick summary
-------------

**Before (v0.3):**

.. code-block:: python

    import pyRVtest

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

    import pyRVtest

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

1. ``model_formulations=(X, Y, ...)`` (tuple/sequence) becomes ``models=[X, Y, ...]`` (list).
2. ``ModelFormulation(model_downstream='bertrand', ...)`` becomes ``Bertrand(...)``; similarly for ``Cournot``, ``Monopoly``, ``PerfectCompetition``, ``MixCournotBertrand``.
3. ``ownership_downstream='firm_ids'`` becomes ``ownership='firm_ids'``.
4. For vertical models (``model_upstream`` set in the old API), use the ``Vertical(downstream=..., upstream=..., ...)`` wrapper; see :ref:`vertical-migration` below.
5. ``kappa_specification_downstream=...`` becomes ``kappa_specification=...`` on the relevant class; use the dedicated ``PartialCollusion`` class for collusion-flavored ownership.
6. ``custom_model_specification={name: callable}`` + ``model_downstream='other'`` becomes ``CustomConductModel(markup_fn=callable, name=name, ownership=...)``.
7. ``cost_scaling`` and ``user_supplied_markups`` keep the same name on the new classes. Per-unit and ad-valorem taxes (``unit_tax``, ``advalorem_tax``, ``advalorem_payer``) move from the model to the :class:`Problem` (v0.4 OQ 14); see the "Taxes and cost scaling" subsection below. The legacy per-model spelling still works and emits a ``DeprecationWarning``.

Each case in detail
-------------------

Simple Bertrand
^^^^^^^^^^^^^^^

**Before:**

.. code-block:: python

    pyRVtest.ModelFormulation(
        model_downstream='bertrand',
        ownership_downstream='firm_ids',
    )

**After:**

.. code-block:: python

    pyRVtest.Bertrand(ownership='firm_ids')

Cournot, Monopoly, PerfectCompetition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Same pattern as Bertrand: the class name replaces the string, and
``ownership_downstream`` becomes ``ownership``. ``PerfectCompetition`` takes no
ownership argument (markups are structurally zero).

.. code-block:: python

    # Before
    pyRVtest.ModelFormulation(model_downstream='cournot', ownership_downstream='firm_ids')
    pyRVtest.ModelFormulation(model_downstream='monopoly', ownership_downstream='firm_ids')
    pyRVtest.ModelFormulation(model_downstream='perfect_competition')

    # After
    pyRVtest.Cournot(ownership='firm_ids')
    pyRVtest.Monopoly(ownership='firm_ids')
    pyRVtest.PerfectCompetition()

Partial collusion
^^^^^^^^^^^^^^^^^

The old API expressed partial collusion via ``kappa_specification_downstream``
on a Bertrand formulation. The new API has a dedicated ``PartialCollusion``
class that requires ``kappa_specification`` and signals intent at the call site:

**Before:**

.. code-block:: python

    pyRVtest.ModelFormulation(
        model_downstream='bertrand',
        ownership_downstream='firm_ids',
        kappa_specification_downstream='collusion_row',
    )

**After:**

.. code-block:: python

    pyRVtest.PartialCollusion(
        ownership='firm_ids',
        kappa_specification='collusion_row',
    )

Mix Cournot/Bertrand
^^^^^^^^^^^^^^^^^^^^

The ``mix_flag`` column name moves from a ``ModelFormulation`` kwarg to the
``MixCournotBertrand`` constructor.

**Before:**

.. code-block:: python

    pyRVtest.ModelFormulation(
        model_downstream='mix_cournot_bertrand',
        ownership_downstream='firm_ids',
        mix_flag='bertrand_products',
    )

**After:**

.. code-block:: python

    pyRVtest.MixCournotBertrand(
        ownership='firm_ids',
        mix_flag='bertrand_products',
    )

Custom conduct (user-supplied markup formula)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The old ``model_downstream='other'`` + ``custom_model_specification={name: fn}``
pattern becomes a direct ``CustomConductModel`` construction:

**Before:**

.. code-block:: python

    def my_markup(ownership, response_matrix, shares):
        ...
        return markups

    pyRVtest.ModelFormulation(
        model_downstream='other',
        ownership_downstream='firm_ids',
        custom_model_specification={'my_model': my_markup},
    )

**After:**

.. code-block:: python

    def my_markup(ownership, response_matrix, shares):
        ...
        return markups

    pyRVtest.CustomConductModel(
        markup_fn=my_markup,
        ownership='firm_ids',
        name='my_model',
    )

.. _vertical-migration:

Vertical (bilateral oligopoly)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the migration that requires the most restructuring. In the old API the
downstream and upstream conducts were both set on a single ``ModelFormulation``
via two different kwargs (``model_downstream`` / ``model_upstream`` and
``ownership_downstream`` / ``ownership_upstream``). The new API uses a
``Vertical(downstream=..., upstream=..., ...)`` wrapper with shared config (tax,
vertical_integration) on the wrapper itself:

**Before:**

.. code-block:: python

    pyRVtest.ModelFormulation(
        model_downstream='bertrand',
        ownership_downstream='firm_ids',
        model_upstream='monopoly',
        ownership_upstream='manufacturer_ids',
        vertical_integration='vi_col',
    )

**After:**

.. code-block:: python

    pyRVtest.Vertical(
        downstream=pyRVtest.Bertrand(ownership='firm_ids'),
        upstream=pyRVtest.Monopoly(ownership='manufacturer_ids'),
        vertical_integration='vi_col',
    )

Each inner conduct carries its own ``ownership`` (and, where applicable,
``kappa_specification``). Tax, vertical-integration, and cost-scaling kwargs go
on the ``Vertical`` wrapper — not on the inner conducts — because they apply to
the combined vertical model.

Taxes and cost scaling
^^^^^^^^^^^^^^^^^^^^^^

v0.4 OQ 14 elevates tax kwargs (``unit_tax``, ``advalorem_tax``,
``advalorem_payer``) to :class:`~pyRVtest.Problem`. Per-unit and
ad-valorem taxes describe the data-generating process, not a firm's
behavioral choice, so the DGP is the right home for them. Individual
conduct models can opt out of Problem-level taxes for salience tests
via the new ``unit_tax_salient`` and ``advalorem_tax_salient`` flags
(both default to ``True``).

The legacy per-model ``unit_tax`` / ``advalorem_tax`` /
``advalorem_payer`` kwargs continue to work and win by precedence
when both are set (each emits a ``DeprecationWarning`` pointing here).
Expect them to be removed in v0.6. ``cost_scaling`` stays on the
conduct model (or on :class:`~pyRVtest.Vertical` for bilateral
oligopoly) because it is a behavioral primitive.

v0.4 extends ``cost_scaling`` to accept either a column name in
``product_data`` (the v0.3 behavior, unchanged) **or** a numeric scalar
(new in v0.4 step 12) broadcast uniformly to every product. The scalar
form is the foundation of the ergonomic
:class:`~pyRVtest.RuleOfThumb` / :class:`~pyRVtest.Keystone` wrappers
described below.

**Before (v0.3 — tax on every model):**

.. code-block:: python

    pyRVtest.ModelFormulation(
        model_downstream='bertrand',
        ownership_downstream='firm_ids',
        advalorem_tax='tax_rate_col',
        advalorem_payer='firm',
        cost_scaling='scale_col',
    )

**Intermediate (v0.4 with legacy per-model taxes; deprecated, still works):**

.. code-block:: python

    pyRVtest.Bertrand(
        ownership='firm_ids',
        advalorem_tax='tax_rate_col',  # deprecation warning
        advalorem_payer='firm',
        cost_scaling='scale_col',
    )

**After (v0.4 preferred — tax on Problem, salience flag on model):**

.. code-block:: python

    problem = pyRVtest.Problem(
        cost_formulation=...,
        instrument_formulation=...,
        product_data=product_data,
        advalorem_tax='tax_rate_col',
        advalorem_payer='firm',
        models=[
            pyRVtest.Bertrand(ownership='firm_ids', cost_scaling='scale_col'),
            pyRVtest.Cournot(ownership='firm_ids', cost_scaling='scale_col'),
            # Salience test: same Bertrand model, but this instance
            # ignores the ad-valorem tax:
            pyRVtest.Bertrand(
                ownership='firm_ids', cost_scaling='scale_col',
                advalorem_tax_salient=False,
            ),
        ],
    )

The Problem-level tax is applied to every model with the salience
flag set to its default ``True``; the third model opts out, so its
implied marginal cost differs even though its raw Bertrand markup is
identical.

Known-coefficient cost shifters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

v0.4 OQ 14 also adds a ``known_coefficients`` kwarg on
:class:`~pyRVtest.Formulation`. Cost shifters with researcher-supplied
(non-estimated) coefficients enter the effective-price line directly:

.. math::

    \texttt{prices\_effective} = \frac{\tau_{\text{av}} \cdot p}
    {1 + \lambda} - \tau_{\text{unit}}
    - \sum_k \gamma_k \cdot x_k

where :math:`(x_k, \gamma_k)` are the known-coefficient column and
coefficient. Per-unit taxes are the leading special case
(``known_coefficients={'tax_col': 1.0}`` is equivalent to
``Problem(unit_tax='tax_col')``). Dearing et al. (2026) work with a
broader class of such shifters.

.. code-block:: python

    cost_formulation = pyRVtest.Formulation(
        '0 + z1',
        known_coefficients={'input_price': 0.75, 'union_wage': 1.0},
    )

Known-coefficient shifters apply uniformly to every model (they are a
DGP primitive, not a behavioral choice, so they carry no salience
flag). Each column must be in ``product_data`` and must NOT appear in
the formula string — doing so would double-count.

Dearing simple-markup models (RuleOfThumb, Keystone, ConstantMarkup)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

v0.4 step 12 adds three simple-markup models from Dearing, Magnolfi, Quint,
Sullivan, and Waldfogel (2026):

* :class:`~pyRVtest.RuleOfThumb` / :class:`~pyRVtest.Keystone` —
  Example 1: price is a fixed multiple :math:`\varphi \geq 1` of
  marginal cost, :math:`p = \varphi \cdot mc`. ``Keystone()`` is the
  ``phi = 2`` shorthand. Implemented as an ergonomic wrapper over the
  existing ``cost_scaling`` machinery (now extended to accept a scalar).

* :class:`~pyRVtest.ConstantMarkup` — Example 7: fixed per-product
  dollar markup :math:`\Delta_{jt} = \zeta_j`. Supply as a scalar
  (identical across products) or as a column name.

The v0.3 pattern of ``PerfectCompetition(cost_scaling='lmbda_col')`` for
per-product rule-of-thumb markups continues to work unchanged.
``RuleOfThumb(phi=...)`` is the v0.4 shorthand for the common case
where :math:`\varphi` is the same across all products:

**v0.3:**

.. code-block:: python

    # per-product lambda via a column of ones would give you phi=2 uniformly
    product_data['lmbda_col'] = 1.0
    pyRVtest.ModelFormulation(
        model_downstream='perfect_competition',
        cost_scaling='lmbda_col',
    )

**v0.4:**

.. code-block:: python

    pyRVtest.RuleOfThumb(phi=2.0)        # or Keystone()
    pyRVtest.ConstantMarkup(markup=0.5)  # fixed $0.50 markup for every product

User-supplied markups
^^^^^^^^^^^^^^^^^^^^^

The ``user_supplied_markups`` kwarg keeps its name. Supply it on the conduct
class (``PerfectCompetition`` is a natural choice if you're not testing the
conduct itself — the math is bypassed when user-supplied markups are set):

**Before:**

.. code-block:: python

    pyRVtest.ModelFormulation(
        model_downstream='bertrand',
        ownership_downstream='firm_ids',
        user_supplied_markups='precomputed_markup_col',
    )

**After:**

.. code-block:: python

    pyRVtest.Bertrand(
        ownership='firm_ids',
        user_supplied_markups='precomputed_markup_col',
    )

``demand_params['rho']`` vs ``demand_params['sigma']``
------------------------------------------------------

The nested-logit correlation parameter inside ``demand_params`` is now
canonically named ``rho`` (matching ``pyblp``'s nomenclature).
``demand_params['sigma']`` continues to work as a deprecated alias with a
once-per-session ``DeprecationWarning``. Supplying both keys raises a
``TypeError``.

**Before:**

.. code-block:: python

    problem = pyRVtest.Problem(
        ...,
        demand_params={
            'alpha': alpha_hat,
            'sigma': [0.3],
            'beta': beta,
            'x_columns': ['intercept', 'x1'],
            'demand_instrument_columns': [...],
        },
    )

**After:**

.. code-block:: python

    problem = pyRVtest.Problem(
        ...,
        demand_params={
            'alpha': alpha_hat,
            'rho': [0.3],
            'beta': beta,
            'x_columns': ['intercept', 'x1'],
            'demand_instrument_columns': [...],
        },
    )

The ``NestedLogitBackend`` class (if you construct one directly) still uses
``sigma=[...]`` as its constructor kwarg, because its internal math follows
the AFSSZ L-level convention where ``sigma_l`` is a per-level parameter.
Only the user-facing ``demand_params`` dict changed names.

Labor-side (v0.4 step 14, new)
------------------------------

v0.4 adds ``Problem(market_side='labor')`` for labor-supply conduct
testing. This is entirely additive — the default ``market_side='product'``
is byte-identical to pre-v0.4 behavior. The labor path is opt-in.

Labor conduct classes live in ``pyRVtest.models.labor`` and are
re-exported at the top level:

* :class:`pyRVtest.Monopsony` — single-firm wage-setter.
* :class:`pyRVtest.BertrandWages` — wage-setting Bertrand.
* :class:`pyRVtest.CournotEmployment` — employment-setting Cournot.
* :class:`pyRVtest.NashBargaining` — raises ``NotImplementedError`` in
  v0.4; formula deferred to v0.5 when labor data is available.

Minimal labor problem:

.. code-block:: python

    problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation('0 + iv0'),
        product_data=labor_df,          # columns: wages, employment_share, ...
        models=[
            pyRVtest.Monopsony(user_supplied_markups='markdown_m1'),
            pyRVtest.PerfectCompetition(user_supplied_markups='markdown_m2'),
        ],
        market_side='labor',
    )
    results = problem.solve()

Column-name defaults for labor mode are ``'wages'`` and
``'employment_share'``; pyRVtest treats the ``shares`` column as a share
(values in ``[0, 1]``, summing to at most 1 per market), so the default
name advertises the units rather than naming a raw quantity. Users with
raw employment counts must normalize to market-level employment shares
first. Override either default via ``column_names``:

.. code-block:: python

    pyRVtest.Problem(
        ...,
        market_side='labor',
        column_names={'price': 'my_wage_col', 'shares': 'my_emp_share_col'},
    )

Sign validation at ``Problem.__init__`` requires ``wages > 0`` and
``employment_share > 0`` on every row. Violations raise
:class:`pyRVtest.ValidationError` with the expected / received / fix
format. A zero-wage row is the most common product-side-sign-convention
leak into labor-side data and is caught immediately.

Product-side models (:class:`~pyRVtest.Bertrand`,
:class:`~pyRVtest.Cournot`, :class:`~pyRVtest.Monopoly`,
:class:`~pyRVtest.MixCournotBertrand`,
:class:`~pyRVtest.PartialCollusion`) are rejected at init under
``market_side='labor'``. :class:`~pyRVtest.PerfectCompetition` and
:class:`~pyRVtest.CustomConductModel` are accepted on both sides.

The :class:`pyRVtest.backends.LaborSupplyBackend` ships as a v0.4
skeleton: constructor and protocol members exist, but
``compute_jacobian`` / ``compute_hessian`` raise
:class:`NotImplementedError`. Users who need a working labor-supply
backend in v0.4 should wrap their own with
:class:`~pyRVtest.backends.UserSuppliedBackend`.

See :doc:`agent_guide` for the longer narrative and the full protocol
surface.

``models=`` vs ``model_formulations=``
--------------------------------------

``Problem`` accepts either keyword, but not both. If you mix them, you get a
``TypeError``. During the deprecation window (v0.4 and v0.5), both keywords
produce byte-identical results for equivalent inputs; they share the same
internal pipeline after a translation step.

Deprecation timeline
--------------------

* **v0.4 (current):** new class-based API lands. ``ModelFormulation`` emits
  ``DeprecationWarning`` once per Python session on first construction.
  ``model_formulations=`` still accepted.
* **v0.5:** same as v0.4; continued migration window.
* **v0.6:** ``ModelFormulation`` removed. ``model_formulations=`` keyword
  removed from ``Problem``. Only the class-based API works.

Suppressing the warning
-----------------------

If you aren't ready to migrate and want to suppress the warning during your
session, use Python's standard filter:

.. code-block:: python

    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='pyRVtest')

We recommend keeping the warning visible; it reminds you that the code you're
relying on will eventually break.

Reporting issues
----------------

If you have a ``ModelFormulation`` configuration that doesn't translate cleanly
to the new API, please open an issue. We can either add an example here or
extend the class-based API to cover it.
