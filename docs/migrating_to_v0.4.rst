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
7. Tax and cost-scaling kwargs (``unit_tax``, ``advalorem_tax``, ``advalorem_payer``, ``cost_scaling``, ``user_supplied_markups``) have the same names on the new classes.

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

Tax kwargs (``unit_tax``, ``advalorem_tax``, ``advalorem_payer``,
``cost_scaling``) keep the same names on the new classes. For single-tier
conduct they live on the conduct class; for vertical they live on the
``Vertical`` wrapper.

**Before:**

.. code-block:: python

    pyRVtest.ModelFormulation(
        model_downstream='bertrand',
        ownership_downstream='firm_ids',
        advalorem_tax='tax_rate_col',
        advalorem_payer='firm',
        cost_scaling='scale_col',
    )

**After:**

.. code-block:: python

    pyRVtest.Bertrand(
        ownership='firm_ids',
        advalorem_tax='tax_rate_col',
        advalorem_payer='firm',
        cost_scaling='scale_col',
    )

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
