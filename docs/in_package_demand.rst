In-package demand estimation
============================

When to use it
--------------

For **plain logit** or **one-level nested-logit** demand, pyRVtest can
estimate the demand-side parameters directly via 2SLS, so you do not
need to run PyBLP separately. The estimators take the same product-data
shape that the rest of pyRVtest expects, return a populated
``demand_params`` dict, and integrate with :class:`pyRVtest.Problem`
both as a standalone preprocessing step and as an inline shortcut on
``Problem`` itself.

Use this path when:

* Your demand system is logit (no random coefficients) or one-level
  nested logit.
* You have suitable excluded instruments (cost shifters, BLP-style
  rival aggregates, or a within-nest IV for nested logit).
* You want a self-contained pipeline without a separate PyBLP solve.

Use PyBLP and pass ``demand_results`` to :class:`~pyRVtest.Problem`
when you need random coefficients, micro-moments, multi-level nesting,
or any other feature outside the linear-2SLS scope.

Math
----

Both estimators apply the standard Berry (1994) inversion and run
2SLS on the resulting linear regression.

**Plain logit** estimates :math:`(\alpha, \beta)` in

.. math::

   \log s_{jt} - \log s_{0t} = X_{jt}' \beta + \alpha p_{jt} + \xi_{jt}

with prices :math:`p_{jt}` as the only endogenous regressor.

**Nested logit (one level)** estimates :math:`(\alpha, \beta, \rho)` in

.. math::

   \log s_{jt} - \log s_{0t}
       = X_{jt}'\beta + \alpha p_{jt} + \rho \log s_{j|g_t} + \xi_{jt}

with two endogenous regressors (:math:`p_{jt}` and :math:`\log s_{j|g}`).
At least two excluded instruments are required for the order condition.

The default 2SLS weight matrix is :math:`(W'W/N)^{-1}` with
:math:`W = [X, Z_{\text{excluded}}]`, the standard textbook choice.
Pass ``W_demand=`` to override.

Plain logit: standalone walkthrough
-----------------------------------

Using the synthetic dataset shipped with the package (see
:func:`pyRVtest.data.simulate_example` for the data-generating
process — perfect-competition truth, logit demand, two single-product
firms, 3000 markets):

.. code-block:: python

   import pyRVtest

   data = pyRVtest.data.load_example()

   estimator = pyRVtest.LogitEstimator(
       product_data=data,
       formulation_X=pyRVtest.Formulation('1 + x1'),
       formulation_Z=pyRVtest.Formulation('0 + z1'),
   )
   demand_params = estimator.solve()

The returned dict carries everything :class:`~pyRVtest.Problem` needs to
construct the demand backend internally::

   {
       'alpha': -0.9986,                          # price coefficient
       'beta': array([2.0149, 0.9955]),           # coefficients on X (constant, x1)
       'rho': array([], dtype=float64),           # empty for plain logit
       'x_columns': ['1', 'x1'],                  # parsed from formulation_X
       'demand_instrument_columns': ['z1', '1', 'x1'],  # full W matrix
       'W_demand': array(...),                    # 3x3 (W'W/N)^-1
   }

DGP truth was :math:`(\alpha, \beta_0, \beta_1) = (-1, 2, 1)`; the
estimator recovers each within standard 2SLS noise at this sample
size.

Pass the populated dict to :class:`~pyRVtest.Problem`:

.. code-block:: python

   results = pyRVtest.Problem(
       cost_formulation=pyRVtest.Formulation('1 + z1 + z2'),
       instrument_formulation=pyRVtest.Formulation('0 + rival_z1 + rival_z2'),
       models=[
           pyRVtest.Bertrand(ownership='firm_ids'),
           pyRVtest.Cournot(ownership='firm_ids'),
           pyRVtest.Monopoly(),
           pyRVtest.PerfectCompetition(),
       ],
       product_data=data,
       demand_params=demand_params,
   ).solve(demand_adjustment=False)
   print(results)

This is the same example as the README quick-start; see that section
for the full output table and a reading of the test results.

Plain logit: inline shortcut
----------------------------

If you do not need the fitted estimator object, the standalone
estimator can be inlined into :class:`~pyRVtest.Problem`. Set
``demand_params['estimate']`` to ``'logit'`` and pass the formulations
in the same dict; ``Problem`` runs the estimator internally before
constructing the demand backend.

.. code-block:: python

   results = pyRVtest.Problem(
       cost_formulation=pyRVtest.Formulation('1 + z1 + z2'),
       instrument_formulation=pyRVtest.Formulation('0 + rival_z1 + rival_z2'),
       models=[
           pyRVtest.Bertrand(ownership='firm_ids'),
           pyRVtest.Cournot(ownership='firm_ids'),
           pyRVtest.Monopoly(),
           pyRVtest.PerfectCompetition(),
       ],
       product_data=data,
       demand_params={
           'estimate': 'logit',
           'formulation_X': pyRVtest.Formulation('1 + x1'),
           'formulation_Z': pyRVtest.Formulation('0 + z1'),
       },
   ).solve(demand_adjustment=False)

Output is bit-identical to the standalone path. Use whichever reads
more naturally for your script. Standalone is preferable when you want
to inspect the fitted estimator object, reuse one ``demand_params``
dict across several :class:`~pyRVtest.Problem` instances, or supply a
custom ``W_demand``.

Nested logit (one level)
------------------------

The nested-logit estimator adds two pieces relative to the plain-logit
case: a ``nesting_ids_column`` argument identifying each product's
nest, and the requirement that ``formulation_Z`` provide enough
instruments to identify both endogenous regressors (price and the
within-nest log-share).

The shipped example dataset has only two single-product firms, so
there is no meaningful nesting structure to demonstrate. For the
remainder of this section, assume ``data`` is a frame whose schema
matches :func:`pyRVtest.data.load_example`'s but with an additional
integer ``nesting_ids`` column and enough products per market to
populate at least two nests. The
``tests/test_nested_logit_estimator.py::_simulate_nested_logit_data``
helper in the source tree generates a valid nested-logit DGP for
copy-and-paste.

.. code-block:: python

   estimator = pyRVtest.NestedLogitEstimator(
       product_data=data,
       formulation_X=pyRVtest.Formulation('1 + x1'),
       formulation_Z=pyRVtest.Formulation('0 + z1 + z2'),
       nesting_ids_column='nesting_ids',
   )
   demand_params = estimator.solve()

The returned dict is the same shape as the logit case plus a populated
``rho`` and a ``nesting_ids_columns`` entry. Hand it to
:class:`~pyRVtest.Problem` exactly as you would the logit
``demand_params``.

Auto-constructed within-nest IV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standard within-nest instrument — the count of products in each
product's own nest — can be added automatically. With
``auto_construct_within_share_iv=True``, the estimator appends a
``count_in_nest_iv`` column to the instrument set, so
``formulation_Z`` can be a single cost shifter:

.. code-block:: python

   estimator = pyRVtest.NestedLogitEstimator(
       product_data=data,
       formulation_X=pyRVtest.Formulation('1 + x1'),
       formulation_Z=pyRVtest.Formulation('0 + z1'),
       nesting_ids_column='nesting_ids',
       auto_construct_within_share_iv=True,
   )
   demand_params = estimator.solve()

The estimator stores an augmented copy of ``product_data`` (with the
new column appended) on its ``product_data`` attribute, so you can
hand that augmented frame to :class:`~pyRVtest.Problem` if you need
the column visible downstream.

Inline shortcut for nested logit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same configuration via the inline shortcut:

.. code-block:: python

   results = pyRVtest.Problem(
       cost_formulation=pyRVtest.Formulation('1 + z1 + z2'),
       instrument_formulation=pyRVtest.Formulation('0 + rival_z1 + rival_z2'),
       models=[pyRVtest.Bertrand(ownership='firm_ids'),
               pyRVtest.PerfectCompetition()],
       product_data=data,
       demand_params={
           'estimate': 'nested_logit',
           'formulation_X': pyRVtest.Formulation('1 + x1'),
           'formulation_Z': pyRVtest.Formulation('0 + z1'),
           'nesting_ids_column': 'nesting_ids',
           'auto_construct_within_share_iv': True,
       },
   ).solve(demand_adjustment=False)

Sanity checks the estimators perform
------------------------------------

The estimators issue informative warnings or errors when input does not
match the expected shape:

* If :math:`\hat\alpha > 0` (a positive estimated price coefficient,
  almost always a sign of a misspecified IV), :class:`~pyRVtest.LogitEstimator`
  emits a ``UserWarning``.
* If :math:`\hat\rho \notin [0, 1)`, :class:`~pyRVtest.NestedLogitEstimator`
  emits a ``UserWarning``. The estimator does **not** auto-clip;
  decide for yourself whether to investigate or override.
* Missing columns, rank-deficient instrument matrices, and
  underidentified nested-logit configurations raise
  :class:`~pyRVtest.exceptions.ValidationError` at construction time.

When to prefer PyBLP
--------------------

The in-package estimators handle two specific demand systems
exceptionally well precisely because they are the two cases where
linear 2SLS suffices. For anything richer, use PyBLP and pass the
resulting ``demand_results`` to :class:`~pyRVtest.Problem`:

* **Random coefficients** (BLP-style demand): PyBLP only.
* **Micro-moments**: PyBLP only.
* **Multi-level nested logit**: PyBLP only; pyRVtest's
  :class:`~pyRVtest.NestedLogitEstimator` handles the one-level case that covers
  most applications.

The :doc:`tutorial` notebook walks through the PyBLP path end-to-end.
