r"""Global options.

Attributes
----------
digits : `int`
    Number of significant digits used when floats are rendered in
    :meth:`~pyRVtest.ProblemResults.to_markdown` and
    :meth:`~pyRVtest.ProblemResults.to_latex` output, and in
    :meth:`~pyRVtest.PanelResults.to_markdown` /
    :meth:`~pyRVtest.PanelResults.to_latex`. The default is ``6``; change
    with ``pyRVtest.options.digits = 4``.

    The older ``format_table`` text-mode summary used by ``print(results)``
    applies its own per-column rounding (3 digits for TRV, 1 for F,
    3 for MCS p-values) and is not affected by this option; the ``digits``
    option controls the DataFrame-backed exports only.

verbose : `bool`
    *Deprecated since v0.4; removed in v0.6.* In v0.3, this gated the
    legacy :func:`~pyRVtest.output.output` helper. v0.4 migrated the
    package to the standard :mod:`logging` module; this attribute no
    longer has any effect. To control pyRVtest output, configure the
    ``pyRVtest`` logger directly::

        import logging
        logging.getLogger('pyRVtest').setLevel(logging.WARNING)
        # or INFO / DEBUG for more detail

    Reading ``pyRVtest.options.verbose`` emits a ``DeprecationWarning``.
    Assignment is accepted silently for backwards compatibility but has
    no effect.

verbose_tracebacks : `bool`
    Whether to include full tracebacks in error messages. By default, full tracebacks are turned off. These can be
    useful when attempting to find the source of an error message. Tracebacks can be turned on with
    ``pyRVtest.options.verbose_tracebacks = True``.
verbose_output : `callable`
    Function used to output status updates. The default function is simply ``print``. The function can be changed, for
    example, to include an indicator that statuses are from this package, with
    ``pyRVtest.verbose_output = lambda x: print(f"pyRVtest: {x}")``.
flush_output : `bool`
    Whether to call ``sys.stdout.flush()`` after outputting a status update. By default, output is not flushed to
    standard output. To force standard output flushes after every status update, set
    ``pyRVtest.options.flush_output = True``.
dtype : `dtype`
    The data type used for internal calculations, which is by default ``numpy.float64``. The other recommended option is
    ``numpy.longdouble``, which is the only extended precision floating point type currently supported by NumPy.
    Although this data type will be used internally, ``numpy.float64`` will be used when passing arrays to optimization
    and fixed point routines, which may not support extended precision. The library underlying :mod:`scipy.linalg`,
    which is used for matrix inversion, may also use ``numpy.float64``.

    One instance in which extended precision can be helpful in the BLP problem is when there are a large number of near
    zero choice probabilities with small integration weights, which, under standard precision are called zeros when in
    aggregate they are nonzero. For example, :ref:`references: Skrainka (2012)` finds that using long doubles is
    sufficient to solve many utility floating point problems.

    The precision of ``numpy.longdouble`` depends on the platform on which NumPy is installed. If the platform in use
    does not support extended precision, using ``numpy.longdouble`` may lead to unreliable results. For example, on
    Windows, NumPy is usually compiled such that ``numpy.longdouble`` often behaves like ``numpy.float64``. Precisions
    can be compared with :class:`numpy.finfo` by running ``numpy.finfo(numpy.float64)`` and
    ``numpy.finfo(numpy.longdouble)``. For more information, refer to
    `this discussion <https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html#extended-precision>`_.

    If extended precisions is supported, the data type can be switched with ``pyblp.options.dtype = numpy.longdouble``.
    On Windows, it is often easier to install Linux in a virtual machine than it is to build NumPy from source with a
    non-standard compiler.

finite_differences_epsilon : `float`
    Perturbation :math:`\epsilon` used to numerically approximate derivatives with central finite differences:

    .. math:: f'(x) = \frac{f(x + \epsilon / 2) - f(x - \epsilon / 2)}{\epsilon}.

    By default, this is the square root of the machine epsilon: ``numpy.sqrt(numpy.finfo(options.dtype).eps)``. The
    typical example where this is used is when computing the Hessian, but it may also be used to compute Jacobians
    required for standard errors when analytic gradients are disabled.

pseudo_inverses : `bool`
    Whether to compute Moore-Penrose pseudo-inverses of matrices with :func:`scipy.linalg.pinv` instead of their classic
    inverses with :func:`scipy.linalg.inv`. This is by default ``True``, so pseudo-inverses will be used. Up to small
    numerical differences, the pseudo-inverse is identical to the classic inverse for invertible matrices. Using the
    pseudo-inverse by default can help alleviate problems from, for example, near-singular weighting matrices.

    To always attempt to compute classic inverses first, set ``pyblp.options.pseudo_inverses = False``. If a classic
    inverse cannot be computed, an error will be displayed, and a pseudo-inverse may be computed instead.

collinear_atol : `float`
    Absolute tolerance for detecting collinear columns in each matrix of product characteristics and instruments:
    :math:`X_1`, :math:`X_2`, :math:`X_3`, :math:`Z_D`, and :math:`Z_S`.

    Each matrix is decomposed into a :math:`QR` decomposition and an error is raised for any column whose diagonal
    element in :math:`R` has a magnitude less than ``collinear_atol + collinear_rtol * sd`` where ``sd`` is the column's
    standard deviation.

    The default absolute tolerance is ``1e-14``. To disable collinearity checks, set
    ``pyblp.options.collinear_atol = pyblp.options.collinear_rtol = 0``.


collinear_rtol : `float`
    Relative tolerance for detecting collinear columns, which is by default also ``1e-14``.
psd_atol : `float`
    Absolute tolerance for detecting non-positive semidefinite matrices. For example, this check is applied to any
    custom weighting matrix, :math:`W`.

    Singular value decomposition factorizes the matrix into :math:`U \Sigma V` and an error is raised if any element in
    the original matrix differs in absolute value from :math:`V' \Sigma V` by more than ``psd_atol + psd_rtol * abs``
    where ``abs`` is the element's absolute value.

    The default tolerance is ``1e-8``. To disable positive semidefinite checks, set
    ``pyblp.options.psd_atol = pyblp.options.psd_rtol = numpy.inf``.

psd_rtol : `float`
    Relative tolerance for detecting non-positive definite matrices, which is by default also ``1e-8``.

demand_adjustment_weight : `str`
    Which demand-side GMM weight matrix to use in the DMSS (2024) Appendix C equation (77)
    first-stage correction when ``demand_adjustment=True`` and ``demand_results`` is a pyblp
    results object.

    - ``'W'`` (default): use ``demand_results.W``, the weight matrix actually used in the
      GMM estimation step. For ``method='1s'`` this is the 2SLS weight
      :math:`(Z'Z/N)^{-1}`. For ``method='2s'`` this is the efficient weight that pyblp used
      in step 2. This matches the DMSS Appendix C formulation, which specifies Λ with the
      weight used to estimate θ̂.
    - ``'updated_W'``: use ``demand_results.updated_W``, pyblp's "next-step" efficient
      weight computed from current-step residuals. Provided for backwards comparison; not
      DMSS-consistent because it does not match the weight used to estimate θ̂.

    This option only affects the PyBLP path (when ``demand_results`` is passed). The
    ``demand_params`` analytic path uses :math:`(Z'Z/N)^{-1}` by default; to reproduce a
    different weight there, pass ``demand_params['W_demand']`` explicitly.

Examples
--------
>>> from pyRVtest import options
>>> options.digits
6
>>> options.demand_adjustment_weight in ('W', 'updated_W')
True
>>> import numpy as np
>>> options.dtype is np.float64
True
"""

import warnings as _warnings
from typing import Any as _Any

import numpy as _np


digits = 6
# ``verbose`` is intentionally NOT defined here so reads go through
# ``__getattr__`` (below) and fire a DeprecationWarning. Assignment is
# still supported (populates the module namespace directly); subsequent
# reads then return the assigned value silently, which preserves the
# common legacy pattern ``options.verbose = False`` without noise.
verbose_tracebacks = False
verbose_output = print
flush_output = False
dtype = _np.float64
finite_differences_epsilon = _np.sqrt(_np.finfo(dtype).eps)
pseudo_inverses = True
collinear_atol = collinear_rtol = 1e-14
psd_atol = psd_rtol = 1e-8
ndraws = 99999
random_seed = 1

# Which pyblp weight matrix to use in the DMSS demand-adjustment correction.
# 'W' is the weight actually used in estimation (DMSS-consistent); 'updated_W' is
# pyblp's next-step efficient weight, provided only for backwards comparison.
demand_adjustment_weight = 'W'


def __getattr__(name: str) -> _Any:
    # PEP 562 module-level attribute access. Only called when ``name`` is
    # not already in the module namespace, which is exactly the hook we
    # want for the deprecated ``verbose`` option: reads that have not
    # been preceded by an assignment go through here and emit the
    # deprecation warning.
    if name == 'verbose':
        _warnings.warn(
            "pyRVtest.options.verbose is deprecated since v0.4 and has no "
            "effect (output was migrated to the standard logging module). "
            "Received a read of options.verbose. "
            "Fix: configure the pyRVtest logger directly, e.g. "
            "logging.getLogger('pyRVtest').setLevel(logging.WARNING). "
            "Removal scheduled for v0.6.",
            DeprecationWarning,
            stacklevel=2,
        )
        return True
    raise AttributeError(f"module 'pyRVtest.options' has no attribute {name!r}")
