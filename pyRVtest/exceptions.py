"""Custom exception hierarchy for pyRVtest.

The hierarchy is deliberately minimal. Only categories used by 2+
call sites are kept as distinct classes. Every custom class also
inherits from a Python built-in (``ValueError``, ``RuntimeError``) so
that existing callers using ``except ValueError:`` and test assertions
using ``pytest.raises(ValueError, ...)`` continue to work unchanged.

Hierarchy
---------

- :class:`PyRVTestError` — root for anything raised by pyRVtest.
- :class:`ValidationError` (subclasses ``ValueError``) — user-supplied
  input failed a validation check.
    - :class:`InstrumentDataError` — Z / instrument-formulation
      validation.
- :class:`BackendError` (subclasses ``RuntimeError``) — demand-backend
  setup or capability failure.
    - :class:`DemandBackendError` (also subclasses ``ValueError``) —
      raised when a backend lacks the inputs needed for a requested
      computation (e.g., demand-adjustment state). Multi-inherits
      ``ValueError`` because existing tests assert
      ``pytest.raises(ValueError, ...)``.
    - :class:`HessianUnavailableError` (also subclasses ``ValueError``)
      — raised when vertical-integration passthrough asks the backend
      for a Hessian and the backend has none (e.g.,
      ``UserSuppliedBackend`` constructed without ``hessian_fn``).

Notes
-----

Classes we considered but dropped because only 0 or 1 call site used
them:

- ``InsufficientMarketDataError`` — only the product-data share-sum
  check touched on this (single site); kept as plain ``ValueError``
  with the expected / received / fix format.
- ``ModelSpecificationError`` — the conduct-model validators all
  currently raise ``TypeError``. A single subclass cannot cleanly
  satisfy both ``ValueError`` and ``TypeError`` assertions across the
  test suite without overloading semantics, so the raised type is
  left as ``TypeError`` and the hierarchy does not introduce a
  conflicting parallel.
"""

from __future__ import annotations


__all__ = [
    'PyRVTestError',
    'ValidationError',
    'InstrumentDataError',
    'BackendError',
    'DemandBackendError',
    'HessianUnavailableError',
]


class PyRVTestError(Exception):
    """Base class for every exception raised by pyRVtest."""


class ValidationError(PyRVTestError, ValueError):
    """User-supplied input failed a validation check.

    Subclasses ``ValueError`` so existing callers using
    ``except ValueError:`` continue to work.
    """


class InstrumentDataError(ValidationError):
    """Instrument columns (``Z``) or instrument_formulation failed validation."""


class BackendError(PyRVTestError, RuntimeError):
    """Demand-backend setup or capability failure.

    Subclasses ``RuntimeError`` rather than ``ValueError`` because the
    error is about a backend not being able to perform a requested
    computation, not about a user input failing a shape check.
    """


class DemandBackendError(BackendError, ValueError):
    """Backend was constructed without the state needed for a call path.

    Multi-inherits ``ValueError`` in addition to :class:`BackendError`
    because ``tests/test_backends.py`` asserts
    ``pytest.raises(ValueError, ...)`` on the existing raise sites.
    """


class HessianUnavailableError(BackendError, ValueError):
    """Backend cannot supply a Hessian for a request that requires one.

    Example: vertical-integration passthrough needs a Hessian; a
    ``UserSuppliedBackend`` constructed without ``hessian_fn`` returns
    ``None`` from ``compute_hessian``, which surfaces as this error.

    Multi-inherits ``ValueError`` because
    ``tests/test_backends.py::test_missing_hessian_raises`` and
    ``tests/test_passthrough.py`` assert
    ``pytest.raises(ValueError, match="returned None"...)``.
    """
