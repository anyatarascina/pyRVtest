.. currentmodule:: pyRVtest


API Documentation
=================

The majority of the package consists of classes, which compartmentalize different aspects of testing models of firm
conduct.

There are some convenience functions as well.


Configuration Classes
---------------------

.. autosummary::
   :toctree: _api
   :template: class_without_methods.rst

   Formulation
   ModelFormulation


Data Manipulation Functions
---------------------------

There are also a number of convenience functions that can be used to compute markups, or manipulate other pyRVtest
objects.

.. autosummary::
   :toctree: _api

   build_ownership
   build_markups
   construct_passthrough_matrix
   evaluate_first_order_conditions
   read_pickle


Problem Class
-------------

.. autosummary::
   :toctree: _api
   :template: class_with_signature.rst

    Problem

Once initialized, the following method solves the problem.

.. autosummary::
   :toctree: _api

   Problem.solve


Problem Results Class
---------------------

Solved problems return the following results class.

.. autosummary::
   :nosignatures:
   :toctree: _api
   :template: class_without_signature.rst

   ProblemResults

The results can be pickled or converted into a dictionary.

.. autosummary::
   :toctree: _api

   ProblemResults.to_pickle


Exceptions
----------

Starting in v0.4, pyRVtest raises a small hierarchy of custom exceptions
for common validation and backend failures. Every custom class
subclasses a Python built-in (``ValueError`` or ``RuntimeError``) so
that existing callers using ``except ValueError:`` continue to work
unchanged.

Error messages follow an **expected / received / fix** structure: what
the check was looking for, what was actually there, and a concrete
next step. Internal-invariant failures are prefixed with ``pyRVtest
internal error:`` and kept terse.

.. autosummary::
   :toctree: _api
   :template: class_without_methods.rst

   PyRVTestError
   ValidationError
   InstrumentDataError
   BackendError
   DemandBackendError
   HessianUnavailableError
