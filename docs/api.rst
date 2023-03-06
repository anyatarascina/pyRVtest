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
