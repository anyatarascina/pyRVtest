.. _tutorial_reference:

Tutorial
========

Three worked tutorials cover the typical entry points into ``pyRVtest``.
Read them in any order; the first is the most common starting point.

Testing firm conduct end-to-end
-------------------------------

The :doc:`_notebooks/testing_firm_conduct` notebook walks through a
complete conduct test on the Nevo (2000, 2001) cereal data. It
estimates BLP demand with ``pyblp``, hands the demand results to
:class:`~pyRVtest.Problem`, runs the test for a Bertrand-vs-Cournot
comparison under several instrument sets, and shows how to read the
RV statistics, F-stat diagnostics, and Model Confidence Set p-values
in the output table. Read this if you already have demand estimates
(or are willing to run ``pyblp`` on your data) and want to learn the
core ``pyRVtest`` workflow.

In-package demand estimation (logit and nested logit)
-----------------------------------------------------

The :doc:`in_package_demand` page covers the
:class:`~pyRVtest.LogitEstimator` and
:class:`~pyRVtest.NestedLogitEstimator` interfaces. Read this if your
demand system is plain logit or one-level nested logit and you want to
keep the entire pipeline inside ``pyRVtest`` rather than running
``pyblp`` first.

The conduct-model library
-------------------------

The :doc:`_notebooks/model_library` notebook is a reference catalogue
of every conduct model class that ships with ``pyRVtest``: standard
oligopoly (:class:`~pyRVtest.Bertrand`, :class:`~pyRVtest.Cournot`,
:class:`~pyRVtest.Monopoly`, :class:`~pyRVtest.PerfectCompetition`,
:class:`~pyRVtest.MixCournotBertrand`,
:class:`~pyRVtest.PartialCollusion`), the Dearing et al. (2026)
simple-markup models (:class:`~pyRVtest.RuleOfThumb`,
:class:`~pyRVtest.ConstantMarkup`),
:class:`~pyRVtest.UserSuppliedMarkups` for hand-supplied markup
columns, and :class:`~pyRVtest.Vertical` for downstream-upstream
bilateral oligopoly. Read this when you need to know what each class
expects, what its FOC looks like, and how to choose between similar
options.

Beyond the tutorials
--------------------

* :doc:`migrating_to_v0.4` — for users coming from v0.3.
* :doc:`custom_demand` — protocol for plugging in a demand system
  pyRVtest does not estimate natively (anything that is not plain
  logit, one-level nested logit, or pyblp).
* :doc:`api` — full API reference.

.. toctree::
   :hidden:

   _notebooks/testing_firm_conduct.ipynb
   _notebooks/model_library.ipynb
