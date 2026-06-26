"""Tests for the standalone precompute functions for the demand adjustment.

``pyRVtest.build_phi_matrix(...)`` and ``pyRVtest.build_markup_derivative(...)``
precompute, from raw ``product_data`` + a fitted demand object (no ``Problem``
needed), the two pieces of the first-stage demand-estimation correction (DMSS
2024 Appendix C eq. 77): the demand-only Phi block and the per-model markup
gradient. They are plugged into
``Problem.solve(demand_adjustment=True, phi_matrix=..., markup_derivative=...)``.

These tests verify that the precomputed path is numerically identical to the
inline path (across both demand backends and multiple instrument sets), that the
two kwargs are independently mixable, that the validation guards fire on
mismatch, and that the objects round-trip through pickle.

The ``logit_dgp_and_estimation`` fixture lives in ``tests/conftest.py``.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyRVtest  # noqa: E402
from pyRVtest.exceptions import ValidationError  # noqa: E402


# ---------------------------------------------------------------------------
# Problem + builder helpers for the two demand backends.
# ---------------------------------------------------------------------------


def _models():
    return [
        pyRVtest.ModelFormulation(
            model_downstream="bertrand", ownership_downstream="firm_ids"
        ),
        pyRVtest.ModelFormulation(model_downstream="perfect_competition"),
    ]


def _common(data, instrument_formulation):
    return dict(
        cost_formulation=pyRVtest.Formulation("1 + z1"),
        instrument_formulation=instrument_formulation,
        model_formulations=_models(),
        product_data=data,
    )


def _two_instrument_sets():
    return [
        pyRVtest.Formulation("0 + rival_x1 + rival_z1"),
        pyRVtest.Formulation("0 + rival_z1 + rival_z1_sq"),
    ]


def _demand_params(pyblp_results, alpha, beta_x):
    beta_0 = float(pyblp_results.beta[pyblp_results.beta_labels.index("1")].item())
    return {
        "alpha": alpha,
        "sigma": [],
        "beta": np.array([beta_0, beta_x]),
        "x_columns": ["intercept", "x1"],
        "demand_instrument_columns": ["rival_x1", "rival_z1_sq", "intercept", "x1"],
    }


def _assert_results_match(r_inline, r_precompute, L, atol=0.0):
    """TRV / F / MCS p-values must match across every instrument set."""
    for l in range(L):
        np.testing.assert_allclose(
            np.asarray(r_inline.TRV[l]), np.asarray(r_precompute.TRV[l]),
            atol=atol, rtol=0.0, err_msg=f"TRV differs for instrument set {l}",
        )
        np.testing.assert_allclose(
            np.asarray(r_inline.F[l]), np.asarray(r_precompute.F[l]),
            atol=atol, rtol=0.0, err_msg=f"F differs for instrument set {l}",
        )
        np.testing.assert_allclose(
            np.asarray(r_inline.MCS_pvalues[l]), np.asarray(r_precompute.MCS_pvalues[l]),
            atol=atol, rtol=0.0, err_msg=f"MCS p-values differ for instrument set {l}",
        )


# ---------------------------------------------------------------------------
# Core equivalence + mixability: precompute == inline, bit-identical.
# ---------------------------------------------------------------------------


class TestEquivalenceBothBackends:
    @pytest.fixture(autouse=True)
    def _quiet(self):
        pyRVtest.options.verbose = False
        yield
        pyRVtest.options.verbose = True

    def test_pyblp_backend_both_objects(self, logit_dgp_and_estimation):
        data, pyblp_results, _, _ = logit_dgp_and_estimation
        problem = pyRVtest.Problem(
            **_common(data, pyRVtest.Formulation("0 + rival_x1 + rival_z1")),
            demand_results=pyblp_results,
        )
        r_inline = problem.solve(demand_adjustment=True)
        phi = pyRVtest.build_phi_matrix(data, demand_results=pyblp_results)
        md = pyRVtest.build_markup_derivative(
            _models(), data, demand_results=pyblp_results
        )
        r_pre = problem.solve(
            demand_adjustment=True, phi_matrix=phi, markup_derivative=md
        )
        _assert_results_match(r_inline, r_pre, L=1, atol=0.0)

    def test_demand_params_backend_both_objects(self, logit_dgp_and_estimation):
        data, pyblp_results, alpha, beta_x = logit_dgp_and_estimation
        dp = _demand_params(pyblp_results, alpha, beta_x)
        problem = pyRVtest.Problem(
            **_common(data, pyRVtest.Formulation("0 + rival_x1 + rival_z1")),
            demand_params=dp,
        )
        r_inline = problem.solve(demand_adjustment=True)
        phi = pyRVtest.build_phi_matrix(data, demand_params=dp)
        md = pyRVtest.build_markup_derivative(_models(), data, demand_params=dp)
        r_pre = problem.solve(
            demand_adjustment=True, phi_matrix=phi, markup_derivative=md
        )
        _assert_results_match(r_inline, r_pre, L=1, atol=0.0)

    def test_phi_only_mixable(self, logit_dgp_and_estimation):
        """Supply only phi_matrix; the markup derivative is computed inline."""
        data, pyblp_results, _, _ = logit_dgp_and_estimation
        problem = pyRVtest.Problem(
            **_common(data, pyRVtest.Formulation("0 + rival_x1 + rival_z1")),
            demand_results=pyblp_results,
        )
        r_inline = problem.solve(demand_adjustment=True)
        phi = pyRVtest.build_phi_matrix(data, demand_results=pyblp_results)
        r_pre = problem.solve(demand_adjustment=True, phi_matrix=phi)
        _assert_results_match(r_inline, r_pre, L=1, atol=0.0)

    def test_markup_derivative_only_mixable(self, logit_dgp_and_estimation):
        """Supply only markup_derivative; the Phi block is computed inline."""
        data, pyblp_results, _, _ = logit_dgp_and_estimation
        problem = pyRVtest.Problem(
            **_common(data, pyRVtest.Formulation("0 + rival_x1 + rival_z1")),
            demand_results=pyblp_results,
        )
        r_inline = problem.solve(demand_adjustment=True)
        md = pyRVtest.build_markup_derivative(
            _models(), data, demand_results=pyblp_results
        )
        r_pre = problem.solve(demand_adjustment=True, markup_derivative=md)
        _assert_results_match(r_inline, r_pre, L=1, atol=0.0)

    def test_reuse_across_multiple_instrument_sets(self, logit_dgp_and_estimation):
        """One precompute, two instrument sets — both match inline."""
        data, pyblp_results, _, _ = logit_dgp_and_estimation
        problem = pyRVtest.Problem(
            **_common(data, _two_instrument_sets()), demand_results=pyblp_results,
        )
        r_inline = problem.solve(demand_adjustment=True)
        phi = pyRVtest.build_phi_matrix(data, demand_results=pyblp_results)
        md = pyRVtest.build_markup_derivative(
            _models(), data, demand_results=pyblp_results
        )
        r_pre = problem.solve(
            demand_adjustment=True, phi_matrix=phi, markup_derivative=md
        )
        _assert_results_match(r_inline, r_pre, L=2, atol=0.0)

    def test_costs_type_log_and_linear(self, logit_dgp_and_estimation):
        """The same precomputed objects work for both linear and log costs.

        Under ``costs_type='log'`` this DGP can yield non-positive implied
        marginal cost, which solve() rejects before the adjustment stage. Either
        way the inline and precompute paths must behave identically.
        """
        data, pyblp_results, _, _ = logit_dgp_and_estimation
        problem = pyRVtest.Problem(
            **_common(data, pyRVtest.Formulation("0 + rival_x1 + rival_z1")),
            demand_results=pyblp_results,
        )
        phi = pyRVtest.build_phi_matrix(data, demand_results=pyblp_results)
        md = pyRVtest.build_markup_derivative(
            _models(), data, demand_results=pyblp_results
        )
        for costs_type in ("linear", "log"):
            inline_err = pre_err = None
            try:
                r_inline = problem.solve(demand_adjustment=True, costs_type=costs_type)
            except ValueError as exc:
                inline_err = str(exc)
            try:
                r_pre = problem.solve(
                    demand_adjustment=True, costs_type=costs_type,
                    phi_matrix=phi, markup_derivative=md,
                )
            except ValueError as exc:
                pre_err = str(exc)
            assert (inline_err is None) == (pre_err is None), (
                f"Paths disagree on feasibility for costs_type={costs_type!r}"
            )
            if inline_err is None:
                _assert_results_match(r_inline, r_pre, L=1, atol=0.0)
            else:
                assert inline_err == pre_err


# ---------------------------------------------------------------------------
# Serialization and float32.
# ---------------------------------------------------------------------------


class TestSerialization:
    @pytest.fixture(autouse=True)
    def _quiet(self):
        pyRVtest.options.verbose = False
        yield
        pyRVtest.options.verbose = True

    def test_pickle_round_trip(self, logit_dgp_and_estimation, tmp_path):
        data, pyblp_results, _, _ = logit_dgp_and_estimation
        problem = pyRVtest.Problem(
            **_common(data, pyRVtest.Formulation("0 + rival_x1 + rival_z1")),
            demand_results=pyblp_results,
        )
        r_inline = problem.solve(demand_adjustment=True)
        phi = pyRVtest.build_phi_matrix(data, demand_results=pyblp_results)
        md = pyRVtest.build_markup_derivative(
            _models(), data, demand_results=pyblp_results
        )
        for obj, fname in ((phi, "phi.pkl"), (md, "md.pkl")):
            with open(tmp_path / fname, "wb") as fh:
                pickle.dump(obj, fh)
        phi2 = pyRVtest.read_pickle(tmp_path / "phi.pkl")
        md2 = pyRVtest.read_pickle(tmp_path / "md.pkl")
        r_pre = problem.solve(
            demand_adjustment=True, phi_matrix=phi2, markup_derivative=md2
        )
        _assert_results_match(r_inline, r_pre, L=1, atol=0.0)

    def test_float32_storage(self, logit_dgp_and_estimation):
        data, pyblp_results, _, _ = logit_dgp_and_estimation
        problem = pyRVtest.Problem(
            **_common(data, pyRVtest.Formulation("0 + rival_x1 + rival_z1")),
            demand_results=pyblp_results,
        )
        r_inline = problem.solve(demand_adjustment=True)
        phi = pyRVtest.build_phi_matrix(
            data, demand_results=pyblp_results, store_float32=True
        )
        md = pyRVtest.build_markup_derivative(
            _models(), data, demand_results=pyblp_results, store_float32=True
        )
        assert phi.dtype == "float32" and phi.h_i.dtype == np.float32
        assert md.dtype == "float32" and md.gradient_markups_raw.dtype == np.float32
        r_pre = problem.solve(
            demand_adjustment=True, phi_matrix=phi, markup_derivative=md
        )
        np.testing.assert_allclose(
            np.asarray(r_inline.TRV[0]), np.asarray(r_pre.TRV[0]),
            atol=1e-4, err_msg="float32 TRV disagreement too large",
        )


# ---------------------------------------------------------------------------
# Validation guards.
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.fixture(autouse=True)
    def _quiet(self):
        pyRVtest.options.verbose = False
        yield
        pyRVtest.options.verbose = True

    @pytest.fixture
    def problem_and_objects(self, logit_dgp_and_estimation):
        data, pyblp_results, _, _ = logit_dgp_and_estimation
        problem = pyRVtest.Problem(
            **_common(data, pyRVtest.Formulation("0 + rival_x1 + rival_z1")),
            demand_results=pyblp_results,
        )
        phi = pyRVtest.build_phi_matrix(data, demand_results=pyblp_results)
        md = pyRVtest.build_markup_derivative(
            _models(), data, demand_results=pyblp_results
        )
        return problem, phi, md

    def test_wrong_type_raises(self, problem_and_objects):
        problem, _, _ = problem_and_objects
        with pytest.raises(TypeError, match="phi_matrix"):
            problem.solve(demand_adjustment=True, phi_matrix=object())
        with pytest.raises(TypeError, match="markup_derivative"):
            problem.solve(demand_adjustment=True, markup_derivative=object())

    def test_object_without_demand_adjustment_flag_raises(self, problem_and_objects):
        problem, phi, md = problem_and_objects
        with pytest.raises(ValueError, match="demand_adjustment=True"):
            problem.solve(demand_adjustment=False, phi_matrix=phi)
        with pytest.raises(ValueError, match="demand_adjustment=True"):
            problem.solve(demand_adjustment=False, markup_derivative=md)

    def test_wrong_dimensions_raise(self, problem_and_objects):
        import dataclasses
        problem, phi, md = problem_and_objects
        bad_phi = dataclasses.replace(phi, N=phi.N + 1)
        with pytest.raises(ValidationError, match="N="):
            problem.solve(demand_adjustment=True, phi_matrix=bad_phi)
        bad_md = dataclasses.replace(md, M=md.M + 1)
        with pytest.raises(ValidationError, match="M="):
            problem.solve(demand_adjustment=True, markup_derivative=bad_md)

    def test_reordered_market_ids_raise(self, problem_and_objects):
        import dataclasses
        problem, phi, md = problem_and_objects
        with pytest.raises(ValidationError, match="market_ids"):
            problem.solve(
                demand_adjustment=True,
                phi_matrix=dataclasses.replace(phi, market_ids_hash="deadbeef"),
            )

    def test_wrong_backend_signature_raises(self, problem_and_objects):
        import dataclasses
        problem, phi, md = problem_and_objects
        with pytest.raises(ValidationError, match="backend"):
            problem.solve(
                demand_adjustment=True,
                markup_derivative=dataclasses.replace(md, backend_signature="Other|N=1"),
            )

    def test_endogenous_cost_with_object_raises(self, problem_and_objects):
        """Supplying a precompute object with endogenous cost raises."""
        problem, phi, _ = problem_and_objects
        problem.endogenous_cost_component = "z1"  # simulate endogenous-cost config
        try:
            with pytest.raises(NotImplementedError, match="endogenous_cost_component"):
                problem.solve(demand_adjustment=True, phi_matrix=phi)
        finally:
            problem.endogenous_cost_component = None
