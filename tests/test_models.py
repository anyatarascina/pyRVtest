"""Unit tests for the class-based ConductModel API (v0.4 step 5a).

These tests validate each class's ``_compute_markup`` and
``_markup_derivative`` against the legacy string-dispatch paths in
``pyRVtest/markups.py::evaluate_first_order_conditions`` and
``pyRVtest/solve/demand_adjustment.py::_analytical_markup_derivative``.

At step 5a the classes exist but are not yet wired into ``Problem``.
Step 5b wires them; step 5c keeps ``ModelFormulation`` working as an
alias. The tests here exercise only the math in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRVtest.markups import evaluate_first_order_conditions
from pyRVtest.models import (
    Bertrand,
    ConductModel,
    Cournot,
    CustomConductModel,
    MixCournotBertrand,
    Monopoly,
    PartialCollusion,
    PerfectCompetition,
    Vertical,
)
from pyRVtest.solve.demand_adjustment import _analytical_markup_derivative


def _random_market(rng, J=4, alpha=-2.0):
    """Synthetic one-market inputs: logit-consistent shares, Jacobian, ownership."""
    raw = rng.uniform(0.5, 2.0, size=J)
    s = 0.5 * raw / raw.sum()  # sums to 0.5 < 1 (outside good share = 0.5)
    D = alpha * (np.diag(s) - np.outer(s, s))
    firm_ids = np.tile(np.arange(J // 2), 2)[:J]
    O = (firm_ids[:, None] == firm_ids[None, :]).astype(float)
    return O, D, s, alpha, firm_ids


def _dispatch_markup(model_str, O, D, s, mix_flag=None):
    """Call the legacy string-dispatch evaluate_first_order_conditions
    to produce a reference markup for comparison.
    """
    J = len(s)
    index = np.arange(J)
    markups_store = np.zeros((J, 1))
    markups_store, _ = evaluate_first_order_conditions(
        index, model_str, O, D, s, markups_store,
        custom_model_specification=None,
        markup_type='downstream', type_mix_flag=mix_flag,
    )
    return markups_store[index, :]


# ---------------------------------------------------------------------------
# Base class guards
# ---------------------------------------------------------------------------

class TestConductModelBase:
    def test_abstract_compute_markup_raises(self):
        cm = ConductModel(ownership='firm_ids')
        rng = np.random.default_rng(0)
        O, D, s, _, _ = _random_market(rng)
        with pytest.raises(NotImplementedError):
            cm._compute_markup(O, D, s)

    def test_abstract_markup_derivative_raises(self):
        cm = ConductModel(ownership='firm_ids')
        rng = np.random.default_rng(0)
        O, D, s, _, _ = _random_market(rng)
        mu = rng.normal(size=len(s))
        with pytest.raises(NotImplementedError):
            cm._markup_derivative(O, D, D.copy(), s, mu)

    def test_validation_advalorem_without_payer_raises(self):
        with pytest.raises(TypeError, match="advalorem_payer"):
            ConductModel(advalorem_tax='tax_col')

    def test_validation_invalid_payer_raises(self):
        with pytest.raises(TypeError, match="'firm' or 'consumer'"):
            ConductModel(advalorem_tax='tax_col', advalorem_payer='whoever')


# ---------------------------------------------------------------------------
# Bertrand
# ---------------------------------------------------------------------------

class TestBertrand:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_compute_markup_matches_string_dispatch(self, seed):
        rng = np.random.default_rng(seed)
        O, D, s, _, _ = _random_market(rng)
        m_class = Bertrand(ownership='firm_ids')._compute_markup(O, D, s)
        m_disp = _dispatch_markup('bertrand', O, D, s)
        np.testing.assert_allclose(m_class, m_disp, atol=1e-14)

    def test_markup_derivative_matches_analytical(self):
        """Bertrand derivative matches _analytical_markup_derivative."""
        rng = np.random.default_rng(42)
        O, D, s, alpha, _ = _random_market(rng)
        mu = Bertrand()._compute_markup(O, D, s).flatten()
        # dD/d(alpha) = D / alpha for linear-in-alpha logit.
        dD = D / alpha
        expected = _analytical_markup_derivative(
            'bertrand', O, D, dD, s, mu, mix_flag_m=None,
            idx=np.arange(len(s)), J_t=len(s),
        )
        got = Bertrand()._markup_derivative(O, D, dD, s, mu)
        np.testing.assert_allclose(got, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# Cournot
# ---------------------------------------------------------------------------

class TestCournot:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_compute_markup_matches_string_dispatch(self, seed):
        rng = np.random.default_rng(seed)
        O, D, s, _, _ = _random_market(rng)
        m_class = Cournot(ownership='firm_ids')._compute_markup(O, D, s)
        m_disp = _dispatch_markup('cournot', O, D, s)
        np.testing.assert_allclose(m_class, m_disp, atol=1e-14)

    def test_markup_derivative_matches_analytical(self):
        rng = np.random.default_rng(99)
        O, D, s, alpha, _ = _random_market(rng)
        mu = Cournot()._compute_markup(O, D, s).flatten()
        dD = D / alpha
        expected = _analytical_markup_derivative(
            'cournot', O, D, dD, s, mu, mix_flag_m=None,
            idx=np.arange(len(s)), J_t=len(s),
        )
        got = Cournot()._markup_derivative(O, D, dD, s, mu)
        np.testing.assert_allclose(got, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# Monopoly
# ---------------------------------------------------------------------------

class TestMonopoly:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_compute_markup_matches_string_dispatch(self, seed):
        rng = np.random.default_rng(seed)
        O, D, s, _, _ = _random_market(rng)
        m_class = Monopoly()._compute_markup(O, D, s)
        m_disp = _dispatch_markup('monopoly', O, D, s)
        np.testing.assert_allclose(m_class, m_disp, atol=1e-14)

    def test_markup_derivative_matches_analytical(self):
        rng = np.random.default_rng(7)
        O, D, s, alpha, _ = _random_market(rng)
        mu = Monopoly()._compute_markup(O, D, s).flatten()
        dD = D / alpha
        expected = _analytical_markup_derivative(
            'monopoly', O, D, dD, s, mu, mix_flag_m=None,
            idx=np.arange(len(s)), J_t=len(s),
        )
        got = Monopoly()._markup_derivative(O, D, dD, s, mu)
        np.testing.assert_allclose(got, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# PerfectCompetition
# ---------------------------------------------------------------------------

class TestPerfectCompetition:
    def test_markup_is_zero(self):
        rng = np.random.default_rng(0)
        O, D, s, _, _ = _random_market(rng)
        m = PerfectCompetition()._compute_markup(O, D, s)
        np.testing.assert_array_equal(m, np.zeros_like(m))

    def test_derivative_is_zero(self):
        rng = np.random.default_rng(0)
        O, D, s, _, _ = _random_market(rng)
        d = PerfectCompetition()._markup_derivative(O, D, D.copy(), s, np.zeros(len(s)))
        np.testing.assert_array_equal(d, np.zeros(len(s)))


# ---------------------------------------------------------------------------
# MixCournotBertrand
# ---------------------------------------------------------------------------

class TestMixCournotBertrand:
    def test_mix_flag_required(self):
        with pytest.raises(TypeError, match="mix_flag is required"):
            MixCournotBertrand(mix_flag=None)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_compute_markup_matches_string_dispatch(self, seed):
        rng = np.random.default_rng(seed)
        O, D, s, _, _ = _random_market(rng, J=4)
        mix_flag = np.array([True, True, False, False])
        model = MixCournotBertrand(mix_flag='mix_col')
        m_class = model._compute_markup_with_flag(O, D, s, mix_flag)
        # Legacy dispatch needs type_mix_flag shaped as (J_total,) array with
        # the same slice applied to index.
        m_disp = _dispatch_markup('mix_cournot_bertrand', O, D, s, mix_flag=mix_flag)
        np.testing.assert_allclose(m_class, m_disp, atol=1e-14)

    def test_markup_derivative_matches_analytical(self):
        rng = np.random.default_rng(321)
        O, D, s, alpha, _ = _random_market(rng, J=4)
        mix_flag = np.array([True, True, False, False])
        model = MixCournotBertrand(mix_flag='mix_col')
        mu = model._compute_markup_with_flag(O, D, s, mix_flag).flatten()
        dD = D / alpha
        # Legacy analytical dispatcher takes mix_flag as a "full" array indexed
        # by idx; here idx = arange(J) so passing the per-market flag works.
        expected = _analytical_markup_derivative(
            'mix_cournot_bertrand', O, D, dD, s, mu,
            mix_flag_m=mix_flag, idx=np.arange(len(s)), J_t=len(s),
        )
        got = model._markup_derivative_with_flag(O, D, dD, s, mu, mix_flag)
        np.testing.assert_allclose(got, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# PartialCollusion
# ---------------------------------------------------------------------------

class TestPartialCollusion:
    def test_kappa_required(self):
        with pytest.raises(TypeError, match="kappa_specification is required"):
            PartialCollusion(ownership='firm_ids')

    def test_same_math_as_bertrand_on_same_ownership(self):
        """Same FOC as Bertrand — only the upstream ownership construction differs.

        For this test we supply the same O to both classes and verify the
        markup formulas agree. (Real partial-collusion ownership is built at
        Models setup by pyblp.build_ownership with the kappa_specification.)
        """
        rng = np.random.default_rng(5)
        O, D, s, _, _ = _random_market(rng)
        m_pc = PartialCollusion(
            ownership='firm_ids', kappa_specification='collusion_row',
        )._compute_markup(O, D, s)
        m_bertrand = Bertrand()._compute_markup(O, D, s)
        np.testing.assert_allclose(m_pc, m_bertrand, atol=1e-14)


# ---------------------------------------------------------------------------
# CustomConductModel
# ---------------------------------------------------------------------------

class TestCustomConductModel:
    def test_requires_callable(self):
        with pytest.raises(TypeError, match="markup_fn must be callable"):
            CustomConductModel(markup_fn="not a callable")

    def test_markup_fn_is_invoked(self):
        def my_markup(O, D, s):
            return np.ones((len(s), 1)) * 0.5
        m = CustomConductModel(markup_fn=my_markup)
        rng = np.random.default_rng(0)
        O, D, s, _, _ = _random_market(rng)
        out = m._compute_markup(O, D, s)
        np.testing.assert_array_equal(out, np.full((len(s), 1), 0.5))

    def test_markup_derivative_raises(self):
        m = CustomConductModel(markup_fn=lambda O, D, s: np.zeros((len(s), 1)))
        rng = np.random.default_rng(0)
        O, D, s, _, _ = _random_market(rng)
        with pytest.raises(NotImplementedError, match="finite-diff"):
            m._markup_derivative(O, D, D.copy(), s, np.zeros(len(s)))


# ---------------------------------------------------------------------------
# Vertical composer
# ---------------------------------------------------------------------------

class TestVertical:
    def test_basic_construction(self):
        v = Vertical(
            downstream=Bertrand(ownership='firm_ids'),
            upstream=Monopoly(ownership='manu_ids'),
            vertical_integration='vi_col',
        )
        assert isinstance(v.downstream, Bertrand)
        assert isinstance(v.upstream, Monopoly)
        assert v.vertical_integration == 'vi_col'

    def test_downstream_type_check(self):
        with pytest.raises(TypeError, match="downstream must be a ConductModel"):
            Vertical(downstream='bertrand', upstream=Monopoly())

    def test_upstream_type_check(self):
        with pytest.raises(TypeError, match="upstream must be a ConductModel"):
            Vertical(downstream=Bertrand(), upstream='monopoly')

    def test_rejects_config_on_inner_downstream(self):
        """vertical_integration / taxes belong on the Vertical wrapper, not
        on inner conducts.
        """
        with pytest.raises(TypeError, match="vertical_integration"):
            Vertical(
                downstream=Bertrand(vertical_integration='vi_col'),
                upstream=Monopoly(),
            )

    def test_rejects_advalorem_tax_on_inner_upstream(self):
        with pytest.raises(TypeError, match="advalorem_tax"):
            Vertical(
                downstream=Bertrand(),
                upstream=Monopoly(advalorem_tax='tax_col', advalorem_payer='firm'),
            )

    def test_validation_advalorem_without_payer_raises(self):
        with pytest.raises(TypeError, match="advalorem_payer"):
            Vertical(
                downstream=Bertrand(),
                upstream=Monopoly(),
                advalorem_tax='tax_col',
            )
