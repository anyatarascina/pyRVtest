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

    # -- Nash-in-(p_B, q_C) first-order conditions (Feenstra-Levinsohn 1995,
    # Proposition 3 / eq. (A22)). Regression guard for the pre-0.4.0b9 sign
    # error, which used D_BB + D_BC D_CC^{-1} D_CB for the Bertrand block.

    @staticmethod
    def _firm_profit_gradient(O, D, s, mu, mix_flag, firm_ids):
        """Numerical FOC residuals for every product under linear demand q = s + D (p - p0).

        Bertrand product j: perturb p_j holding the other Bertrand prices and
        all Cournot *quantities* fixed (Cournot prices re-solve). Cournot
        product j: perturb q_j holding the other Cournot quantities and all
        Bertrand *prices* fixed. Returns d(own-firm profit)/d(own strategic
        variable) at the candidate markups, one entry per product; the
        markups are equilibrium markups iff every entry is zero.
        """
        J = len(s)
        b = mix_flag.astype(bool)
        c = ~b
        p0 = np.ones(J)
        cost = p0 - mu
        D_CC_inv = np.linalg.inv(D[np.ix_(c, c)])

        def profits_at_prices(p):
            q = s + D @ (p - p0)
            return np.array([np.sum((p - cost)[firm_ids == f] * q[firm_ids == f]) for f in firm_ids])

        def prices_from_quantities(q_c_target, p_b):
            # solve q_C(p) = target for p_C given p_B (linear demand => exact)
            p = p0.copy()
            p[b] = p_b
            rhs = q_c_target - s[c] - D[np.ix_(c, b)] @ (p_b - p0[b])
            p[c] = p0[c] + D_CC_inv @ rhs
            return p

        eps = 1e-6
        grad = np.zeros(J)
        for j in range(J):
            if b[j]:
                def f(h):
                    p_b = p0[b].copy()
                    p_b[np.flatnonzero(b) == j] += h
                    return profits_at_prices(prices_from_quantities(s[c], p_b))[j]
            else:
                def f(h):
                    q_c = s[c].copy()
                    q_c[np.flatnonzero(c) == j] += h
                    return profits_at_prices(prices_from_quantities(q_c, p0[b]))[j]
            grad[j] = (f(eps) - f(-eps)) / (2 * eps)
        return grad

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_markups_satisfy_mixed_nash_focs(self, seed):
        rng = np.random.default_rng(seed)
        J = 6
        O, D, s, _, firm_ids = _random_market(rng, J=J)
        # Break the logit symmetry of D so a transpose error would be caught.
        D = D + rng.uniform(-0.05, 0.05, size=(J, J)) * (1 - np.eye(J))
        # Conduct is a firm-level attribute: firms 0 and 1 set prices, firm 2 quantities.
        mix_flag = np.isin(firm_ids, [0, 1])
        model = MixCournotBertrand(mix_flag='mix_col')
        mu = model._compute_markup_with_flag(O, D, s, mix_flag).flatten()
        b, c = mix_flag, ~mix_flag
        grad = self._firm_profit_gradient(O, D, s, mu, mix_flag, firm_ids)
        np.testing.assert_allclose(grad[b], 0.0, atol=1e-8)
        # The Cournot block mirrors ``Cournot._compute_markup`` (``-(O * D^{-1}) @ s``,
        # no transpose), which is exact only for a symmetric Jacobian; check it there.
        O_s, D_s, s_s, _, _ = _random_market(np.random.default_rng(seed), J=J)
        mu_s = model._compute_markup_with_flag(O_s, D_s, s_s, mix_flag).flatten()
        np.testing.assert_allclose(
            self._firm_profit_gradient(O_s, D_s, s_s, mu_s, mix_flag, firm_ids), 0.0, atol=1e-8,
        )
        # The pre-0.4.0b9 formula (plus sign, no transpose) violates the
        # Bertrand players' FOCs.
        D_CC_inv = np.linalg.inv(D[np.ix_(c, c)])
        wrong = mu.copy()
        wrong[b] = np.linalg.solve(
            O[np.ix_(b, b)] * (D[np.ix_(b, c)] @ D_CC_inv @ D[np.ix_(c, b)] + D[np.ix_(b, b)]), -s[b],
        )
        grad_wrong = self._firm_profit_gradient(O, D, s, wrong, mix_flag, firm_ids)
        assert np.abs(grad_wrong[b]).max() > 1e-3

    def test_bertrand_block_reduces_to_bertrand_without_cross_effects(self):
        rng = np.random.default_rng(7)
        O, D, s, _, firm_ids = _random_market(rng, J=4)
        mix_flag = firm_ids == 0
        b, c = mix_flag, ~mix_flag
        D0 = D.copy()
        D0[np.ix_(b, c)] = 0.0
        D0[np.ix_(c, b)] = 0.0
        mu = MixCournotBertrand(mix_flag='m')._compute_markup_with_flag(O, D0, s, mix_flag).flatten()
        mu_bertrand = Bertrand()._compute_markup(O[np.ix_(b, b)], D0[np.ix_(b, b)], s[b]).flatten()
        mu_cournot = Cournot()._compute_markup(O[np.ix_(c, c)], D0[np.ix_(c, c)], s[c]).flatten()
        np.testing.assert_allclose(mu[b], mu_bertrand, atol=1e-14)
        np.testing.assert_allclose(mu[c], mu_cournot, atol=1e-14)

    def test_bertrand_players_markups_exceed_pure_bertrand(self):
        """Feenstra-Levinsohn (1995, p. 31): price setters facing quantity
        setters charge more than under all-Bertrand play."""
        rng = np.random.default_rng(11)
        O, D, s, _, firm_ids = _random_market(rng, J=6)
        mix_flag = np.isin(firm_ids, [0, 1])
        mu = MixCournotBertrand(mix_flag='m')._compute_markup_with_flag(O, D, s, mix_flag).flatten()
        mu_bertrand = Bertrand()._compute_markup(O, D, s).flatten()
        assert np.all(mu[mix_flag] > mu_bertrand[mix_flag])

    def test_single_price_setter_matches_cournot_markup(self):
        """A single-product Bertrand firm whose rivals all set quantities faces
        the same residual demand as a quantity setter, so its markup equals
        its Cournot markup."""
        rng = np.random.default_rng(3)
        O, D, s, _, firm_ids = _random_market(rng, J=6)
        firm_ids = np.array([0, 1, 1, 2, 2, 2])
        O = (firm_ids[:, None] == firm_ids[None, :]).astype(float)
        mix_flag = np.array([True, False, False, False, False, False])
        mu = MixCournotBertrand(mix_flag='m')._compute_markup_with_flag(O, D, s, mix_flag).flatten()
        mu_cournot = Cournot()._compute_markup(O, D, s).flatten()
        np.testing.assert_allclose(mu[0], mu_cournot[0], rtol=1e-12)

    def test_markup_derivative_matches_finite_difference(self):
        rng = np.random.default_rng(5)
        J = 6
        O, D, s, alpha, _ = _random_market(rng, J=J)
        D = D + rng.uniform(-0.05, 0.05, size=(J, J)) * (1 - np.eye(J))
        dD = rng.normal(size=(J, J)) * 0.1
        mix_flag = np.array([True, True, False, False, True, False])
        model = MixCournotBertrand(mix_flag='m')
        mu = model._compute_markup_with_flag(O, D, s, mix_flag).flatten()
        got = model._markup_derivative_with_flag(O, D, dD, s, mu, mix_flag)
        h = 1e-6
        mu_plus = model._compute_markup_with_flag(O, D + h * dD, s, mix_flag).flatten()
        mu_minus = model._compute_markup_with_flag(O, D - h * dD, s, mix_flag).flatten()
        fd = (mu_plus - mu_minus) / (2 * h)
        np.testing.assert_allclose(got, fd, atol=1e-7)


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
