"""v0.4 step 14: labor-side Problem hooks and sign validation.

Covers:

- :class:`pyRVtest.Problem` accepts ``market_side='labor'`` with labor-
  side conduct models; rejects product-side models on a labor Problem.
- ``column_names`` override path resolves labor column defaults
  (``'wages'`` / ``'employment_share'``) to user-chosen names.
- Sign validation: negative or zero wages / employment-shares on any
  row raises :class:`pyRVtest.ValidationError` with the expected /
  received / fix format.
- :class:`pyRVtest.ProblemResults` ``__str__`` exposes the labor-side
  terminology (``markdown`` / ``MRP`` / ``wage``) when constructed on a
  labor Problem, keeping product-side output byte-identical.
- The four labor conduct classes (:class:`Monopsony`,
  :class:`BertrandWages`, :class:`CournotEmployment`) compute markdowns
  on hand-built (D, O, s) inputs; :class:`NashBargaining` raises
  :class:`NotImplementedError` with a v0.5 pointer.
- :class:`pyRVtest.backends.LaborSupplyBackend` instantiates;
  ``compute_jacobian`` / ``compute_hessian`` raise
  :class:`NotImplementedError` with a v0.5 pointer.

End-to-end Problem tests use ``user_supplied_markups`` so they do NOT
depend on :class:`LaborSupplyBackend`: the backend skeleton is exercised
separately via its own unit tests. When the labor backend lands in v0.5
this file should grow an end-to-end solve test analogous to
:func:`tests.test_analytical._run_pyrvtest_base`.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import pyRVtest
from pyRVtest.backends import LaborSupplyBackend
from pyRVtest.exceptions import ValidationError


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _build_labor_product_data(
        wage_col='wages', emp_col='employment_share', zero_wage_row=None,
) -> pd.DataFrame:
    """Build a 2-firm x T-market labor DGP with positive wages and employment shares.

    ``zero_wage_row`` optionally forces one wage entry to zero (for sign-
    validation tests). T is large enough for the collinearity diagnostic to
    run on the stacked [w, z0] matrix (needs rows > columns).
    """
    rng = np.random.default_rng(seed=12345)
    T = 30
    n_firms = 2
    N = T * n_firms
    market_ids = np.repeat(np.arange(T), n_firms)
    firm_ids = np.tile(np.arange(n_firms), T)
    wages = 1.0 + rng.uniform(low=0.1, high=1.5, size=N)
    if zero_wage_row is not None:
        wages[zero_wage_row] = 0.0
    # Employment share in (0, 0.5) per row so within-market shares sum < 1.
    employment = rng.uniform(low=0.1, high=0.4, size=N)
    # Plausible markdowns/MRPs for user_supplied_markups path; exact values
    # don't matter for the plumbing tests, only that they're well-defined.
    markdown_m1 = 0.1 * wages
    markdown_m2 = np.zeros(N)  # perfect competition analogue: no markdown
    cost_shifter = rng.standard_normal(N)
    iv0 = rng.standard_normal(N)
    iv1 = rng.standard_normal(N)
    iv2 = rng.standard_normal(N)
    return pd.DataFrame({
        'market_ids': market_ids,
        'firm_ids': firm_ids,
        wage_col: wages,
        emp_col: employment,
        'markdown_m1': markdown_m1,
        'markdown_m2': markdown_m2,
        'cost_shifter': cost_shifter,
        'iv0': iv0,
        'iv1': iv1,
        'iv2': iv2,
    })


# ---------------------------------------------------------------------------
# Problem(market_side='labor'): acceptance + rejection.
# ---------------------------------------------------------------------------

class TestProblemLaborMode:
    def test_labor_mode_accepts_labor_models(self):
        """Labor-mode Problem accepts Monopsony / BertrandWages / CournotEmployment."""
        df = _build_labor_product_data()
        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
            product_data=df,
            models=[
                pyRVtest.Monopsony(user_supplied_markups='markdown_m1'),
                pyRVtest.PerfectCompetition(user_supplied_markups='markdown_m2'),
            ],
            market_side='labor',
        )
        assert problem._market_side == 'labor'
        assert problem._labor_column_names == {'price': 'wages', 'shares': 'employment_share'}

    def test_labor_mode_rejects_product_models(self):
        """Labor Problem with Bertrand raises ValidationError."""
        df = _build_labor_product_data()
        with pytest.raises(ValidationError, match=r"labor-side conduct model"):
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
                instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
                product_data=df,
                models=[pyRVtest.Bertrand(ownership='firm_ids')],
                market_side='labor',
            )

    def test_labor_mode_rejection_error_names_offender(self):
        """Error message enumerates the bad model indexes for fast debug."""
        df = _build_labor_product_data()
        with pytest.raises(ValidationError, match=r"models\[1\]=cournot"):
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
                instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
                product_data=df,
                models=[
                    pyRVtest.Monopsony(user_supplied_markups='markdown_m1'),
                    pyRVtest.Cournot(ownership='firm_ids'),
                ],
                market_side='labor',
            )

    def test_column_names_override(self):
        """column_names={'price': ..., 'shares': ...} rebinds default columns."""
        df = _build_labor_product_data(wage_col='my_wage', emp_col='my_emp')
        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
            product_data=df,
            models=[
                pyRVtest.Monopsony(user_supplied_markups='markdown_m1'),
                pyRVtest.PerfectCompetition(user_supplied_markups='markdown_m2'),
            ],
            market_side='labor',
            column_names={'price': 'my_wage', 'shares': 'my_emp'},
        )
        assert problem._labor_column_names == {'price': 'my_wage', 'shares': 'my_emp'}

    def test_column_names_rejects_bad_keys(self):
        """Unknown column_names keys raise ValidationError with the typo."""
        df = _build_labor_product_data()
        with pytest.raises(ValidationError, match=r"unexpected keys"):
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
                instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
                product_data=df,
                models=[pyRVtest.Monopsony(user_supplied_markups='markdown_m1')],
                market_side='labor',
                column_names={'wage': 'wages'},  # typo: should be 'price'
            )

    def test_column_names_without_labor_mode_raises(self):
        """column_names passed on a product-side Problem raises ValidationError."""
        df = _build_labor_product_data()
        df['prices'] = df['wages']
        df['shares'] = df['employment_share']
        with pytest.raises(ValidationError, match=r"market_side='labor'"):
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
                instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
                product_data=df,
                models=[pyRVtest.Bertrand(ownership='firm_ids')],
                column_names={'price': 'wages'},
            )

    def test_invalid_market_side_raises(self):
        """market_side='product_and_labor' raises immediately."""
        df = _build_labor_product_data()
        with pytest.raises(ValidationError, match=r"'product' or 'labor'"):
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
                instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
                product_data=df,
                models=[pyRVtest.Monopsony(user_supplied_markups='markdown_m1')],
                market_side='product_and_labor',
            )

    def test_product_mode_default_is_byte_identical(self):
        """market_side='product' (default) leaves _labor_column_names=None."""
        # Product-side fixture with 'prices' / 'shares' named columns.
        rng = np.random.default_rng(seed=12345)
        T = 30
        N = 2 * T
        df = pd.DataFrame({
            'market_ids': np.repeat(np.arange(T), 2),
            'firm_ids': np.tile([0, 1], T),
            'prices': 1.0 + rng.uniform(0.1, 1.5, size=N),
            'shares': rng.uniform(0.1, 0.4, size=N),
            'markup_m1': 0.1 * np.ones(N),
            'markup_m2': np.zeros(N),
            'cost_shifter': rng.standard_normal(N),
            'iv0': rng.standard_normal(N),
            'iv1': rng.standard_normal(N),
            'iv2': rng.standard_normal(N),
        })
        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
            product_data=df,
            models=[
                pyRVtest.Bertrand(ownership='firm_ids', user_supplied_markups='markup_m1'),
                pyRVtest.PerfectCompetition(user_supplied_markups='markup_m2'),
            ],
        )
        assert problem._market_side == 'product'
        assert problem._labor_column_names is None


# ---------------------------------------------------------------------------
# Sign validation per plan §4.5.
# ---------------------------------------------------------------------------

class TestLaborSignValidation:
    def test_zero_wage_row_raises(self):
        df = _build_labor_product_data(zero_wage_row=0)
        with pytest.raises(ValidationError) as excinfo:
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
                instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
                product_data=df,
                models=[pyRVtest.Monopsony(user_supplied_markups='markdown_m1')],
                market_side='labor',
            )
        msg = str(excinfo.value)
        # expected / received / fix format from plan §4.5.
        assert 'Expected' in msg
        assert 'Received' in msg
        assert 'Fix' in msg
        assert "'wages'" in msg

    def test_negative_wage_row_raises(self):
        df = _build_labor_product_data()
        df.loc[1, 'wages'] = -0.5
        with pytest.raises(ValidationError, match=r"wages <= 0"):
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
                instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
                product_data=df,
                models=[pyRVtest.Monopsony(user_supplied_markups='markdown_m1')],
                market_side='labor',
            )

    def test_zero_employment_row_raises(self):
        df = _build_labor_product_data()
        df.loc[2, 'employment_share'] = 0.0
        # Products also rejects shares <= 0; ensure we hit the labor-side
        # validator first so the message names the user's
        # 'employment_share' column.
        with pytest.raises(ValidationError) as excinfo:
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
                instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
                product_data=df,
                models=[pyRVtest.Monopsony(user_supplied_markups='markdown_m1')],
                market_side='labor',
            )
        msg = str(excinfo.value)
        assert "'employment_share'" in msg
        assert 'Fix' in msg

    def test_missing_wage_column_raises(self):
        df = _build_labor_product_data().drop(columns=['wages'])
        with pytest.raises(ValidationError, match=r"'wages'"):
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
                instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
                product_data=df,
                models=[pyRVtest.Monopsony(user_supplied_markups='markdown_m1')],
                market_side='labor',
            )

    def test_validation_error_is_value_error(self):
        """Callers using except ValueError still catch the labor validator."""
        df = _build_labor_product_data(zero_wage_row=0)
        with pytest.raises(ValueError):  # ValidationError subclasses ValueError.
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
                instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
                product_data=df,
                models=[pyRVtest.Monopsony(user_supplied_markups='markdown_m1')],
                market_side='labor',
            )


# ---------------------------------------------------------------------------
# ProblemResults label branching.
# ---------------------------------------------------------------------------

def _solve_labor_problem(df) -> pyRVtest.ProblemResults:
    problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
        product_data=df,
        models=[
            pyRVtest.Monopsony(user_supplied_markups='markdown_m1'),
            pyRVtest.PerfectCompetition(user_supplied_markups='markdown_m2'),
        ],
        market_side='labor',
    )
    return problem.solve(demand_adjustment=False, clustering_adjustment=False)


def test_labor_mode_results_string_shows_labor_terms():
    """print(results) under labor mode shows 'markdown' / 'MRP' / 'wage'."""
    df = _build_labor_product_data()
    results = _solve_labor_problem(df)
    assert results._market_side == 'labor'
    out = str(results)
    # Regex assertion per brief: match the three labor-side labels.
    assert re.search(r'markdown', out), f"expected 'markdown' in results str, got:\n{out}"
    assert re.search(r'MRP', out), f"expected 'MRP' in results str, got:\n{out}"
    assert re.search(r'wage', out), f"expected 'wage' in results str, got:\n{out}"


def test_product_mode_results_string_unchanged():
    """Product-side results string uses the classic title (no labor banner).

    Uses the existing analytical base DGP (tests.test_analytical._build_base_dgp)
    rather than a hand-rolled one so we don't duplicate DGP boilerplate and
    inherit its well-tested numerical stability (sqrt-of-variance path).
    """
    from tests.test_analytical import _build_base_dgp, _run_pyrvtest_base
    product_data, _ = _build_base_dgp()
    results = _run_pyrvtest_base(product_data, clustering=False)
    assert results._market_side == 'product'
    out = str(results)
    # Labor-side banner added in step 14d must not leak into product-mode str.
    assert '(labor: markdown / MRP / wage)' not in out
    # Title preserved byte-for-byte except for the classic instrument index.
    assert 'Testing Results - Instruments z0' in out


# ---------------------------------------------------------------------------
# Direct markup math on labor conduct classes.
# ---------------------------------------------------------------------------

class TestLaborConductClasses:
    def test_monopsony_markup(self):
        # Upward-sloping supply: diag(D) positive; markdown = D^{-1} s.
        D = np.array([[2.0, -0.5], [-0.5, 2.0]])
        s = np.array([0.3, 0.3])
        m = pyRVtest.Monopsony()
        mu = m._compute_markup(np.eye(2), D, s)
        expected = np.linalg.solve(D, s.reshape(-1, 1))
        np.testing.assert_allclose(mu, expected, atol=1e-12)
        assert np.all(mu > 0)  # non-trivial labor markdown under upward supply

    def test_bertrand_wages_markup(self):
        D = np.array([[2.0, -0.5], [-0.5, 2.0]])
        O = np.array([[1.0, 0.0], [0.0, 1.0]])  # single-firm employers
        s = np.array([0.3, 0.3])
        mu = pyRVtest.BertrandWages()._compute_markup(O, D, s)
        expected = np.linalg.solve(O * D.T, s.reshape(-1, 1))
        np.testing.assert_allclose(mu, expected, atol=1e-12)

    def test_cournot_employment_markup(self):
        D = np.array([[2.0, -0.5], [-0.5, 2.0]])
        O = np.eye(2)
        s = np.array([0.3, 0.3])
        mu = pyRVtest.CournotEmployment()._compute_markup(O, D, s)
        expected = (O * np.linalg.inv(D)) @ s.reshape(-1, 1)
        np.testing.assert_allclose(mu, expected, atol=1e-12)

    def test_nash_bargaining_raises_not_implemented(self):
        """NashBargaining signals v0.5 deferral with a useful message."""
        nb = pyRVtest.NashBargaining(outside_option='res_wage')
        with pytest.raises(NotImplementedError) as excinfo:
            nb._compute_markup(np.eye(2), np.eye(2), np.array([0.3, 0.3]))
        assert 'v0.5' in str(excinfo.value)
        assert 'v0.4' in str(excinfo.value)  # names the skipped release explicitly

    def test_nash_bargaining_requires_outside_option(self):
        with pytest.raises(TypeError, match=r"outside_option"):
            pyRVtest.NashBargaining(outside_option='')

    def test_sign_flip_vs_product_side(self):
        """Sign convention: labor markdown sign is OPPOSITE of product markup."""
        D_product = np.array([[-2.0, 0.5], [0.5, -2.0]])   # downward demand
        D_labor = -D_product                                # upward supply
        s = np.array([0.3, 0.3])
        bertrand_mu = pyRVtest.Bertrand()._compute_markup(np.eye(2), D_product, s)
        bertrand_wages_md = pyRVtest.BertrandWages()._compute_markup(np.eye(2), D_labor, s)
        np.testing.assert_allclose(bertrand_mu, bertrand_wages_md, atol=1e-12)


# ---------------------------------------------------------------------------
# LaborSupplyBackend skeleton tests.
# ---------------------------------------------------------------------------

class TestLaborSupplyBackendSkeleton:
    def test_instantiates_with_defaults(self):
        b = LaborSupplyBackend(alpha=1.2)
        assert b.n_parameters == 1
        assert b.theta_names == ['alpha']

    def test_nested_logit_shape_exposes_rho(self):
        b = LaborSupplyBackend(alpha=1.2, rho=[0.3])
        assert b.n_parameters == 2
        assert b.theta_names == ['alpha', 'rho[0]']

    def test_compute_jacobian_raises_not_implemented(self):
        b = LaborSupplyBackend(alpha=1.2)
        with pytest.raises(NotImplementedError) as excinfo:
            b.compute_jacobian(market_id=0)
        msg = str(excinfo.value)
        assert 'v0.4' in msg
        assert 'v0.5' in msg  # useful pointer to when it lands
        assert 'UserSuppliedBackend' in msg  # one of the listed workarounds

    def test_compute_hessian_raises_not_implemented(self):
        b = LaborSupplyBackend(alpha=1.2)
        with pytest.raises(NotImplementedError) as excinfo:
            b.compute_hessian(market_id=0)
        assert 'v0.5' in str(excinfo.value)

    def test_perturbed_raises_not_implemented(self):
        b = LaborSupplyBackend(alpha=1.2)
        with pytest.raises(NotImplementedError):
            with b.perturbed(theta_index=0, delta=0.01):
                pass  # pragma: no cover

    def test_skeleton_does_not_implement_supports_demand_adjustment(self):
        """v0.4 skeleton opts out of SupportsDemandAdjustment per plan §4.5."""
        from pyRVtest.backends import SupportsDemandAdjustment
        b = LaborSupplyBackend(alpha=1.2)
        assert not isinstance(b, SupportsDemandAdjustment)

    def test_skeleton_is_demand_backend(self):
        from pyRVtest.backends import DemandBackend
        b = LaborSupplyBackend(alpha=1.2)
        assert isinstance(b, DemandBackend)
