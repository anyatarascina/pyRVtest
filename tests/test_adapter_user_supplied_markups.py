"""Regression tests for v0.4.0rc1 fix: ``ModelFormulation(user_supplied_markups=...,
ownership_downstream='firm_ids')`` now translates to :class:`~pyRVtest.UserSuppliedMarkups`
instead of asserting.

Background (Lorenzo 2026-04-18 review memo, P0 item 1): the production pattern
used by carRV's ``conduct_test.py`` crashed with an ``AssertionError`` in
``pyRVtest/models/_adapter.py`` after the v0.4 refactor. The CHANGELOG claim
"v0.3 scripts run unchanged modulo one-line deprecation warnings" was false
for this case. These tests pin the fix.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import pyRVtest
from pyRVtest.models._adapter import from_model_formulation


def _make_tiny_dgp(seed: int = 1, T: int = 6, J: int = 2) -> pd.DataFrame:
    """Mirror ``tests/test_problem_state_isolation.py::_make_tiny_dgp`` so the
    fixture is familiar and markups can be precomputed without a pyblp solve.
    """
    rng = np.random.default_rng(seed=seed)
    N = T * J
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)
    alpha = -2.0
    x1 = rng.normal(size=N)
    prices = rng.uniform(0.8, 2.0, size=N)
    u = 0.4 * x1 + rng.normal(scale=0.2, size=N)
    delta = u + alpha * prices
    shares = np.zeros(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        e = np.exp(delta[idx])
        shares[idx] = e / (1.0 + e.sum())
    z1 = rng.normal(size=N) + 1.5
    z2 = rng.normal(size=N)
    markups_m1 = np.zeros(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        s_t = shares[idx]
        fids = firm_ids[idx]
        O = (fids[:, None] == fids[None, :]).astype(float)
        D = alpha * (np.diag(s_t) - np.outer(s_t, s_t))
        markups_m1[idx] = -np.linalg.solve(O * D.T, s_t).flatten()
    markups_m2 = np.zeros(N)
    return pd.DataFrame({
        'market_ids': market_ids,
        'firm_ids': firm_ids,
        'x1': x1,
        'prices': prices,
        'shares': shares,
        'z1': z1, 'z2': z2,
        'markups_m1': markups_m1,
        'markups_m2': markups_m2,
        'cost_shifter': rng.uniform(0.5, 1.5, size=N),
    })


class TestAdapterRouting:
    """``ModelFormulation(user_supplied_markups=...)`` routes to
    :class:`~pyRVtest.UserSuppliedMarkups`."""

    def test_translates_to_user_supplied_markups(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            mf = pyRVtest.ModelFormulation(
                user_supplied_markups='mkup_col',
                ownership_downstream='firm_ids',
            )
        result = from_model_formulation(mf)
        assert isinstance(result, pyRVtest.UserSuppliedMarkups)
        assert result.user_supplied_markups == 'mkup_col'
        assert result.ownership == 'firm_ids'

    def test_tax_kwargs_carry_through(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            mf = pyRVtest.ModelFormulation(
                user_supplied_markups='mkup_col',
                ownership_downstream='firm_ids',
                unit_tax='tax_col',
                advalorem_tax='vat_col',
                advalorem_payer='consumer',
                unit_tax_salient=False,
            )
        result = from_model_formulation(mf)
        assert isinstance(result, pyRVtest.UserSuppliedMarkups)
        assert result.unit_tax == 'tax_col'
        assert result.advalorem_tax == 'vat_col'
        assert result.advalorem_payer == 'consumer'
        assert result.unit_tax_salient is False
        assert result.advalorem_tax_salient is True


class TestLorenzoReproNoCrash:
    """Lorenzo's exact repro (2026-04-18 memo, P0 item 1): this used to crash
    with ``AssertionError: ModelFormulation without model_downstream cannot
    be translated to a ConductModel``. Now it should construct cleanly.
    """

    def test_problem_construction_does_not_crash(self):
        df = _make_tiny_dgp()
        pyRVtest.options.verbose = False
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            problem = pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
                instrument_formulation=pyRVtest.Formulation('0 + z1 + z2'),
                product_data=df,
                demand_results=None,
                model_formulations=(
                    pyRVtest.ModelFormulation(
                        user_supplied_markups='markups_m1',
                        ownership_downstream='firm_ids',
                    ),
                    pyRVtest.ModelFormulation(
                        user_supplied_markups='markups_m2',
                        ownership_downstream='firm_ids',
                    ),
                ),
            )
        # If we got here, no assert fired.
        assert problem is not None


class TestBitParityWithClassAPI:
    """TRV / F from the legacy ModelFormulation path and the new class-based
    path are bit-identical. This is the migration-compat guarantee the
    CHANGELOG promises.
    """

    def test_trv_and_f_match_class_api(self):
        df = _make_tiny_dgp()
        pyRVtest.options.verbose = False

        # Legacy ModelFormulation path (translated to UserSuppliedMarkups
        # internally by the adapter).
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            p_mf = pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
                instrument_formulation=pyRVtest.Formulation('0 + z1 + z2'),
                product_data=df,
                demand_results=None,
                model_formulations=(
                    pyRVtest.ModelFormulation(
                        user_supplied_markups='markups_m1',
                        ownership_downstream='firm_ids',
                    ),
                    pyRVtest.ModelFormulation(
                        user_supplied_markups='markups_m2',
                        ownership_downstream='firm_ids',
                    ),
                ),
            )
        r_mf = p_mf.solve()

        # Class-based equivalent.
        p_cl = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=pyRVtest.Formulation('0 + z1 + z2'),
            product_data=df,
            demand_results=None,
            models=[
                pyRVtest.UserSuppliedMarkups(markups='markups_m1', ownership='firm_ids'),
                pyRVtest.UserSuppliedMarkups(markups='markups_m2', ownership='firm_ids'),
            ],
        )
        r_cl = p_cl.solve()

        # Off-diagonal TRV / F entries are the real test payload; diagonals
        # are NaN by convention. Compare finite entries bit-identically.
        trv_mf = np.array(r_mf.TRV)
        trv_cl = np.array(r_cl.TRV)
        mask = np.isfinite(trv_mf) & np.isfinite(trv_cl)
        assert mask.any(), "No finite TRV entries to compare."
        np.testing.assert_array_equal(trv_mf[mask], trv_cl[mask])

        f_mf = np.array(r_mf.F)
        f_cl = np.array(r_cl.F)
        mask_f = np.isfinite(f_mf) & np.isfinite(f_cl)
        np.testing.assert_array_equal(f_mf[mask_f], f_cl[mask_f])

        # Markups themselves should be exactly the input columns.
        np.testing.assert_array_equal(
            np.array(r_mf.markups[0]).flatten(),
            df['markups_m1'].to_numpy(),
        )

    def test_matches_bertrand_with_user_supplied_markups(self):
        """Historic equivalence: the old ``ModelFormulation(model_downstream='bertrand',
        user_supplied_markups='col', ...)`` path produces the same TRV / F as the
        new ``UserSuppliedMarkups`` path, since the FOC never runs when markups
        are supplied.
        """
        df = _make_tiny_dgp()
        pyRVtest.options.verbose = False

        p_bertrand = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=pyRVtest.Formulation('0 + z1 + z2'),
            product_data=df,
            demand_results=None,
            models=[
                pyRVtest.Bertrand(ownership='firm_ids', user_supplied_markups='markups_m1'),
                pyRVtest.PerfectCompetition(user_supplied_markups='markups_m2'),
            ],
        )
        r_bertrand = p_bertrand.solve()

        p_usm = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=pyRVtest.Formulation('0 + z1 + z2'),
            product_data=df,
            demand_results=None,
            models=[
                pyRVtest.UserSuppliedMarkups(markups='markups_m1', ownership='firm_ids'),
                pyRVtest.UserSuppliedMarkups(markups='markups_m2', ownership='firm_ids'),
            ],
        )
        r_usm = p_usm.solve()

        trv_b = np.array(r_bertrand.TRV)
        trv_u = np.array(r_usm.TRV)
        mask = np.isfinite(trv_b) & np.isfinite(trv_u)
        np.testing.assert_array_equal(trv_b[mask], trv_u[mask])


class TestUserSuppliedMarkupsClass:
    """Construction and import of the new class."""

    def test_exported_at_package_level(self):
        assert hasattr(pyRVtest, 'UserSuppliedMarkups')
        assert pyRVtest.UserSuppliedMarkups is not None

    def test_construct_with_markups_only(self):
        model = pyRVtest.UserSuppliedMarkups(markups='col')
        assert model.user_supplied_markups == 'col'
        assert model.ownership is None
        assert model._model_name == 'user_supplied'

    def test_construct_with_ownership(self):
        model = pyRVtest.UserSuppliedMarkups(markups='col', ownership='firm_ids')
        assert model.ownership == 'firm_ids'

    def test_rejects_non_string_markups(self):
        with pytest.raises(TypeError, match='non-empty string'):
            pyRVtest.UserSuppliedMarkups(markups=123)

    def test_rejects_empty_string_markups(self):
        with pytest.raises(TypeError, match='non-empty string'):
            pyRVtest.UserSuppliedMarkups(markups='')

    def test_rejects_duplicate_user_supplied_markups_kwarg(self):
        with pytest.raises(TypeError, match='both markups='):
            pyRVtest.UserSuppliedMarkups(
                markups='col', user_supplied_markups='other_col',
            )

    def test_compute_markup_raises(self):
        """_compute_markup is unreachable in normal use but must raise if
        called directly (matches CustomConductModel pattern)."""
        model = pyRVtest.UserSuppliedMarkups(markups='col')
        with pytest.raises(NotImplementedError, match='bypassing FOC dispatch'):
            model._compute_markup(
                np.eye(2), np.eye(2), np.array([0.3, 0.3]),
            )

    def test_markup_derivative_raises(self):
        model = pyRVtest.UserSuppliedMarkups(markups='col')
        with pytest.raises(NotImplementedError, match='no demand-parameter'):
            model._markup_derivative(
                np.eye(2), np.eye(2), np.zeros((2, 2)),
                np.array([0.3, 0.3]), np.array([0.1, 0.1]),
            )


class TestAdapterRejectsFullyEmptyFormulation:
    """Defensive: if somehow a ModelFormulation reaches the adapter with both
    model_downstream and user_supplied_markups unset, raise a user-facing
    ValidationError (not an internal assert).
    """

    def test_raises_validation_error(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            mf = pyRVtest.ModelFormulation(
                user_supplied_markups='mkup',
                ownership_downstream='firm_ids',
            )
        # Mutate post-construction to hit the adapter's defensive branch.
        mf._user_supplied_markups = None
        mf._model_downstream = None
        with pytest.raises(pyRVtest.ValidationError, match='either model_downstream'):
            from_model_formulation(mf)
