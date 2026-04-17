"""v0.4 step 10: tests for :class:`pyRVtest.PanelResults`.

Covers the public API:

- constructor validation (non-empty, mapping shape, homogeneous model
  roster);
- mapping-like access (``keys``, ``__getitem__``, ``__iter__``,
  ``__len__``, ``__contains__``);
- :meth:`PanelResults.to_dataframe` long-form aggregation;
- :meth:`PanelResults.rejection_rates` and
  :meth:`PanelResults.summary_df` rejection-rate computation (including
  a hand-computed check against the child ``TRV`` values);
- :meth:`PanelResults.to_latex` and :meth:`PanelResults.to_markdown`
  renderings.

Child :class:`ProblemResults` instances are built via the base DGP
fixture from ``tests/test_analytical.py`` at three seeds, yielding a
3-key panel with homogeneous model roster but independent data. This
means rejection flags vary across keys (which is what we want for the
rate test).
"""

from __future__ import annotations

from typing import Dict, Hashable, List

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pyRVtest import PanelResults, ProblemResults

from .test_analytical import _build_base_dgp, _run_pyrvtest_base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _run_at_seed(seed: int) -> ProblemResults:
    """Run the base DGP at a chosen seed and return ProblemResults."""
    product_data, _ = _build_base_dgp(seed=seed)
    return _run_pyrvtest_base(product_data, clustering=False)


@pytest.fixture(scope='module')
def three_key_panel() -> PanelResults:
    """Panel with 3 keys across 3 seeds — model roster is homogeneous."""
    results: Dict[Hashable, ProblemResults] = {
        ('market_A', 2020): _run_at_seed(12345),
        ('market_A', 2021): _run_at_seed(23456),
        ('market_B', 2020): _run_at_seed(34567),
    }
    return PanelResults(results=results)


@pytest.fixture(scope='module')
def two_key_panel() -> PanelResults:
    """Smaller 2-key panel with string keys."""
    return PanelResults(results={
        'Q1': _run_at_seed(11111),
        'Q2': _run_at_seed(22222),
    })


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstruction:
    """Constructor input validation and success path."""

    def test_happy_path_three_keys(self, three_key_panel):
        assert isinstance(three_key_panel, PanelResults)
        assert len(three_key_panel) == 3

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match='empty'):
            PanelResults(results={})

    def test_non_mapping_raises(self):
        one = _run_at_seed(12345)
        with pytest.raises(TypeError, match='mapping'):
            # A bare list is not a mapping.
            PanelResults(results=[one])  # type: ignore[arg-type]

    def test_non_problemresults_value_raises(self):
        with pytest.raises(TypeError, match='ProblemResults'):
            PanelResults(results={'k': 42})  # type: ignore[dict-item]

    def test_mismatched_model_sets_raises(self):
        """A child with a different number of candidate models must raise.

        We fake the second child rather than solving a 3-model
        Problem — Problem.solve() destabilizes on a 3-model case built
        from only two genuinely distinct markup columns, which is
        orthogonal to what we want to test here. ``PanelResults.__init__``
        only inspects ``len(pr.markups)``, so a lightweight stub
        suffices.
        """
        real = _run_at_seed(12345)

        class _Stub(ProblemResults):  # noqa: D401 — minimal fake
            # Skip the real __init__ entirely; set only what __init__
            # reads (markups length for the model-count check).
            def __init__(self) -> None:  # type: ignore[override]
                self.markups = [np.zeros(1), np.zeros(1), np.zeros(1)]

        stub_three = _Stub()
        assert len(stub_three.markups) == 3  # sanity

        with pytest.raises(ValueError, match='Mismatched model sets'):
            PanelResults(results={'a': real, 'b': stub_three})


# ---------------------------------------------------------------------------
# Mapping-like access
# ---------------------------------------------------------------------------

class TestMappingInterface:
    def test_keys_returns_insertion_order(self, three_key_panel):
        keys = three_key_panel.keys()
        assert keys == [('market_A', 2020), ('market_A', 2021), ('market_B', 2020)]

    def test_len(self, three_key_panel):
        assert len(three_key_panel) == 3

    def test_getitem(self, three_key_panel):
        pr = three_key_panel[('market_A', 2020)]
        assert isinstance(pr, ProblemResults)

    def test_contains(self, three_key_panel):
        assert ('market_A', 2020) in three_key_panel
        assert ('missing', 9999) not in three_key_panel

    def test_iter(self, three_key_panel):
        collected: List[Hashable] = list(iter(three_key_panel))
        assert collected == three_key_panel.keys()


# ---------------------------------------------------------------------------
# Long-form DataFrame export
# ---------------------------------------------------------------------------

class TestToDataFrame:
    def test_returns_dataframe(self, three_key_panel):
        df = three_key_panel.to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_panel_key_column_is_leftmost(self, three_key_panel):
        df = three_key_panel.to_dataframe()
        assert df.columns[0] == 'panel_key'

    def test_row_count(self, three_key_panel):
        # Each child contributes L * M * (M - 1) rows; here L=1, M=2 so
        # 2 ordered pairs per key times 3 keys = 6.
        df = three_key_panel.to_dataframe()
        assert len(df) == 6

    def test_all_keys_present(self, three_key_panel):
        df = three_key_panel.to_dataframe()
        keys_in_frame = {
            tuple(k) if isinstance(k, tuple) else k for k in df['panel_key']
        }
        assert keys_in_frame == set(three_key_panel.keys())


# ---------------------------------------------------------------------------
# Rejection rates: hand-computed check
# ---------------------------------------------------------------------------

class TestRejectionRates:
    def test_returns_dataframe(self, three_key_panel):
        out = three_key_panel.rejection_rates()
        assert isinstance(out, pd.DataFrame)
        assert set(['instrument_set', 'model_i', 'model_j',
                    'rejection_rate', 'n_keys']).issubset(out.columns)

    def test_n_keys_sums_to_panel_size(self, three_key_panel):
        out = three_key_panel.rejection_rates()
        # All children share one instrument set / one unordered pair, so
        # n_keys == 3 on the single output row.
        assert len(out) == 1
        assert int(out['n_keys'].iloc[0]) == 3

    def test_matches_hand_computed_rate(self, three_key_panel):
        """Rejection rate computed manually from child TRV values."""
        alpha = 0.05
        crit = float(norm.ppf(1.0 - alpha / 2.0))
        # Count children whose |TRV[0][0,1]| > crit.
        n_reject = 0
        for key in three_key_panel.keys():
            pr = three_key_panel[key]
            trv = float(np.asarray(pr.TRV[0])[0, 1])
            if np.isfinite(trv) and abs(trv) > crit:
                n_reject += 1
        expected_rate = n_reject / len(three_key_panel)

        out = three_key_panel.rejection_rates(alpha=alpha)
        np.testing.assert_allclose(
            float(out['rejection_rate'].iloc[0]), expected_rate, atol=1e-12
        )

    def test_alpha_affects_rate(self, three_key_panel):
        """A larger alpha (looser critical value) cannot lower rejection."""
        loose = three_key_panel.rejection_rates(alpha=0.50)
        tight = three_key_panel.rejection_rates(alpha=0.001)
        assert float(loose['rejection_rate'].iloc[0]) >= \
               float(tight['rejection_rate'].iloc[0])


# ---------------------------------------------------------------------------
# Wide-form summary
# ---------------------------------------------------------------------------

class TestSummaryDF:
    def test_columns(self, three_key_panel):
        out = three_key_panel.summary_df()
        expected = [
            'instrument_set', 'instrument_set_label',
            'model_i', 'model_j', 'model_i_label', 'model_j_label',
            'rejection_rate', 'n_keys',
        ]
        assert list(out.columns) == expected

    def test_rate_consistent_with_rejection_rates(self, three_key_panel):
        summary = three_key_panel.summary_df()
        rates = three_key_panel.rejection_rates()
        np.testing.assert_allclose(
            summary['rejection_rate'].values,
            rates['rejection_rate'].values,
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# LaTeX and markdown renderings
# ---------------------------------------------------------------------------

class TestToLatex:
    def test_returns_string(self, two_key_panel):
        tex = two_key_panel.to_latex()
        assert isinstance(tex, str)
        assert 'tabular' in tex

    def test_writes_file(self, two_key_panel, tmp_path):
        out_path = tmp_path / "panel.tex"
        ret = two_key_panel.to_latex(path=out_path)
        assert ret is None
        assert out_path.exists()
        assert 'tabular' in out_path.read_text()

    def test_caption_and_label(self, two_key_panel):
        tex = two_key_panel.to_latex(
            caption="Panel rejection rates", label="tab:panel"
        )
        assert "Panel rejection rates" in tex
        assert "tab:panel" in tex


class TestToMarkdown:
    def test_returns_string(self, two_key_panel):
        md = two_key_panel.to_markdown()
        assert isinstance(md, str)
        assert md.strip() != ''

    def test_writes_file(self, two_key_panel, tmp_path):
        out_path = tmp_path / "panel.md"
        ret = two_key_panel.to_markdown(path=out_path)
        assert ret is None
        assert out_path.exists()
        content = out_path.read_text()
        assert "| " in content

    def test_header_contains_rejection_rate(self, two_key_panel):
        md = two_key_panel.to_markdown()
        first_line = md.splitlines()[0]
        assert 'rejection_rate' in first_line
        assert 'n_keys' in first_line


# ---------------------------------------------------------------------------
# __repr__ smoke test
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_includes_n_keys(self, three_key_panel):
        r = repr(three_key_panel)
        assert 'n_keys=3' in r
        assert 'n_models=2' in r
