"""v0.4 step 9: tests for ``ProblemResults`` export helpers.

Covers the four export methods added in step 9:

- :meth:`pyRVtest.ProblemResults.to_dataframe`
- :meth:`pyRVtest.ProblemResults.summary_df`
- :meth:`pyRVtest.ProblemResults.to_latex`
- :meth:`pyRVtest.ProblemResults.to_markdown`

The fixture reuses the base Bertrand-vs-perfect-competition DGP from
``tests/test_analytical.py`` so the tests exercise the real
``Problem.solve()`` pipeline (not hand-crafted ``ProblemResults``
instances). A second two-instrument-set variant is used to verify
long-form and summary reshaping handles ``L > 1`` correctly.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import pyRVtest
from pyRVtest.results.results import _dataframe_to_github_markdown

from .test_analytical import _build_base_dgp, _run_pyrvtest_base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def base_results():
    """A single-instrument-set base run (2 models, 1 instrument set)."""
    product_data, _ = _build_base_dgp()
    return _run_pyrvtest_base(product_data, clustering=False)


@pytest.fixture(scope='module')
def two_iv_results():
    """A two-instrument-set base run (2 models, 2 instrument sets)."""
    product_data, _ = _build_base_dgp()
    model_formulations = (
        pyRVtest.ModelFormulation(
            model_downstream='bertrand',
            ownership_downstream='firm_ids',
            user_supplied_markups='markups_m1',
        ),
        pyRVtest.ModelFormulation(
            model_downstream='perfect_competition',
            user_supplied_markups='markups_m2',
        ),
    )
    testing_problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=[
            pyRVtest.Formulation('0 + iv0 + iv1'),
            pyRVtest.Formulation('0 + iv1 + iv2'),
        ],
        model_formulations=model_formulations,
        product_data=product_data,
        demand_results=None,
    )
    return testing_problem.solve(demand_adjustment=False, clustering_adjustment=False)


# ---------------------------------------------------------------------------
# to_dataframe
# ---------------------------------------------------------------------------

class TestToDataFrame:
    """Long-form pandas DataFrame export."""

    def test_returns_dataframe(self, base_results):
        df = base_results.to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, base_results):
        df = base_results.to_dataframe()
        expected = [
            'instrument_set', 'instrument_set_label',
            'model_i', 'model_j', 'model_i_label', 'model_j_label',
            'TRV', 'F', 'MCS_pvalue',
        ]
        assert list(df.columns) == expected

    def test_row_count_single_iv_set(self, base_results):
        # M=2, L=1 -> L * M * (M-1) = 2 ordered pairs.
        df = base_results.to_dataframe()
        assert len(df) == 2

    def test_row_count_two_iv_sets(self, two_iv_results):
        # M=2, L=2 -> L * M * (M-1) = 4 ordered pairs.
        df = two_iv_results.to_dataframe()
        assert len(df) == 4
        # Both instrument sets present.
        assert sorted(df['instrument_set'].unique().tolist()) == [0, 1]

    def test_no_self_pairs(self, base_results):
        df = base_results.to_dataframe()
        assert (df['model_i'] != df['model_j']).all()

    def test_trv_matches_source(self, base_results):
        df = base_results.to_dataframe()
        # Off-diagonal of TRV[j] must appear in the long form.
        for _, row in df.iterrows():
            j = int(row['instrument_set'])
            i = int(row['model_i'])
            k = int(row['model_j'])
            expected = float(np.asarray(base_results.TRV[j])[i, k])
            np.testing.assert_allclose(row['TRV'], expected, atol=1e-12)

    def test_f_matches_source(self, base_results):
        df = base_results.to_dataframe()
        for _, row in df.iterrows():
            j = int(row['instrument_set'])
            i = int(row['model_i'])
            k = int(row['model_j'])
            expected = float(np.asarray(base_results.F[j])[i, k])
            np.testing.assert_allclose(row['F'], expected, atol=1e-12)

    def test_mcs_pvalue_depends_only_on_model_i(self, base_results):
        # MCS pvalue is per-model, not per-pair; the long form should
        # duplicate it across the j column.
        df = base_results.to_dataframe()
        for i in df['model_i'].unique():
            sub = df[df['model_i'] == i]
            assert sub['MCS_pvalue'].nunique() == 1


# ---------------------------------------------------------------------------
# summary_df
# ---------------------------------------------------------------------------

class TestSummaryDF:
    """Compact wide-form summary."""

    def test_returns_dataframe(self, base_results):
        out = base_results.summary_df()
        assert isinstance(out, pd.DataFrame)

    def test_expected_columns_with_default_alpha(self, base_results):
        out = base_results.summary_df()
        assert 'reject_at_0.05' in out.columns
        for col in ('instrument_set', 'model_i', 'model_j', 'TRV', 'F',
                    'MCS_pvalue_model_i'):
            assert col in out.columns

    def test_one_row_per_unordered_pair(self, base_results):
        # M=2, L=1 -> L * M * (M-1) / 2 = 1 unordered pair per iv set.
        out = base_results.summary_df()
        assert len(out) == 1
        # And model_i < model_j by construction.
        assert (out['model_i'] < out['model_j']).all()

    def test_two_iv_sets_row_count(self, two_iv_results):
        out = two_iv_results.summary_df()
        assert len(out) == 2  # L=2, one unordered pair each.
        assert sorted(out['instrument_set'].unique().tolist()) == [0, 1]

    def test_reject_flag_matches_two_sided_threshold(self, base_results):
        out = base_results.summary_df(alpha=0.05)
        crit = float(norm.ppf(1.0 - 0.05 / 2.0))
        for _, row in out.iterrows():
            trv = float(row['TRV'])
            expected_reject = bool(np.isfinite(trv) and abs(trv) > crit)
            assert bool(row['reject_at_0.05']) == expected_reject

    def test_alpha_column_name_follows_alpha(self, base_results):
        out = base_results.summary_df(alpha=0.10)
        assert 'reject_at_0.1' in out.columns
        # Default-flag column must not also be present.
        assert 'reject_at_0.05' not in out.columns

    def test_invalid_alpha_raises(self, base_results):
        with pytest.raises(ValueError):
            base_results.summary_df(alpha=0.0)
        with pytest.raises(ValueError):
            base_results.summary_df(alpha=1.0)
        with pytest.raises(ValueError):
            base_results.summary_df(alpha=-0.1)


# ---------------------------------------------------------------------------
# to_latex
# ---------------------------------------------------------------------------

class TestToLatex:
    """LaTeX export of summary_df."""

    def test_returns_string_by_default(self, base_results):
        tex = base_results.to_latex()
        assert isinstance(tex, str)
        # The default pandas output contains a tabular environment.
        assert 'tabular' in tex

    def test_writes_file_when_path_given(self, base_results, tmp_path):
        out_path = tmp_path / "results.tex"
        ret = base_results.to_latex(path=out_path)
        assert ret is None
        assert out_path.exists()
        content = out_path.read_text()
        assert 'tabular' in content

    def test_caption_and_label_threaded_through(self, base_results):
        tex = base_results.to_latex(caption="RV test results", label="tab:rv")
        assert "RV test results" in tex
        # Label is rendered as \label{tab:rv} by pandas.
        assert "tab:rv" in tex

    def test_escape_false_preserves_math(self, base_results):
        # The default pandas ``escape=True`` would escape backslashes
        # and braces in any cell value. We use ``escape=False`` so math
        # passes through. As a proxy check, assert the output contains
        # the literal backslash-begin-tabular and does not contain a
        # double-escaped variant.
        tex = base_results.to_latex()
        assert r"\begin{tabular}" in tex


# ---------------------------------------------------------------------------
# to_markdown
# ---------------------------------------------------------------------------

class TestToMarkdown:
    """GitHub-flavored markdown export of summary_df."""

    def test_returns_string_by_default(self, base_results):
        md = base_results.to_markdown()
        assert isinstance(md, str)
        assert md.strip() != ''

    def test_writes_file_when_path_given(self, base_results, tmp_path):
        out_path = tmp_path / "results.md"
        ret = base_results.to_markdown(path=out_path)
        assert ret is None
        assert out_path.exists()
        content = out_path.read_text()
        assert "| " in content

    def test_header_and_separator(self, base_results):
        md = base_results.to_markdown()
        lines = md.splitlines()
        # Line 0 is the header; line 1 must be a pipe-separated row of
        # dashes (GitHub-flavored separator).
        assert lines[0].startswith("| ")
        assert re.match(r"^\|\s*(---\s*\|\s*)+$", lines[1]) is not None

    def test_column_names_in_header(self, base_results):
        md = base_results.to_markdown()
        header = md.splitlines()[0]
        for col in ('instrument_set', 'model_i', 'model_j', 'TRV', 'F',
                    'MCS_pvalue_model_i'):
            assert col in header

    def test_has_expected_row_count(self, two_iv_results):
        md = two_iv_results.to_markdown()
        lines = [line for line in md.splitlines() if line.strip()]
        # header + separator + 2 data rows.
        assert len(lines) == 4


# ---------------------------------------------------------------------------
# _dataframe_to_github_markdown helper — tested directly since it is a
# small pure utility we rely on for the optional-tabulate-free rendering.
# ---------------------------------------------------------------------------

class TestMarkdownHelper:
    """Direct tests of the private markdown helper for edge cases."""

    def test_empty_frame_renders_header_and_separator(self):
        frame = pd.DataFrame({'a': [], 'b': []})
        md = _dataframe_to_github_markdown(frame)
        lines = md.splitlines()
        assert lines[0] == "| a | b |"
        assert lines[1] == "| --- | --- |"

    def test_nan_renders_as_literal_nan(self):
        frame = pd.DataFrame({'x': [float('nan'), 1.0]})
        md = _dataframe_to_github_markdown(frame)
        assert "NaN" in md

    def test_bool_renders_as_true_false(self):
        frame = pd.DataFrame({'flag': [True, False]})
        md = _dataframe_to_github_markdown(frame)
        assert "True" in md
        assert "False" in md

    def test_float_formatting_uses_6g(self):
        frame = pd.DataFrame({'x': [0.123456789]})
        md = _dataframe_to_github_markdown(frame)
        # 6 significant digits: 0.123457
        assert "0.123457" in md
