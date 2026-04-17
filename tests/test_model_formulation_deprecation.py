"""Verify ModelFormulation's once-per-session DeprecationWarning (v0.4 step 5c).

The warning fires on the first construction of ModelFormulation in a Python
session and not again. The implementation uses a class-level flag so we
don't mutate global warnings filter state.
"""

from __future__ import annotations

import warnings

import pytest

import pyRVtest
from pyRVtest.formulation import ModelFormulation


@pytest.fixture(autouse=True)
def _reset_deprecation_flag():
    """Reset the once-per-session flag around each test so the warning
    state is isolated. Without this, one test triggering the warning would
    prevent another test from observing it.
    """
    saved = ModelFormulation._deprecation_warned
    ModelFormulation._deprecation_warned = False
    yield
    ModelFormulation._deprecation_warned = saved


class TestModelFormulationDeprecationWarning:
    def test_first_construction_emits_warning(self):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            pyRVtest.ModelFormulation(
                model_downstream='bertrand', ownership_downstream='firm_ids',
            )
        dep_warnings = [
            w for w in recorded if issubclass(w.category, DeprecationWarning)
        ]
        assert len(dep_warnings) == 1, (
            f"Expected one DeprecationWarning on first ModelFormulation "
            f"construction, got {len(dep_warnings)}."
        )
        message = str(dep_warnings[0].message)
        assert 'ModelFormulation is deprecated' in message
        assert 'Bertrand' in message  # migration hint mentions the new class

    def test_second_construction_does_not_emit_warning(self):
        # First construction sets the flag.
        pyRVtest.ModelFormulation(
            model_downstream='bertrand', ownership_downstream='firm_ids',
        )
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            pyRVtest.ModelFormulation(
                model_downstream='cournot', ownership_downstream='firm_ids',
            )
        dep_warnings = [
            w for w in recorded if issubclass(w.category, DeprecationWarning)
        ]
        assert len(dep_warnings) == 0, (
            "Second ModelFormulation construction in the same session should "
            f"not re-emit the DeprecationWarning; got {len(dep_warnings)}."
        )


class TestNewAPIDoesNotEmitWarning:
    """Constructing the new ConductModel / Vertical classes must NOT emit
    the ModelFormulation deprecation warning.
    """

    def test_bertrand_is_silent(self):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            pyRVtest.Bertrand(ownership='firm_ids')
        dep_warnings = [
            w for w in recorded if issubclass(w.category, DeprecationWarning)
        ]
        assert len(dep_warnings) == 0

    def test_vertical_is_silent(self):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            pyRVtest.Vertical(
                downstream=pyRVtest.Bertrand(ownership='firm_ids'),
                upstream=pyRVtest.Monopoly(ownership='manu_ids'),
                vertical_integration='vi_col',
            )
        dep_warnings = [
            w for w in recorded if issubclass(w.category, DeprecationWarning)
        ]
        assert len(dep_warnings) == 0
