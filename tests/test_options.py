"""v0.4.0rc1: ``pyRVtest.options`` behavior.

Background (Marco's 2026-04-18 break-it email): ``options.verbose`` and
``options.digits`` were declared in :mod:`pyRVtest.options` but nothing
read them after the v0.4 logging refactor. rc1 wires ``digits`` through
the DataFrame formatters and emits a ``DeprecationWarning`` on reads of
``verbose`` pointing at the replacement logging API.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

import pyRVtest
from pyRVtest.results._format import _dataframe_to_github_markdown


class TestDigitsOption:
    def test_default_is_six(self):
        # rc1 aligned the default with the previous hardcoded :.6g behavior
        # in ``_dataframe_to_github_markdown``.
        assert pyRVtest.options.digits == 6

    def test_override_respected_in_markdown(self):
        original = pyRVtest.options.digits
        try:
            pyRVtest.options.digits = 4
            frame = pd.DataFrame({'x': [0.123456789]})
            md = _dataframe_to_github_markdown(frame)
            assert "0.1235" in md
        finally:
            pyRVtest.options.digits = original

    def test_override_respected_at_default(self):
        frame = pd.DataFrame({'x': [0.123456789]})
        md = _dataframe_to_github_markdown(frame)
        assert "0.123457" in md


class TestVerboseDeprecation:
    """``options.verbose`` reads emit a DeprecationWarning pointing at the
    logging API. Writes are silent (to protect the 20+ existing tests that
    use ``options.verbose = False`` as a quietness lever).

    Note: ``hasattr(opt, 'verbose')`` returns True even with no prior
    assignment, because PEP 562 ``__getattr__`` returns ``True`` rather
    than raising ``AttributeError`` for this name. Tests introspect the
    module namespace directly via ``vars(opt)`` to detect whether the
    attribute has been assigned.
    """

    def _save_and_clear_verbose(self, opt):
        """Return (had_assignment, prior_value). Deletes the namespace
        entry if one existed so the subsequent read goes through
        ``__getattr__``.
        """
        namespace = vars(opt)
        if 'verbose' in namespace:
            prior = namespace['verbose']
            del namespace['verbose']
            return True, prior
        return False, None

    def _restore_verbose(self, opt, had_assignment, prior):
        namespace = vars(opt)
        if had_assignment:
            namespace['verbose'] = prior
        else:
            namespace.pop('verbose', None)

    def test_read_without_prior_write_emits_deprecation(self):
        import pyRVtest.options as opt
        had, prior = self._save_and_clear_verbose(opt)
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                _ = opt.verbose
            msgs = [
                w for w in captured
                if issubclass(w.category, DeprecationWarning)
                and 'options.verbose is deprecated' in str(w.message)
            ]
            assert msgs, (
                f"Expected DeprecationWarning on read; got "
                f"{[str(w.message) for w in captured]}"
            )
        finally:
            self._restore_verbose(opt, had, prior)

    def test_read_after_write_is_silent(self):
        import pyRVtest.options as opt
        had, prior = self._save_and_clear_verbose(opt)
        try:
            opt.verbose = False
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                assert opt.verbose is False
            assert not any(
                issubclass(w.category, DeprecationWarning)
                and 'options.verbose' in str(w.message)
                for w in captured
            ), 'Deprecation warning should not fire after assignment.'
        finally:
            self._restore_verbose(opt, had, prior)
