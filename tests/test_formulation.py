"""Tests for formulation.py — ModelFormulation validation and pickling."""

import pickle
import pytest
import pyRVtest


class TestModelFormulationReduce:
    """Fix 2.3: __reduce__ must round-trip all fields through pickle."""

    def test_pickle_roundtrip_basic(self):
        mf = pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids')
        mf2 = pickle.loads(pickle.dumps(mf))
        assert mf2._model_downstream == 'bertrand'
        assert mf2._ownership_downstream == 'firm_ids'
        assert mf2._model_upstream is None

    def test_pickle_roundtrip_all_fields(self):
        """Every field set on __init__ must survive pickle round-trip."""
        mf = pyRVtest.ModelFormulation(
            model_downstream='bertrand',
            model_upstream='bertrand',
            ownership_downstream='firm_ids_down',
            ownership_upstream='firm_ids_up',
            custom_model_specification=None,
            vertical_integration='vi_col',
            unit_tax='tax_col',
            advalorem_tax='av_tax_col',
            advalorem_payer='firm',
            cost_scaling='scale_col',
        )
        mf2 = pickle.loads(pickle.dumps(mf))
        assert mf2._model_downstream == 'bertrand'
        assert mf2._model_upstream == 'bertrand'
        assert mf2._ownership_downstream == 'firm_ids_down'
        assert mf2._ownership_upstream == 'firm_ids_up'
        assert mf2._custom_model_specification is None
        assert mf2._vertical_integration == 'vi_col'
        assert mf2._unit_tax == 'tax_col'
        assert mf2._advalorem_tax == 'av_tax_col'
        assert mf2._advalorem_payer == 'firm'
        assert mf2._cost_scaling == 'scale_col'

    def test_pickle_roundtrip_mix_flag(self):
        mf = pyRVtest.ModelFormulation(
            model_downstream='mix_cournot_bertrand',
            ownership_downstream='firm_ids',
            mix_flag='is_bertrand',
        )
        mf2 = pickle.loads(pickle.dumps(mf))
        assert mf2._model_downstream == 'mix_cournot_bertrand'
        assert mf2._mix_flag == 'is_bertrand'


class TestModelFormulationValidation:
    """Fix 2.7: model_downstream='other' without custom_model_specification should raise."""

    def test_other_without_custom_spec_raises(self):
        with pytest.raises((TypeError, ValueError)):
            pyRVtest.ModelFormulation(
                model_downstream='other',
                ownership_downstream='firm_ids',
                custom_model_specification=None,
            )

    def test_other_with_custom_spec_ok(self):
        mf = pyRVtest.ModelFormulation(
            model_downstream='other',
            ownership_downstream='firm_ids',
            custom_model_specification={'my_model': lambda O, D, s: s * 0.5},
        )
        assert mf._model_downstream == 'other'
