"""Formulation of data matrices and absorption of fixed effects."""

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, Union

from pyblp.utilities.basics import Array
from pyblp.configurations.formulation import Absorb, Formulation  # noqa: F401


class ModelFormulation(object):
    r"""Configuration for designing matrices and absorbing fixed effects.

    For each model, the user can specify the downstream and upstream (optional) models, the downstream and upstream
    ownership structure, a custom model and markup formula, and vertical integration. The user can also choose to forgo
    markup computation and specify their own markups with `user_supplied_markups`. Additionally, there are
    specifications related to testing conduct with taxes.

    There is a built-in library of models that the researcher can choose from.

    Here, we have another difference with PyBLP.  In PyBLP, if one wants to build an ownership matrix, there must be a
    variable called `firm_id` in the `product_data`.  With pyRVtest, the researcher can pass any variable in the
    `product_data` as `ownership_downstream` and from this, the ownership matrix in each market will be built.

    .. note::
        We are working on adding additional models to this library as well as options for the researcher to specify
        their own markup function.)

    Parameters
    ----------
    model_downstream : `str, optional`
        The model of conduct for downstream firms (or if no vertical structure, the model of conduct). One of
        "bertrand", "cournot", "monopoly", "perfect_competition", or "other".
    model_upstream : `str, optional`
        The model of conduct for upstream firms. One of "bertrand", "cournot", "monopoly", "perfect_competition", or
        "other".
    ownership_downstream: `str, optional`
        Column indicating which firm ids to use for ownership matrix construction for downstream firms.
    ownership_upstream: `str, optional`
        Column indicating which firm ids to use for ownership matrix construction for upstream firms.
    custom_model_specification: `dict, optional`
        A dictionary containing an optional custom markup formula specified by the user. The specified function must
        consist of objects computed within the package.
    vertical_integration: `str, optional`
        The column name for the data column which indicates the vertical ownership structure.
    unit_tax: `str, optional`
        The column name for the vector containing information on unit taxes.
    advalorem_tax: `str, optional`
        The column name for the vector containing information on advalorem taxes.
    advalorem_payer: `str, optional`
        A string indicating who pays for the advalorem tax in the given model.
    cost_scaling: `str, optional`
        The column name for the cost scaling parameter.
    kappa_specification_downstream: `Union[str, Callable[[Any, Any], float]]], optional`
        Information on the degree of cooperation among downstream firms for each market.
    kappa_specification_upstream: `Union[str, Callable[[Any, Any], float]]], optional`
        Information on the degree of cooperation among upstream firms for each market.
    user_supplied_markups: `str, optional`
        The name of the column containing user-supplied markups.
    mix_flag: `str, optional`
        Column name for a boolean vector indicating which products compete in prices (True = Bertrand, False = Cournot)
        within each market. Required when `model_downstream='mix_cournot_bertrand'`.

    """

    _model_downstream: Optional[str]
    _model_upstream: Optional[str]
    _ownership_downstream: Optional[str]
    _ownership_upstream: Optional[str]
    _custom_model_specification: Optional[dict]
    _vertical_integration: Optional[str]
    _unit_tax: Optional[str]
    _advalorem_tax: Optional[str]
    _advalorem_payer: Optional[str]
    _cost_scaling: Optional[str]
    _kappa_specification_downstream: Optional[Union[str, Callable[[Any, Any], float]]]
    _kappa_specification_upstream: Optional[Union[str, Callable[[Any, Any], float]]]
    _user_supplied_markups: Optional[str]
    _mix_flag: Optional[str]

    def __init__(
            self, model_downstream: Optional[str] = None, model_upstream: Optional[str] = None,
            ownership_downstream: Optional[str] = None, ownership_upstream: Optional[str] = None,
            custom_model_specification: Optional[dict] = None, vertical_integration: Optional[str] = None,
            unit_tax: Optional[str] = None, advalorem_tax: Optional[str] = None, advalorem_payer: Optional[str] = None,
            cost_scaling: Optional[str] = None,
            kappa_specification_downstream: Optional[Union[str, Callable[[Any, Any], float]]] = None,
            kappa_specification_upstream: Optional[Union[str, Callable[[Any, Any], float]]] = None,
            user_supplied_markups: Optional[str] = None, mix_flag: Optional[str] = None) -> None:
        """Parse the formula into patsy terms and SymPy expressions. In the process, validate it as much as possible
        without any data.
        """

        # validate the parameters
        model_set = {'monopoly', 'cournot', 'bertrand', 'perfect_competition', 'mix_cournot_bertrand', 'other'}
        if model_downstream is None and user_supplied_markups is None:
            raise TypeError("Either model_downstream or user_supplied_markups must be provided.")
        if model_downstream is not None and model_downstream not in model_set:
            raise TypeError("model_downstream must be monopoly, bertrand, cournot, perfect_competition, or other.")
        if model_upstream is not None and model_upstream not in model_set:
            raise TypeError("model_upstream must be monopoly, bertrand, cournot, perfect_competition, or other.")
        if model_upstream is not None and model_downstream in {'cournot'} and model_upstream in {'cournot'}:
            raise TypeError("model_upstream and model_downstream cannot both be cournot.")
        if ownership_downstream is not None and not isinstance(ownership_downstream, str):
            raise TypeError("ownership_downstream must be a None or a str.")
        if ownership_upstream is not None and not isinstance(ownership_upstream, str):
            raise TypeError("ownership_upstream must be a None or a str.")
        if model_upstream is not None and not isinstance(ownership_upstream, str):
            raise TypeError("ownership_upstream must be a str when upstream model defined.")
        if vertical_integration is not None and not isinstance(vertical_integration, str):
            raise TypeError("vertical_integration must be a None or a str.")
        if unit_tax is not None and not isinstance(unit_tax, str):
            raise TypeError("unit_tax must be a None or a str.")
        if advalorem_tax is not None and not isinstance(advalorem_tax, str):
            raise TypeError("advalorem_tax must be a None or a str.")
        if advalorem_payer is not None and advalorem_payer not in {'firm', 'consumer', 'firms', 'consumers'}:
            raise TypeError("advalorem_payer must be a None, firm, or consumer.")
        if advalorem_tax is not None and advalorem_payer is None:
            raise TypeError("advalorem_payer must be defined as firm or consumer when allowing for advalorem taxes.")
        if cost_scaling is not None and not isinstance(cost_scaling, str):
            raise TypeError("cost_scaling must be a None or a str.")
        if mix_flag is not None and not isinstance(mix_flag, str):
            raise TypeError("mix_flag must be None or a str.")
        if model_downstream == 'mix_cournot_bertrand' and mix_flag is None:
            raise TypeError("mix_flag must be provided when model_downstream='mix_cournot_bertrand'.")
        if mix_flag is not None and model_downstream != 'mix_cournot_bertrand':
            raise TypeError("mix_flag is only valid when model_downstream='mix_cournot_bertrand'.")
        if model_downstream == 'other' and custom_model_specification is None and user_supplied_markups is None:
            raise TypeError("custom_model_specification must be provided when model_downstream='other'.")

        # parse the formulas into patsy terms
        self._model_downstream = model_downstream
        self._model_upstream = model_upstream
        self._ownership_downstream = ownership_downstream
        self._ownership_upstream = ownership_upstream
        self._custom_model_specification = custom_model_specification
        self._vertical_integration = vertical_integration
        self._unit_tax = unit_tax
        self._advalorem_tax = advalorem_tax
        self._advalorem_payer = advalorem_payer
        self._kappa_specification_downstream = kappa_specification_downstream
        self._kappa_specification_upstream = kappa_specification_upstream
        self._cost_scaling = cost_scaling
        self._user_supplied_markups = user_supplied_markups
        self._mix_flag = mix_flag

    def __reduce__(self) -> Tuple[Type['ModelFormulation'], Tuple]:
        """Handle pickling."""
        return (self.__class__, (
            self._model_downstream, self._model_upstream,
            self._ownership_downstream, self._ownership_upstream,
            self._custom_model_specification, self._vertical_integration,
            self._unit_tax, self._advalorem_tax, self._advalorem_payer,
            self._cost_scaling,
            self._kappa_specification_downstream, self._kappa_specification_upstream,
            self._user_supplied_markups, self._mix_flag,
        ))

    def __str__(self) -> str:
        """Format the terms as a string."""
        names: List[str] = [self._model_downstream, self._model_upstream]
        return ' + '.join(names)

    def _build_matrix(self, data: Mapping) -> Dict:
        """Convert a mapping from variable names to arrays into a dictionary of model configuration values."""
        model_mapping: Dict[Union[str, Array]] = {}
        model_mapping.update({
            'model_downstream': self._model_downstream,
            'model_upstream': self._model_upstream,
            'ownership_downstream': self._ownership_downstream,
            'ownership_upstream': self._ownership_upstream,
            'custom_model_specification': self._custom_model_specification,
            'vertical_integration': self._vertical_integration,
            'unit_tax': self._unit_tax,
            'advalorem_tax': self._advalorem_tax,
            'advalorem_payer': self._advalorem_payer,
            'cost_scaling': self._cost_scaling,
            'kappa_specification_downstream': self._kappa_specification_downstream,
            'kappa_specification_upstream': self._kappa_specification_upstream,
            'user_supplied_markups': self._user_supplied_markups,
            'mix_flag': self._mix_flag
        })
        return model_mapping
