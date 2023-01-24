"""Formulation of data matrices and absorption of fixed effects."""

import token
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Type, Union

import numpy as np
import patsy
import patsy.builtins
import patsy.contrasts
import patsy.desc
import patsy.design_info
import patsy.origin
from pyblp.utilities.basics import (Array, Data, StringRepresentation, extract_size, interact_ids)
from pyblp.configurations.formulation import (
    Absorb, ColumnFormulation, CategoricalTreatment, parse_terms, design_matrix, parse_term_expression
)
import sympy as sp
import sympy.parsing.sympy_parser


class Formulation(StringRepresentation):
    r"""Configuration for designing matrices and absorbing fixed effects.

    .. note::
        This class is a copy of the Formulation class from PyBLP.

    Internally, the `patsy <https://patsy.readthedocs.io/en/stable/>`_ package is used to convert data and R-style
    formulas into matrices. All of the standard
    `binary operators <https://patsy.readthedocs.io/en/stable/formulas.html#operators>`_ can be used to design complex
    matrices of factor interactions:

        - ``+`` - Set union of terms.
        - ``-`` - Set difference of terms.
        - ``*`` - Short-hand. The formula ``a * b`` is the same as ``a + b + a:b``.
        - ``/`` - Short-hand. The formula ``a / b`` is the same as ``a + a:b``.
        - ``:`` - Interactions between two sets of terms.
        - ``**`` - Interactions up to an integer degree.

    However, since factors need to be differentiated (for example, when computing elasticities), only the most essential
    functions are supported:

        - ``C`` - Mark a variable as categorical. See :func:`patsy.builtins.C`. Arguments are not supported.
        - ``I`` - Encapsulate mathematical operations. See :func:`patsy.builtins.I`.
        - ``log`` - Natural logarithm function.
        - ``exp`` - Natural exponential function.

    Data associated with variables should generally already be transformed. However, when encapsulated by ``I()``, these
    operators function like normal mathematical operators on numeric variables: ``+`` adds, ``-`` subtracts, ``*``
    multiplies, ``/`` divides, and ``**`` exponentiates.

    Internally, mathematical operations are parsed and evaluated by the `SymPy <https://www.sympy.org/en/index.html>`_
    package, which is also used to symbolically differentiate terms when derivatives are needed.

    Parameters
    ----------
    formula : `str`
        R-style formula used to design a matrix. Variable names will be validated when this formulation and data are
        passed to a function that uses them. By default, an intercept is included, which can be removed with ``0`` or
        ``-1``. If ``absorb`` is specified, intercepts are ignored.
    absorb : `str, optional`
        R-style formula used to design a matrix of categorical variables representing fixed effects, which will be
        absorbed into the matrix designed by ``formula`` by the `PyHDFE <https://pyhdfe.readthedocs.io/en/stable/>`_
        package. Fixed effect absorption is only supported for some matrices. Unlike ``formula``, intercepts are
        ignored. Only categorical variables are supported.
    absorb_method : `str, optional`
        Method by which fixed effects will be absorbed. For a full list of supported methods, refer to the
        ``residualize_method`` argument of :func:`pyhdfe.create`.

        By default, the simplest methods are used: simple de-meaning for a single fixed effect and simple iterative
        de-meaning by way of the method of alternating projections (MAP) for multiple dimensions of fixed effects. For
        multiple dimensions, non-accelerated MAP is unlikely to be the fastest algorithm. If fixed effect absorption
        seems to be taking a long time, consider using a different method such as ``'lsmr'``, using ``absorb_options``
        to specify a MAP acceleration method, or configuring other options such as termination tolerances.
    absorb_options : `dict, optional`
        Configuration options for the chosen ``method``, which will be passed to the ``options`` argument of
        :func:`pyhdfe.create`.

    """

    _formula: str
    _absorb: Optional[str]
    _absorb_method: Optional[str]
    _absorb_options: dict
    _terms: List[patsy.desc.Term]
    _absorbed_terms: List[patsy.desc.Term]
    _expressions: List[sp.Expr]
    _absorbed_expressions: List[sp.Expr]
    _names: Set[str]
    _absorbed_names: Set[str]

    def __init__(
            self, formula: str, absorb: Optional[str] = None, absorb_method: Optional[str] = None,
            absorb_options: Optional[Mapping] = None) -> None:
        """Parse the formula into patsy terms and SymPy expressions. In the process, validate it as much as possible
        without any data.
        """

        # validate the formulas
        if not isinstance(formula, str):
            raise TypeError("formula must be a str.")
        if absorb is not None and not isinstance(absorb, str):
            raise TypeError("absorb must be a None or a str.")

        # parse the formulas into patsy terms
        self._formula = formula
        self._absorb = absorb
        self._terms = parse_terms(formula)
        self._absorbed_terms: List[patsy.desc.Term] = []
        if absorb is not None:
            self._absorbed_terms = parse_terms(f'{absorb} - 1')

        # ignore intercepts if there are any absorbed terms and check that there is at least one term
        if self._absorbed_terms:
            self._terms = [t for t in self._terms if t != patsy.desc.INTERCEPT]
        if not self._terms:
            raise patsy.PatsyError("formula has no terms.", patsy.origin.Origin(formula, 0, len(formula)))

        # parse the terms into SymPy expressions and extract variable names
        self._expressions = [parse_term_expression(t) for t in self._terms]
        self._absorbed_expressions = [parse_term_expression(t) for t in self._absorbed_terms]
        self._names = {str(s) for e in self._expressions for s in e.free_symbols}
        self._absorbed_names = {str(s) for e in self._absorbed_expressions for s in e.free_symbols}
        if sum(not e.free_symbols for e in self._expressions) > 1:
            origin = patsy.origin.Origin(formula, 0, len(formula))
            raise patsy.PatsyError("formula should have at most one constant term.", origin)
        if self._absorbed_expressions and any(not e.free_symbols for e in self._absorbed_expressions):
            assert absorb is not None
            origin = patsy.origin.Origin(absorb, 0, len(absorb))
            raise patsy.PatsyError("absorb should not have any constant terms.", origin)

        # validate fixed effect absorption options
        if absorb_method is not None and not isinstance(absorb_method, str):
            raise TypeError("absorb_method must be None or a string.")
        if absorb_options is None:
            absorb_options = {}
        elif not isinstance(absorb_options, dict):
            raise TypeError("absorb_options must be None or a dict.")
        self._absorb_method = absorb_method
        self._absorb_options = absorb_options

    def __reduce__(self) -> Tuple[Type['Formulation'], Tuple]:
        """Handle pickling."""
        return (self.__class__, (self._formula, self._absorb, self._absorb_method, self._absorb_options))

    def __str__(self) -> str:
        """Format the terms as a string."""
        names: List[str] = []
        for term in self._terms:
            names.append('1' if term == patsy.desc.INTERCEPT else term.name())
        for absorbed_term in self._absorbed_terms:
            names.append(f'Absorb[{absorbed_term.name()}]')
        return ' + '.join(names)

    def _build_matrix(self, data: Mapping) -> Tuple[Array, List['ColumnFormulation'], Data]:
        """Convert a mapping from variable names to arrays into the designed matrix, a list of column formulations that
        describe the columns of the matrix, and a mapping from variable names to arrays of data underlying the matrix,
        which include unchanged continuous variables and indicators constructed from categorical variables.
        """

        # normalize the data
        data_mapping: Data = {}
        for name in self._names:
            try:
                data_mapping[name] = np.asarray(data[name]).flatten()
            except Exception as exception:
                origin = patsy.origin.Origin(self._formula, 0, len(self._formula))
                raise patsy.PatsyError(f"Failed to load data for '{name}'.", origin) from exception

        # always have at least one column to represent the size of the data
        if not data_mapping:
            data_mapping = {'': np.zeros(extract_size(data))}

        # design the matrix (adding an intercept term if there are absorbed terms gets Patsy to use reduced coding)
        if self._absorbed_terms:
            matrix_design = design_matrix([patsy.desc.INTERCEPT] + self._terms, data_mapping)
        else:
            matrix_design = design_matrix(self._terms, data_mapping)

        # store matrix column indices and build column formulations for each designed column (ignore the intercept if
        #   it was added only to get Patsy to use reduced coding)
        column_indices: List[int] = []
        column_formulations: List[ColumnFormulation] = []
        for term, expression in zip(self._terms, self._expressions):
            if term != patsy.desc.INTERCEPT or not self._absorbed_terms:
                term_slice = matrix_design.term_slices[term]
                for index in range(term_slice.start, term_slice.stop):
                    column_indices.append(index)
                    formula = '1' if term == patsy.desc.INTERCEPT else matrix_design.column_names[index]
                    column_formulations.append(ColumnFormulation(formula, expression))

        # construct a mapping from continuous variable names that appear in at least one column to their arrays
        underlying_data: Data = {}
        for formulation in column_formulations:
            for symbol in formulation.expression.free_symbols:
                underlying_data[symbol.name] = data_mapping.get(symbol.name)

        # supplement the mapping with indicators constructed from categorical variables
        for factor, info in matrix_design.factor_infos.items():
            if info.type == 'categorical':
                indicator_design = design_matrix([patsy.desc.Term([factor])], data_mapping)
                indicator_matrix = build_matrix(indicator_design, data_mapping)
                for name, indicator in zip(indicator_design.column_names, indicator_matrix.T):
                    symbol = CategoricalTreatment.parse_full_symbol(name)
                    if symbol.name in underlying_data:
                        underlying_data[symbol.name] = indicator

        matrix = build_matrix(matrix_design, data_mapping)
        return matrix[:, column_indices], column_formulations, underlying_data

    def _build_ids(self, data: Mapping) -> Array:
        """Convert a mapping from variable names to arrays into the designed matrix of IDs to be absorbed."""

        # normalize the data
        data_mapping: Data = {}
        for name in self._absorbed_names:
            try:
                data_mapping[name] = np.asarray(data[name]).flatten()
            except Exception as exception:
                assert self._absorb is not None
                origin = patsy.origin.Origin(self._absorb, 0, len(self._absorb))
                raise patsy.PatsyError(f"Failed to load data for '{name}'.", origin) from exception

        # build columns of absorbed IDs
        ids_columns: List[Array] = []
        for term in self._absorbed_terms:
            factor_columns: List[Array] = []
            term_design = design_matrix([term], data_mapping)
            for factor, info in term_design.factor_infos.items():
                if info.type != 'categorical':
                    raise patsy.PatsyError("Only categorical variables can be absorbed.", factor.origin)
                symbol = parse_expression(factor.name())
                factor_columns.append(data_mapping[symbol.name])
            ids_columns.append(interact_ids(*factor_columns))

        return np.column_stack(ids_columns)

    def _build_absorb(self, ids: Array) -> 'Absorb':
        """Build a function used to absorb fixed effects defined by columns of IDs."""
        import pyhdfe
        return Absorb(pyhdfe.create(
            ids, drop_singletons=False, compute_degrees=False, residualize_method=self._absorb_method,
            options=self._absorb_options
        ))


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

    def __init__(
            self, model_downstream: Optional[str] = None, model_upstream: Optional[str] = None,
            ownership_downstream: Optional[str] = None, ownership_upstream: Optional[str] = None,
            custom_model_specification: Optional[dict] = None, vertical_integration: Optional[str] = None,
            unit_tax: Optional[str] = None, advalorem_tax: Optional[str] = None, advalorem_payer: Optional[str] = None,
            cost_scaling: Optional[str] = None,
            kappa_specification_downstream: Optional[Union[str, Callable[[Any, Any], float]]] = None,
            kappa_specification_upstream: Optional[Union[str, Callable[[Any, Any], float]]] = None,
            user_supplied_markups: Optional[str] = None) -> None:
        """Parse the formula into patsy terms and SymPy expressions. In the process, validate it as much as possible
        without any data.
        """

        # validate the parameters
        model_set = {'monopoly', 'cournot', 'bertrand', 'perfect_competition', 'other'}
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

    def __reduce__(self) -> Tuple[Type['Formulation'], Tuple]:
        """Handle pickling."""
        return (self.__class__, (
            self._model_downstream, self._model_upstream, self._ownership_downstream, self._ownership_upstream,
            self._custom_model_specification, self._vertical_integration, self._custom_model_specification,
            self._kappa_specification_downstream, self._kappa_specification_upstream, self._user_supplied_markups
        ))

    def __str__(self) -> str:
        """Format the terms as a string."""
        names: List[str] = [self._model_downstream, self._model_upstream]
        return ' + '.join(names)

    def _build_matrix(self, data: Mapping) -> Dict:
        """Convert a mapping from variable names to arrays into the designed matrix, a list of column formulations that
        describe the columns of the matrix, and a mapping from variable names to arrays of data underlying the matrix,
        which include unchanged continuous variables and indicators constructed from categorical variables.
        """
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
            'user_supplied_markups': self._user_supplied_markups
        })
        return model_mapping


def build_matrix(design: patsy.design_info.DesignInfo, data: Mapping) -> Array:
    """Build a matrix according to its design and data mapping variable names to arrays.

    .. note::
        This function is a copy from PyBLP for computational speed.
    """

    # identify the number of rows in the data
    size = next(iter(data.values())).shape[0]

    # if the design lacks factors, it must consist of only an intercept term
    if not design.factor_infos:
        return np.ones((size, 1))

    # build the matrix and raise an exception if there are any null values
    matrix = patsy.build.build_design_matrices([design], data, NA_action='raise')[0].base

    # if the design did not use any data, the matrix may be a single row that needs to be stacked to the proper height
    return matrix if matrix.shape[0] == size else np.repeat(matrix[[0]], size, axis=0)


def parse_expression(string: str, mark_categorical: bool = False) -> sp.Expr:
    """Parse a SymPy expression from a string. Optionally, preserve the categorical marker function instead of treating
    it like the identify function.

     .. note::
        This function is a copy from PyBLP for computational speed.
    """

    # list reserved patsy and SymPy names that represent special functions and classes
    patsy_function_names = {'I', 'C'}
    sympy_function_names = {'log', 'exp'}
    sympy_class_names = {'Add', 'Mul', 'Pow', 'Integer', 'Float', 'Symbol'}

    # build a mapping from reserved names to the functions and classes that they represent (patsy functions are dealt
    #   with after parsing)
    mapping = {n: sp.Function(n) for n in patsy_function_names}
    mapping.update({n: getattr(sp, n) for n in sympy_function_names | sympy_class_names})

    def transform_tokens(tokens: List[Tuple[int, str]], _: Any, __: Any) -> List[Tuple[int, str]]:
        """Validate a list of tokens and add any unrecognized names as new SymPy symbols."""
        transformed: List[Tuple[int, str]] = []
        symbol_candidate = None
        for code, value in tokens:
            if code not in {token.NAME, token.OP, token.NUMBER, token.NEWLINE, token.ENDMARKER}:
                raise ValueError(f"The token '{value}' is invalid.")
            if code == token.OP and value not in {'+', '-', '*', '/', '**', '(', ')'}:
                raise ValueError(f"The operation '{value}' is invalid.")
            if code == token.OP and value == '(' and symbol_candidate is not None:
                raise ValueError(f"The function '{symbol_candidate}' is invalid.")
            if code != token.NAME or value in set(mapping) - sympy_class_names:
                transformed.append((code, value))
                symbol_candidate = None
                continue
            if value in sympy_class_names | {'Intercept'}:
                raise ValueError(f"The name '{value}' is invalid.")
            transformed.extend([(token.NAME, 'Symbol'), (token.OP, '('), (token.NAME, repr(value)), (token.OP, ')')])
            symbol_candidate = value

        return transformed

    # define a function that validates the appearance of categorical marker functions
    def validate_categorical(candidate: sp.Expr, depth: int = 0, categorical: bool = False) -> None:
        """Recursively validate that all categorical marker functions in an expression accept only a single variable
        argument and that they are not arguments to other functions.
        """
        if categorical and depth > 1:
            raise ValueError("The C function must not be an argument to another function.")
        for arg in candidate.args:
            if categorical and not isinstance(arg, sp.Symbol):
                raise ValueError("The C function accepts only a single variable.")
            validate_categorical(arg, depth + 1, candidate.func == mapping['C'])

    # parse the expression, validate it by attempting to represent it as a string, and validate categorical markers
    try:
        expression = sympy.parsing.sympy_parser.parse_expr(string, mapping, [transform_tokens], evaluate=False)
        str(expression)
        validate_categorical(expression)
    except (TypeError, ValueError) as exception:
        raise ValueError(f"The expression '{string}' is malformed.") from exception

    # replace patsy functions with the identity function, unless categorical variables are to be explicitly marked
    for name in patsy_function_names:
        if name != 'C' or not mark_categorical:
            expression = expression.replace(mapping[name], sp.Id)

    return expression
