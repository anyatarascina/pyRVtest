"""Economy-level conduct testing problem functionality."""

import contextlib
import itertools
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from pyblp.utilities.algebra import precisely_identify_collinearity
from pyblp.utilities.basics import Array, RecArray, StringRepresentation, format_seconds, format_table, get_indices, output
from scipy.linalg import inv, fractional_matrix_power
from scipy.stats import norm
import statsmodels.api as sm

from .. import options
from ..configurations.formulation import Absorb, Formulation, ModelFormulation
from ..construction import build_markups
from ..primitives import Container, Models, Products, read_critical_values_tables
from ..results.problem_results import ProblemResults


class Problem(Container, StringRepresentation):
    r"""A firm conduct testing-type problem.

    This class is initialized using the relevant data and formulations, and solved with :meth:`Problem.solve`.

    Parameters
    ----------
    cost_formulation: `Formulation`
        :class:`Formulation` is a list of the variables for observed product characteristics. All observed cost shifters
        included in this formulation must be variables in the `product_data`. To use a constant, one would replace `0`
        with `1`. To absorb fixed effects, specify `absorb = 'C(variable)'`, where the `variable` must also be in the
        `product_data`. Including this option implements fixed effects absorption using
        [PYHDFE](https://github.com/jeffgortmaker/pyhdfe), a companion package to PyBLP.

    instrument_formulation: `Formulation or sequence of Formulation`
        :class:`Formulation` is list of the variables used as excluded instruments for testing. For each instrument
        formulation, there should never be a constant. The user can specify as many instrument formulations as desired.
        All instruments must be variables in `product_data`.

        .. note::
            **Our instrument naming conventions differ from PyBLP**. With PyBLP, one specifies the excluded instruments
            for demand estimation via a naming convention in the product_data: each excluded instrument for demand
            estimation begins with `"demand_instrument"` followed by a number ( i.e., `demand_instrument0`). In
            pyRVtest, you specify directly the names of the variables in the `product_data` that you want to use as
            excluded instruments for testing (i.e., if you want to test with one instrument using the variable in the
            `product_data` named, "transportation_cost" one could specify
            `pyRVtest.Formulation('0 + transportation_cost')`.

    model_formulations: `sequence of ModelFormulation`
        :class:`ModelFormulation` defines the models that the researcher wants to test. There must be at least two
        instances of `ModelFormulation` specified to run the firm conduct testing procedure.

    product_data: `structured array-like`
        This is the data containing product and market observable characteristics, as well as instruments.

    pyblp_results`: `structured array-like`
        The results object returned by `pyblp.solve`.

   """

    model_formulations: Sequence[Optional[ModelFormulation]]
    cost_formulation: Formulation
    instrument_formulation: Formulation
    markups: RecArray
    unique_market_ids: Array
    unique_nesting_ids: Array
    unique_product_ids: Array
    T: int
    N: int
    Dict_K: Dict[Union[str, tuple], Tuple[Optional[Array]]] = {}
    M: int
    EC: int
    H: int
    L: int
    _market_indices: Dict[Hashable, int]
    _product_market_indices: Dict[Hashable, Array]
    _max_J: int
    _absorb_cost_ids: Optional[Absorb]

    def __init__(
            self, cost_formulation: Formulation, instrument_formulation: Sequence[Formulation],
            product_data: Mapping, demand_results: Mapping, model_formulations: Sequence[ModelFormulation] = None,
            markup_data: Optional[RecArray] = None) -> None:
        """Initialize the underlying economy with product and agent data before absorbing fixed effects."""

        output("Initializing the problem ...")
        start_time = time.time()

        if markup_data is None:
            M = len(model_formulations)
        else:
            M = np.shape(markup_data)[0]

        if hasattr(instrument_formulation, '__len__'):
            L = len(instrument_formulation)
        else:
            L = 1

        if not isinstance(cost_formulation, Formulation):
            raise TypeError("cost_formulation must be a single Formulation instance.")

        if L == 1:
            if not isinstance(instrument_formulation, Formulation):
                raise TypeError("instrument_formulation must be a single Formulation instance.")
        elif L > 1:
            if not all(isinstance(f, Formulation) for f in instrument_formulation):
                raise TypeError("Each formulation in instrument_formulation must be a Formulation.")

        products = Products(
            cost_formulation=cost_formulation, instrument_formulation=instrument_formulation, product_data=product_data
        )
        if markup_data is None:
            models = Models(model_formulations=model_formulations, product_data=product_data)
            markups = [None] * M
        else:
            models = None
            markups = markup_data

        super().__init__(products, models)

        self.cost_formulation = cost_formulation
        self.instrument_formulation = instrument_formulation
        self.model_formulations = model_formulations
        self.demand_results = demand_results
        self.markups = markups

        self.unique_market_ids = np.unique(self.products.market_ids.flatten())
        self.unique_nesting_ids = np.unique(self.products.nesting_ids.flatten())
        self.unique_product_ids = np.unique(self.products.product_ids.flatten())

        self.N = self.products.shape[0]
        self.T = self.unique_market_ids.size
        self.L = len(self.instrument_formulation) if hasattr(self.instrument_formulation, '__len__') else 1
        for instrument in range(self.L):
            self.Dict_K.update({"K{0}".format(instrument): self.products["Z{0}".format(instrument)].shape[1]})
        self.M = len(self.model_formulations) if self.markups[0] is None else np.shape(self.markups)[0]
        self.EC = self.products.cost_ids.shape[1]
        self.H = self.unique_nesting_ids.size

        self._market_indices = {t: i for i, t in enumerate(self.unique_market_ids)}
        self._product_market_indices = get_indices(self.products.market_ids)
        self._max_J = max(i.size for i in self._product_market_indices.values())

        self._absorb_cost_ids = None
        if self.EC > 0:
            assert cost_formulation is not None
            self._absorb_cost_ids = cost_formulation._build_absorb(self.products.cost_ids)

        if max(options.collinear_atol, options.collinear_rtol) > 0:
            cost_shifters = self.products.w
            common_message = "To disable collinearity checks, set options.collinear_atol = options.collinear_rtol = 0."
            collinear, successful = precisely_identify_collinearity(cost_shifters)
            if not successful:
                raise ValueError(
                    f"Failed to compute the QR decomposition of w while checking for collinearity issues. "
                    f"{common_message}"
                )
            if collinear.any():
                raise ValueError(
                    f"Detected collinearity issues with w. "
                    f"{common_message}"
                )
            for instrument in range(self.L):
                cost_shifters = self.products.w
                cost_shifters = np.append(cost_shifters, self.products["Z{0}".format(instrument)], axis=1)
                collinear, successful = precisely_identify_collinearity(cost_shifters)
                if not successful:
                    raise ValueError(
                        f"Failed to compute the QR decomposition of [w,z" + str(instrument) + "] while checking for "
                        f"collinearity issues. "
                        f"{common_message}"
                    )
                if collinear.any():
                    raise ValueError(
                        f"Detected collinearity issues with [w,z" + str(instrument) + "]."
                        f"{common_message}"
                    )

            output(f"Initialized the problem after {format_seconds(time.time() - start_time)}.")
            output("")
            output(self)

    def __str__(self) -> str:
        """Format economy information as a string."""
        return "\n\n".join([self._format_dimensions(), self._format_formulations(), self._format_model_formulations()])

    def _format_dimensions(self) -> str:
        """Format information about the nonzero dimensions of the economy as a string."""
        header = []
        values = []
        for key in ['T', 'N', 'M', 'L']:
            value = getattr(self, key)
            if value > 0:
                header.append(f" {key} ")
                values.append(str(value))
        for instrument in range(self.L):
            header.append("d_z{0}".format(instrument))
            values.append(str(self.Dict_K["K{0}".format(instrument)]))
        return format_table(header, values, title="Dimensions")

    def _format_formulations(self) -> str:
        """Format information about the formulations of the economy as a string."""
        named_formulations = [(self._w_formulation, "w: Marginal Cost")]
        for instruments in range(self.L):
            named_formulations.append((
                self.Dict_Z_formulation["_Z{0}_formulation".format(instruments)],
                "z{0}: Instruments".format(instruments)
            ))
        data = []
        for formulations, name in named_formulations:
            if any(formulations):
                data.append([name] + [str(f) for f in formulations])
        max_formulations = max(len(r[1:]) for r in data)
        header = ["Column Indices:"] + [f" {i} " for i in range(max_formulations)]
        return format_table(header, *data, title="Formulations")

    def _format_model_formulations(self) -> str:
        """Format information about the model formulations as a string."""
        data = []
        if self.markups[0] is None:
            data.append(["Model - Downstream"] + [self.models["models_downstream"][i] for i in range(self.M)])
            data.append(["Model - Upstream"] + [self.models["models_upstream"][i] for i in range(self.M)])
            data.append(["Firm IDs - Downstream"] + [self.models["firm_ids_downstream"][i] for i in range(self.M)])
            data.append(["Firm IDs - Upstream"] + [self.models["firm_ids_upstream"][i] for i in range(self.M)])
            data.append(["VI Index"] + [self.models["vertical_integration_index"][i] for i in range(self.M)])
            data.append(["Cost Scaling Column"] + [self.models["cost_scaling_column"][i] for i in range(self.M)])
            data.append(["Unit Tax"] + [self.models["unit_tax_name"][i] for i in range(self.M)])
            data.append(["Advalorem Tax"] + [self.models["advalorem_tax_name"][i] for i in range(self.M)])
            data.append(["Advalorem Payer"] + [self.models["advalorem_payer"][i] for i in range(self.M)])
            data.append(
                ["User Supplied Markups"] + [self.models["user_supplied_markups_name"][i] for i in range(self.M)]
            )
            header = [" "] + [f" {i} " for i in range(self.M)]
        else:
            data.append(["Markups Supplied by User"])
            header = [" "]
        return format_table(header, *data, title="Models")

    def solve(
            self, demand_adjustment: Optional[bool] = False,
            clustering_adjustment: Optional[bool] = False,
            costs_type: Optional[str] = 'linear',
            mc_correction: Optional[Array] = None
    ) -> ProblemResults:
        r"""Solve the problem.

        Given demand estimates from PyBLP, we compute implied markups for each model :math:`m` being tested. Marginal
        cost is a linear function of observed cost shifters and an unobserved shock.

        The rest of the testing procedure is done for each pair of models, for each set of instruments. A GMM measure of
        fit is computed for each model-instrument pair. This measure of fit is used to construct the test statistic.

        Parameters
        ----------
        demand_adjustment: Optional[bool]
            (optional, default is False) Configuration that allows user to specify whether to compute a two-step demand
            adjustment. Options are True or False.
        clustering_adjustment: Optional[str]
            (optional, default is unadjusted) Configuration that specifies whether to compute clustered standard errors.
            Options are True or False.

        Returns
        -------
        `ProblemResults`
            :class:`ProblemResults` of the solved problem.

        """

        output("Solving the problem ...")
        step_start_time = time.time()

        M = self.M
        N = self.N
        L = self.L
        markups = self.markups
        critical_values_power, critical_values_size = read_critical_values_tables()

        self._validate_solve_args(demand_adjustment, clustering_adjustment)

        markups_upstream = np.zeros(M, dtype=options.dtype)
        markups_downstream = np.zeros(M, dtype=options.dtype)
        advalorem_tax_adj = [None] * M
        prices_effective = [None] * M
        markups_effective = [None] * M

        if markups[0] is None:
            output('Computing Markups ...')
            markups, markups_downstream, markups_upstream = self._perturb_and_build_markups()

        marginal_cost = self.products.prices - markups

        unit_tax = self.models["unit_tax"]
        advalorem_tax = self.models["advalorem_tax"]
        cost_scaling = self.models["cost_scaling"]
        for m in range(M):
            condition = self.models["advalorem_payer"][m] == "consumer"
            advalorem_tax_adj[m] = 1 / (1 + advalorem_tax[m]) if condition else (1 - advalorem_tax[m])
            prices_effective[m] = (advalorem_tax_adj[m] * self.products.prices / (1 + cost_scaling[m]) - unit_tax[m])
            markups_effective[m] = (advalorem_tax_adj[m] / (1 + cost_scaling[m])) * markups[m]
            marginal_cost[m] = prices_effective[m] - markups_effective[m]

        if costs_type == "log" and not demand_adjustment:
            if np.any(marginal_cost < 0):
                raise ValueError("Can't generate log costs with negative marginal cost!")
            marginal_cost = np.log(marginal_cost)

        if mc_correction is not None:
            if marginal_cost.shape != mc_correction.shape:
                raise ValueError(
                    "The dimensions of marginal cost correction don't match the dimensions of the marginal cost."
                )
            marginal_cost = marginal_cost + mc_correction

        markups_orthogonal, omega, tau_list = self._prepare_orthogonal_variables(
            M, N, markups_effective, marginal_cost
        )

        gradient_markups = H_prime_wd = H = h_i = h = None
        if demand_adjustment:
            gradient_markups, H_prime_wd, H, h_i, h = self._compute_demand_adjustment_gradient(
                N, advalorem_tax_adj, cost_scaling
            )

        g_list = [None] * L
        Q_list = [None] * L
        RV_numerator_list = [None] * L
        RV_denominator_list = [None] * L
        test_statistic_RV_list = [None] * L
        F_statistic_list = [None] * L
        unscaled_F_statistic_list = [None] * L
        MCS_p_values_list = [None] * L
        rho_list = [None] * L
        F_cv_size_list = [None] * L
        F_cv_power_list = [None] * L
        symbols_size_list = [None] * L
        symbols_power_list = [None] * L

        for instrument in range(L):
            r = self._compute_instrument_results(
                instrument, M, N, omega, demand_adjustment, gradient_markups,
                H_prime_wd, H, h_i, h, clustering_adjustment,
                critical_values_size, critical_values_power
            )
            g_list[instrument] = r['g']
            Q_list[instrument] = r['Q']
            RV_numerator_list[instrument] = r['RV_numerator']
            RV_denominator_list[instrument] = r['RV_denominator']
            test_statistic_RV_list[instrument] = r['rv_test_statistic']
            F_statistic_list[instrument] = r['F']
            unscaled_F_statistic_list[instrument] = r['unscaled_F']
            MCS_p_values_list[instrument] = r['mcs_pvalues']
            rho_list[instrument] = r['rho']
            F_cv_size_list[instrument] = r['F_cv_size']
            F_cv_power_list[instrument] = r['F_cv_power']
            symbols_size_list[instrument] = r['symbols_size']
            symbols_power_list[instrument] = r['symbols_power']

        results = ProblemResults(Progress(
            self, markups, markups_downstream, markups_upstream, markups_orthogonal, marginal_cost,
            tau_list, g_list, Q_list, RV_numerator_list, RV_denominator_list,
            test_statistic_RV_list, F_statistic_list, MCS_p_values_list, rho_list, unscaled_F_statistic_list,
            F_cv_size_list, F_cv_power_list, symbols_size_list, symbols_power_list
        ))
        output(f"Solved the problem after {format_seconds(time.time() - step_start_time)}.")
        output("")
        output(results)
        return results

    def _validate_solve_args(self, demand_adjustment: bool, clustering_adjustment: bool) -> None:
        """Validate arguments passed to solve."""
        if not isinstance(demand_adjustment, bool):
            raise TypeError("demand_adjustment must be a boolean (one of True or False).")
        if not isinstance(clustering_adjustment, bool):
            raise TypeError("clustering_adjustment must be a boolean (one of True or False).")
        if clustering_adjustment and np.shape(self.products.clustering_ids)[1] != 1:
            raise ValueError("product_data.clustering_ids must be specified with clustering_adjustment True.")
        for m in range(self.M):
            if self.model_formulations[m]._user_supplied_markups is not None:
                if clustering_adjustment or demand_adjustment:
                    raise ValueError(
                        "If using own markups, demand_adjustment and clustering_adjustment should be False."
                    )

    def _perturb_and_build_markups(self):
        """Call build_markups with current demand_results and model specifications."""
        return build_markups(
            self.products, self.demand_results, self.models["models_downstream"],
            self.models["ownership_downstream"], self.models["models_upstream"],
            self.models["ownership_upstream"], self.models["vertical_integration"],
            self.models["custom_model_specification"], self.models["user_supplied_markups"],
            self.models["mix_flag"]
        )

    def _prepare_orthogonal_variables(self, M: int, N: int, markups_effective: list, marginal_cost: Array):
        """Absorb fixed effects and residualize markups and marginal costs w.r.t. cost shifters."""
        markups_orthogonal = np.zeros((M, N), dtype=options.dtype)
        marginal_cost_orthogonal = np.zeros((M, N), dtype=options.dtype)
        tau_list = np.zeros((M, self.products.w.shape[1]), dtype=options.dtype)
        omega = np.zeros((M, N), dtype=options.dtype)

        if self._absorb_cost_ids is not None:
            output("Absorbing cost-side fixed effects ...")
            self.products.w, _ = self._absorb_cost_ids(self.products.w)
            for m in range(M):
                value, _ = self._absorb_cost_ids(markups_effective[m])
                markups_orthogonal[m] = np.squeeze(value)
                value, _ = self._absorb_cost_ids(marginal_cost[m])
                marginal_cost_orthogonal[m] = np.squeeze(value)
        else:
            for m in range(M):
                markups_orthogonal[m] = np.squeeze(markups_effective[m])
                marginal_cost_orthogonal[m] = np.squeeze(marginal_cost[m])

        if self.products.w.any():
            for m in range(M):
                res = sm.OLS(markups_orthogonal[m], self.products.w).fit()
                markups_orthogonal[m] = res.resid
                res = sm.OLS(marginal_cost_orthogonal[m], self.products.w).fit()
                omega[m] = res.resid
                tau_list[m] = res.params
        else:
            omega = marginal_cost_orthogonal

        return markups_orthogonal, omega, tau_list

    def _compute_demand_adjustment_gradient(self, N: int, advalorem_tax_adj: list, cost_scaling: list):
        """Compute the finite-difference gradient of markups w.r.t. demand parameters."""
        M = self.M
        ZD = self.demand_results.problem.products.ZD
        WD = self.demand_results.updated_W
        h = self.demand_results.moments
        h_i = ZD * self.demand_results.xi
        K2 = self.demand_results.problem.K2
        D = self.demand_results.problem.D

        XD = self.demand_results.problem.products.X1
        XD_column_names = self.demand_results.problem.products.dtype.fields['X1'][2]
        price_in_linear_parameters = 'prices' in XD_column_names
        if price_in_linear_parameters:
            XD = np.delete(XD, XD_column_names.index('prices'), 1)

        partial_y_theta = (
            np.append(self.demand_results.xi_by_theta_jacobian, -self.demand_results.problem.products.prices, axis=1)
            if price_in_linear_parameters else self.demand_results.xi_by_theta_jacobian
        )

        if self.demand_results.problem.ED > 0:
            partial_y_theta = self.demand_results.problem._absorb_demand_ids(partial_y_theta)
            partial_y_theta = np.reshape(
                partial_y_theta[0], [N, len(self.demand_results.theta) + int(price_in_linear_parameters)]
            )

        if not XD.shape[1]:
            partial_xi_theta = partial_y_theta
        else:
            try:
                product = XD @ inv(XD.T @ ZD @ WD @ ZD.T @ XD) @ (XD.T @ ZD @ WD @ ZD.T @ partial_y_theta)
                partial_xi_theta = partial_y_theta - product
            except Exception:
                output(
                    "Dimension mismatch occurred. This can happen if you specify a supply side in the demand "
                    "estimation."
                )

        H = 1 / N * (np.transpose(ZD) @ partial_xi_theta)
        H_prime_wd = np.transpose(H) @ WD

        epsilon = options.finite_differences_epsilon
        gradient_markups = np.zeros(
            (M, N, len(self.demand_results.theta) + int(price_in_linear_parameters)), dtype=options.dtype
        )
        theta_index = 0
        delta_estimate = self.demand_results.delta

        def apply_tax_adjustment(markups_list):
            for m in range(M):
                markups_list[m] = (advalorem_tax_adj[m] * markups_list[m]) / (1 + cost_scaling[m])
            return markups_list

        # perturb sigma parameters
        for (i, j) in itertools.product(range(K2), range(K2)):
            if not self.demand_results.sigma[i, j] == 0:
                sigma_initial = self.demand_results.sigma[i, j]
                self.demand_results._sigma[i, j] = sigma_initial - epsilon / 2
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    self.demand_results._delta = self.demand_results.compute_delta()
                markups_l, _, _ = self._perturb_and_build_markups()
                self.demand_results._sigma[i, j] = sigma_initial + epsilon / 2
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    self.demand_results._delta = self.demand_results.compute_delta()
                markups_u, _, _ = self._perturb_and_build_markups()
                gradient_markups = self._compute_first_difference_markups(
                    apply_tax_adjustment(markups_u), apply_tax_adjustment(markups_l), epsilon, theta_index,
                    gradient_markups
                )
                self.demand_results._sigma[i, j] = sigma_initial
                theta_index += 1

        # perturb pi parameters
        for (i, j) in itertools.product(range(K2), range(D)):
            if not self.demand_results.pi[i, j] == 0:
                pi_initial = self.demand_results.pi[i, j]
                markups_l, _, _ = self._compute_perturbation(i, j, pi_initial - epsilon / 2)
                markups_u, _, _ = self._compute_perturbation(i, j, pi_initial + epsilon / 2)
                gradient_markups = self._compute_first_difference_markups(
                    apply_tax_adjustment(markups_u), apply_tax_adjustment(markups_l), epsilon, theta_index,
                    gradient_markups
                )
                self.demand_results._pi[i, j] = pi_initial
                theta_index += 1

        self.demand_results._delta = delta_estimate

        # perturb alpha (price coefficient in linear parameters)
        price_index = [idx for idx, v in enumerate(self.demand_results.beta_labels) if v == 'prices']
        if price_index:
            alpha_initial = self.demand_results.beta[price_index].copy()
            self.demand_results._beta[price_index] = alpha_initial - epsilon / 2
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                self.demand_results._delta = self.demand_results.compute_delta()
            markups_l, _, _ = self._perturb_and_build_markups()
            self.demand_results._beta[price_index] = alpha_initial + epsilon / 2
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                self.demand_results._delta = self.demand_results.compute_delta()
            markups_u, _, _ = self._perturb_and_build_markups()
            gradient_markups = self._compute_first_difference_markups(
                apply_tax_adjustment(markups_u), apply_tax_adjustment(markups_l), epsilon, theta_index, gradient_markups
            )
            self.demand_results._beta[price_index] = alpha_initial
            theta_index += 1

        # perturb rho (nesting parameter)
        if len(self.demand_results.rho) != 0:
            rho_initial = self.demand_results.rho.copy()
            self.demand_results._rho = rho_initial - epsilon / 2
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                self.demand_results._delta = self.demand_results.compute_delta()
            markups_l, _, _ = self._perturb_and_build_markups()
            self.demand_results._rho = rho_initial + epsilon / 2
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                self.demand_results._delta = self.demand_results.compute_delta()
            markups_u, _, _ = self._perturb_and_build_markups()
            gradient_markups = self._compute_first_difference_markups(
                apply_tax_adjustment(markups_u), apply_tax_adjustment(markups_l), epsilon, theta_index, gradient_markups
            )
            self.demand_results._rho = rho_initial

        return gradient_markups, H_prime_wd, H, h_i, h

    def _compute_instrument_results(
            self, instrument: int, M: int, N: int, omega: Array,
            demand_adjustment: bool, gradient_markups: Optional[Array],
            H_prime_wd: Optional[Array], H: Optional[Array],
            h_i: Optional[Array], h: Optional[Array],
            clustering_adjustment: bool,
            critical_values_size: Array, critical_values_power: Array
    ) -> dict:
        """Compute all test statistics for a single instrument set."""
        instruments = self.products["Z{0}".format(instrument)]
        K = np.shape(instruments)[1]

        if self._absorb_cost_ids is not None:
            Z_orthogonal, _ = self._absorb_cost_ids(instruments)
        else:
            Z_orthogonal = instruments
        if self.products.w.any():
            Z_orthogonal = np.reshape(sm.OLS(Z_orthogonal, self.products.w).fit().resid, [N, K])

        W_inverse = np.reshape(1 / N * (Z_orthogonal.T @ Z_orthogonal), [K, K])
        weight_matrix = np.linalg.pinv(W_inverse)

        # GMM moments and fit for each model
        g = np.zeros((M, K), dtype=options.dtype)
        Q = np.zeros(M, dtype=options.dtype)
        for m in range(M):
            g[m] = 1 / N * (Z_orthogonal.T @ omega[m])
            Q[m] = g[m].T @ weight_matrix @ g[m]

        # RV numerator
        test_statistic_numerator = np.zeros((M, M))
        for m in range(M):
            for i in range(m):
                test_statistic_numerator[i, m] = math.sqrt(N) * (Q[i] - Q[m])

        # psi for each model and RV denominator
        W_12 = fractional_matrix_power(weight_matrix, 0.5)
        W_34 = fractional_matrix_power(weight_matrix, 0.75)
        psi = np.zeros((M, N, K), dtype=options.dtype)
        if demand_adjustment:
            adjustment_value = np.zeros((M, K, H_prime_wd.shape[1]), dtype=options.dtype)
        for m in range(M):
            psi_bar = W_12 @ g[m] - .5 * W_34 @ W_inverse @ W_34 @ g[m]
            W_34_Zg = (Z_orthogonal @ W_34 @ g[m])[:, np.newaxis]
            mc_col = omega[m][:, np.newaxis]
            psi_i = (mc_col * Z_orthogonal) @ W_12 - 0.5 * W_34_Zg * (Z_orthogonal @ W_34.T)
            psi[m] = psi_i - np.transpose(psi_bar)
            if demand_adjustment:
                G_k = -1 / N * np.transpose(Z_orthogonal) @ gradient_markups[m]
                adjustment_value[m] = W_12 @ G_k @ inv(H_prime_wd @ H) @ H_prime_wd
                psi[m] = psi[m] - (h_i - np.transpose(h)) @ np.transpose(adjustment_value[m])

        test_statistic_denominator = np.zeros((M, M))
        covariance_mc = np.zeros((M, M))
        for m in range(M):
            for i in range(m):
                variance_covariance = self._compute_variance_covariance(m, i, N, clustering_adjustment, psi)
                weighted_variance = W_12 @ variance_covariance @ W_12
                operations = np.array([1, 1, -2])
                moments = np.array([
                    g[i].T @ weighted_variance[0] @ g[i],
                    g[m].T @ weighted_variance[1] @ g[m],
                    g[i].T @ weighted_variance[2] @ g[m]
                ]).flatten()
                covariance_mc[i, m] = moments[2]
                covariance_mc[m, i] = moments[2]
                covariance_mc[m, m] = moments[1]
                covariance_mc[i, i] = moments[0]
                test_statistic_denominator[i, m] = math.sqrt(4 * (operations.T @ moments))

        # RV test statistic (upper triangle only; lower triangle and diagonal are NaN)
        rv_test_statistic = np.full((M, M), np.nan)
        for m in range(M):
            for i in range(m):
                rv_test_statistic[i, m] = test_statistic_numerator[i, m] / test_statistic_denominator[i, m]

        # F statistics
        phi = np.zeros([M, N, K])
        for m in range(M):
            ols_results = sm.OLS(omega[m], Z_orthogonal).fit()
            e = np.reshape(ols_results.resid, [N, 1])
            phi[m] = (e * Z_orthogonal) @ weight_matrix
            if demand_adjustment:
                phi[m] = phi[m] - (h_i - np.transpose(h)) @ np.transpose(W_12 @ adjustment_value[m])

        unscaled_F = np.zeros((M, M))
        F = np.full((M, M), np.nan)
        rho = np.zeros((M, M))
        F_cv_size = np.empty((M, M), dtype=object)
        F_cv_power = np.empty((M, M), dtype=object)
        symbols_size = np.empty((M, M), dtype=object)
        symbols_power = np.empty((M, M), dtype=object)

        for (m, i) in itertools.product(range(M), range(M)):
            if i < m:
                variance = self._compute_variance_covariance(m, i, N, clustering_adjustment, phi)
                sigma = 1 / K * np.array([
                    np.trace(variance[0] @ W_inverse),
                    np.trace(variance[1] @ W_inverse),
                    np.trace(variance[2] @ W_inverse)
                ])
                numerator_sqrt = sigma[0] - sigma[1]
                denominator_sqrt = np.sqrt((sigma[0] + sigma[1]) ** 2 - 4 * sigma[2] ** 2)
                rho[i, m] = numerator_sqrt / denominator_sqrt
                rho_squared = rho[i, m] ** 2

                operations = np.array([sigma[1], sigma[0], -2 * sigma[2]])
                moments = np.array([
                    g[i].T @ weight_matrix @ g[i],
                    g[m].T @ weight_matrix @ g[m],
                    g[i].T @ weight_matrix @ g[m]
                ]).flatten()
                F_numerator = operations @ moments
                F_denominator = sigma[0] * sigma[1] - sigma[2] ** 2
                unscaled_F[i, m] = N / (2 * K) * F_numerator / F_denominator
                F[i, m] = (1 - rho_squared) * unscaled_F[i, m]

                rho_lookup = min(np.round(np.abs(rho[i, m]), 2), 0.99)
                K_lookup = K if K <= 30 else 30
                ind = np.where(
                    (critical_values_size['K'] == K_lookup) & (critical_values_size['rho'] == rho_lookup)
                )[0][0]
                F_cv_size[i, m] = np.array([
                    critical_values_size['r_125'][ind],
                    critical_values_size['r_10'][ind],
                    critical_values_size['r_075'][ind]
                ], dtype=object)
                F_cv_power[i, m] = np.array([
                    critical_values_power['r_50'][ind],
                    critical_values_power['r_75'][ind],
                    critical_values_power['r_95'][ind]
                ], dtype=object)

                symbols_size[i, m] = (
                    " " if F[i, m] < F_cv_size[i, m][0] else
                    "*" if F[i, m] < F_cv_size[i, m][1] else
                    "**" if F[i, m] < F_cv_size[i, m][2] else "***"
                )
                symbols_power[i, m] = (
                    " " if F[i, m] < F_cv_power[i, m][0] else
                    "^" if F[i, m] < F_cv_power[i, m][1] else
                    "^^" if F[i, m] < F_cv_power[i, m][2] else "^^^"
                )
            else:
                symbols_size[i, m] = ""
                symbols_power[i, m] = ""

        # model confidence set
        all_model_combinations = list(itertools.combinations(range(M), 2))
        n_combinations = len(all_model_combinations)
        model_confidence_set_variance = np.zeros([n_combinations, 1])
        sigma_mcs = np.zeros([n_combinations, n_combinations])
        for index_i, model_i in enumerate(all_model_combinations):
            model_confidence_set_variance[index_i] = test_statistic_denominator[model_i[0], model_i[1]] / 2
            for index_j, model_j in enumerate(all_model_combinations):
                term1 = covariance_mc[model_i[0], model_j[0]] - covariance_mc[model_i[1], model_j[0]]
                term2 = covariance_mc[model_i[0], model_j[1]] - covariance_mc[model_i[1], model_j[1]]
                sigma_mcs[index_j, index_i] = term1 - term2
        sigma_mcs = sigma_mcs / (model_confidence_set_variance @ model_confidence_set_variance.T)

        mcs_pvalues = self._compute_mcs(
            rv_test_statistic, sigma_mcs, model_confidence_set_variance, M, all_model_combinations
        )

        return {
            'g': g, 'Q': Q,
            'RV_numerator': test_statistic_numerator,
            'RV_denominator': test_statistic_denominator,
            'rv_test_statistic': rv_test_statistic,
            'F': F, 'unscaled_F': unscaled_F,
            'mcs_pvalues': mcs_pvalues,
            'rho': rho,
            'F_cv_size': F_cv_size,
            'F_cv_power': F_cv_power,
            'symbols_size': symbols_size,
            'symbols_power': symbols_power,
        }

    def _compute_mcs(
            self, rv_test_statistic: Array, sigma_mcs: Array,
            model_confidence_set_variance: Array, M: int, all_model_combinations: list
    ) -> Array:
        """Compute model confidence set p-values by iteratively eliminating the worst-fitting model."""
        model_confidence_set = np.array(range(M))
        mcs_pvalues = np.ones([M, 1])
        converged = False
        while not converged:
            if np.shape(model_confidence_set)[0] == 2:
                max_test_statistic = rv_test_statistic[model_confidence_set[0], model_confidence_set[1]]
                if np.sign(max_test_statistic) >= 0:
                    worst_fit = model_confidence_set[0]
                    max_test_statistic = -max_test_statistic
                else:
                    worst_fit = model_confidence_set[1]
                mcs_pvalues[worst_fit] = 2 * norm.cdf(max_test_statistic)
                converged = True
            else:
                current_combinations = list(itertools.combinations(model_confidence_set, 2))
                sigma_index = np.array(
                    [all_model_combinations.index(c) for c in current_combinations], dtype=int
                )
                model_1 = [c[0] for c in current_combinations]
                model_2 = [c[1] for c in current_combinations]
                test_stats = rv_test_statistic[model_1, model_2]
                index = np.argmax(abs(test_stats))
                max_test_statistic = test_stats[index]

                if np.sign(max_test_statistic) >= 0:
                    worst_fit = model_1[index]
                else:
                    worst_fit = model_2[index]
                    max_test_statistic = -max_test_statistic

                cov = sigma_mcs[sigma_index[:, None], sigma_index]
                simulated = np.random.multivariate_normal(np.zeros(len(current_combinations)), cov, options.ndraws)
                mcs_pvalues[worst_fit] = np.mean(np.amax(abs(simulated), 1) > max_test_statistic)
                model_confidence_set = np.delete(
                    model_confidence_set, np.where(model_confidence_set == worst_fit)
                )
        return mcs_pvalues

    def _compute_first_difference_markups(
            self, markups_u: list, markups_l: list, epsilon: float, theta_index: int, gradient_markups: Array
    ) -> Array:
        """Compute finite-difference approximation of markup gradient w.r.t. a single demand parameter."""
        for m in range(self.M):
            diff_markups = (markups_u[m] - markups_l[m]) / epsilon
            if self._absorb_cost_ids is not None:
                diff_markups, _ = self._absorb_cost_ids(diff_markups)
                if self.products.w.any():
                    gradient_markups[m][:, theta_index] = sm.OLS(diff_markups, self.products.w).fit().resid
                else:
                    gradient_markups[m][:, theta_index] = np.squeeze(diff_markups)
        return gradient_markups

    def _compute_perturbation(self, i: int, j: int, perturbation: float):
        """Perturb pi[i, j] to the given value, recompute delta, and return new markups."""
        self.demand_results._pi[i, j] = perturbation
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            self.demand_results._delta = self.demand_results.compute_delta()
        return self._perturb_and_build_markups()

    def _compute_variance_covariance(
            self, m: int, i: int, N: int, clustering_adjustment: bool, var: Array
    ) -> Array:
        """Compute the sandwich variance-covariance matrix for model pair (m, i)."""
        variance_covariance = 1 / N * np.array([
            var[i].T @ var[i], var[m].T @ var[m], var[i].T @ var[m]
        ])
        if clustering_adjustment:
            cluster_ids = np.unique(self.products.clustering_ids)
            for j in cluster_ids:
                index = np.where(self.products.clustering_ids == j)[0]
                var1_l = var[i][index, :]
                var2_l = var[m][index, :]
                var1_c = var1_l
                var2_c = var2_l
                for k in range(len(index) - 1):
                    var1_c = np.roll(var1_c, 1, axis=0)
                    var2_c = np.roll(var2_c, 1, axis=0)
                    variance_covariance = variance_covariance + 1 / N * np.array([
                        var1_l.T @ var1_c, var2_l.T @ var2_c, var1_l.T @ var2_c
                    ])
        return variance_covariance


@dataclass
class Progress:
    """Structured information passed from Problem.solve to ProblemResults."""
    problem: 'Problem'
    markups: Array
    markups_downstream: Array
    markups_upstream: Array
    markups_orthogonal: Array
    marginal_cost: Array
    tau_list: Array
    g: Array
    Q: Array
    RV_numerator: Array
    RV_denominator: Array
    test_statistic_RV: Array
    F: Array
    MCS_pvalues: Array
    rho: Array
    unscaled_F: Array
    F_cv_size_list: Array
    F_cv_power_list: Array
    symbols_size_list: Array
    symbols_power_list: Array
