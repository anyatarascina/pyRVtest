"""LaborSupplyBackend: nested-logit-style labor-supply demand backend (skeleton).

v0.4 step 14b. This is a deliberately-bare skeleton: the class shape,
constructor signature, and the ``DemandBackend`` protocol members exist
so downstream wiring (``Problem(market_side='labor')``, the
``_construct_demand_backend`` dispatch, pickling round-trips, test
imports) is exercised. The analytical labor-supply Jacobian / Hessian
are deferred to v0.5 when real labor-project data is available.

``LaborSupplyBackend`` implements the core :class:`DemandBackend`
protocol only. It does NOT implement
:class:`SupportsDemandAdjustment`; extending it for labor-side
first-stage adjustment is an explicit v0.5 task. Users who want
demand adjustment on a labor supply system today should wrap their own
backend (see :class:`pyRVtest.backends.UserSuppliedBackend` and
``docs/custom_demand.rst``).

The shape mirrors :class:`pyRVtest.backends.NestedLogitBackend` so that
the v0.5 implementer can reuse the same constructor plumbing and fill
``compute_jacobian`` / ``compute_hessian`` in place.

Typical v0.5 usage (target shape):

    from pyRVtest.backends.labor import LaborSupplyBackend

    backend = LaborSupplyBackend(
        alpha=1.2,                          # upward-sloping supply: alpha > 0.
        rho=[0.4],                          # nesting parameter(s).
        product_data=labor_supply_table,
        wage_column='wages',
        employment_column='employment_share',
    )
    Problem(market_side='labor', demand_backend=backend, ...)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Hashable, Iterator, List, Mapping, Optional

from numpy.typing import NDArray
from typing_extensions import TypeAlias


__all__ = ['LaborSupplyBackend']


_NDArray: TypeAlias = NDArray[Any]


_V05_DEFERRAL_FIX = (
    "Fix: full implementation deferred to v0.5 when labor-project data "
    "is available. See .claude/plans/v0.4-refactor.md §4.5 for the "
    "schedule. Short-term workarounds: (a) use a "
    "pyRVtest.backends.UserSuppliedBackend wrapping a manually computed "
    "labor-supply Jacobian; (b) use Monopsony / BertrandWages / "
    "CournotEmployment with a precomputed demand_jacobian via the "
    "UserSuppliedBackend path; (c) wait for the v0.5 release."
)


class LaborSupplyBackend:
    """Skeleton labor-supply demand backend (nested logit shape).

    v0.4 ships the class shape and protocol surface; the Jacobian /
    Hessian math is deferred to v0.5 when labor-project data ships.

    Parameters
    ----------
    alpha
        Coefficient on the wage in the labor-supply utility. For
        upward-sloping supply this is positive (symmetric flip of the
        product-side ``alpha`` sign convention).
    rho
        Nesting parameter(s), one per nest level. Empty list or ``None``
        is equivalent to plain logit labor supply.
    product_data
        Record / mapping with per-worker-observation rows. Expected
        columns are discovered at solve time (wages, employment share,
        nesting ids). Specific names can be configured via
        ``wage_column`` / ``employment_column`` (defaults ``'wages'`` /
        ``'employment_share'``) matching Open Question 6 of the plan.
    wage_column
        Column name for the wage vector. Default ``'wages'``.
    employment_column
        Column name for the employment-share vector. Default
        ``'employment_share'``.

    Raises
    ------
    NotImplementedError
        :meth:`compute_jacobian` and :meth:`compute_hessian` raise with
        a pointer to v0.5. The constructor itself is non-raising so that
        tests, migration guides, and Problem wiring can exercise the
        class without blocking on data.

    Examples
    --------
    >>> import numpy as np
    >>> from pyRVtest.backends.labor import LaborSupplyBackend
    >>> backend = LaborSupplyBackend(
    ...     alpha=1.2,
    ...     product_data={
    ...         'market_ids': np.array([0, 0]),
    ...         'wages': np.array([1.0, 2.0]),
    ...         'employment_share': np.array([0.3, 0.3]),
    ...     },
    ... )
    >>> backend.n_parameters
    1
    >>> try:
    ...     backend.compute_jacobian(market_id=0)
    ... except NotImplementedError as e:
    ...     print('v0.5 deferral:', str(e).split('. ', 1)[0] + '.')
    v0.5 deferral: LaborSupplyBackend.compute_jacobian is not implemented in v0.4.
    """

    def __init__(
            self,
            alpha: float,
            rho: Optional[List[float]] = None,
            product_data: Optional[Mapping[str, Any]] = None,
            wage_column: str = 'wages',
            employment_column: str = 'employment_share',
    ) -> None:
        self._alpha = float(alpha)
        self._rho: List[float] = list(rho) if rho is not None else []
        self._product_data = product_data
        self._wage_column = wage_column
        self._employment_column = employment_column

    # ------------------------------------------------------------------
    # DemandBackend protocol: identification.
    # ------------------------------------------------------------------

    @property
    def n_parameters(self) -> int:
        return 1 + len(self._rho)

    @property
    def theta_names(self) -> List[str]:
        names: List[str] = ['alpha']
        for i, _ in enumerate(self._rho):
            names.append(f'rho[{i}]')
        return names

    # ------------------------------------------------------------------
    # DemandBackend protocol: primitives.
    # ------------------------------------------------------------------

    def compute_jacobian(self, market_id: Optional[Hashable] = None) -> _NDArray:
        """Return ∂s/∂w for labor supply (upward-sloping).

        v0.4 skeleton stub. The v0.5 implementation should follow the
        same analytical-derivative structure as
        :class:`pyRVtest.backends.NestedLogitBackend`, mirrored with
        the upward-supply sign convention (diagonal entries positive,
        off-diagonal entries negative within a firm or nest).
        """
        raise NotImplementedError(
            "LaborSupplyBackend.compute_jacobian is not implemented in v0.4. "
            "The class is a skeleton: constructor and protocol members exist "
            "so Problem(market_side='labor') wiring can be validated, but "
            "the analytical Jacobian is deferred until labor-project data "
            "is available.\n" +
            _V05_DEFERRAL_FIX
        )

    def compute_hessian(self, market_id: Hashable) -> Optional[_NDArray]:
        """Return ∂²s/∂w² for labor supply, shape (J_t, J_t, J_t).

        v0.4 skeleton stub. Required by vertical-integration passthrough
        on the labor side (joint product + labor conduct testing, which
        is explicitly a v0.5+ goal per plan §2.2 deferred list).
        """
        raise NotImplementedError(
            "LaborSupplyBackend.compute_hessian is not implemented in v0.4. "
            "Joint product + labor conduct testing (which is where the "
            "labor Hessian matters, via cross-side vertical passthrough) "
            "is explicitly on the v0.5+ defer list.\n" +
            _V05_DEFERRAL_FIX
        )

    # ------------------------------------------------------------------
    # DemandBackend protocol: perturbation.
    # ------------------------------------------------------------------

    @contextmanager
    def perturbed(self, theta_index: int, delta: float) -> Iterator['LaborSupplyBackend']:
        """Context manager: theta[theta_index] += delta; restore on exit.

        v0.4 skeleton stub. Implementing perturbation requires the same
        Jacobian machinery as :meth:`compute_jacobian`; implementers
        should populate both together.
        """
        raise NotImplementedError(
            "LaborSupplyBackend.perturbed is not implemented in v0.4. "
            "Finite-difference demand adjustment requires a working "
            "compute_jacobian; the two must ship together.\n" +
            _V05_DEFERRAL_FIX
        )
        # Unreachable; present only so mypy sees a yield statement and
        # classifies the method as a generator.
        yield self  # pragma: no cover
