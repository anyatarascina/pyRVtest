"""Vertical model: downstream + upstream conduct with Villas-Boas passthrough.

v0.4 step 5a. ``Vertical`` is a composer class that bundles two
``ConductModel`` instances (downstream and upstream) with the shared
tax / cost-scaling / vertical-integration configuration that applies
to the COMBINED vertical model. It is NOT itself a ``ConductModel``
subclass — it's a structural container that ``Problem`` unpacks at
setup.

Usage::

    Vertical(
        downstream=Bertrand(ownership='firm_ids'),
        upstream=Monopoly(ownership='manufacturer_ids'),
        vertical_integration='vi_col',
        advalorem_tax='tax_col',
        advalorem_payer='firm',
    )

Rationale for a distinct class rather than a ``model_upstream``
keyword on the downstream class: the vertical case shares *config*
(vertical_integration, taxes) that logically belongs on the combined
model, not on either tier individually. Nesting one ``ConductModel``
inside another conflates "conduct math for this tier" with "config
for the combined model." The wrapper makes vertical structure visible
at the call site and each tier's config lives on the right object.

Upstream markups use the Villas-Boas (2007) passthrough matrix; the
computation lives in ``pyRVtest/markups.py::construct_passthrough_matrix``
(legacy pyblp path) and ``_construct_passthrough_from_hessian`` (backend
path) and will be exposed as ``pyRVtest.build_passthrough`` in step 11.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

from .base import ConductModel


__all__ = ['Vertical']


class Vertical:
    """Bilateral oligopoly: downstream and upstream conduct bundled.

    Parameters
    ----------
    downstream : ConductModel
        Conduct for downstream firms (retailers). Typically Bertrand,
        Cournot, or Monopoly. The downstream instance carries its own
        ``ownership`` and (if applicable) ``kappa_specification``.
    upstream : ConductModel
        Conduct for upstream firms (manufacturers). Same set of
        conduct types; carries its own ``ownership`` /
        ``kappa_specification``.
    vertical_integration : str, optional
        Column name indicating which products are vertically integrated
        (e.g., retailer store brands). See ``markups.py``'s
        upstream-markup combination logic.
    unit_tax, advalorem_tax, advalorem_payer, cost_scaling : str, optional
        Taxes / cost scaling applied to the combined vertical model.
    user_supplied_markups : str, optional
        Column name of pre-computed total markups; if supplied, bypasses
        the conduct math entirely (for both tiers).

    Examples
    --------
    >>> from pyRVtest import Vertical, Bertrand, Monopoly
    >>> v = Vertical(
    ...     downstream=Bertrand(ownership='firm_ids'),
    ...     upstream=Monopoly(ownership='manufacturer_ids'),
    ...     vertical_integration='vi_col',
    ... )
    >>> type(v.downstream).__name__
    'Bertrand'
    >>> type(v.upstream).__name__
    'Monopoly'
    >>> v.vertical_integration
    'vi_col'
    """

    def __init__(
            self,
            downstream: ConductModel,
            upstream: ConductModel,
            vertical_integration: Optional[str] = None,
            unit_tax: Optional[str] = None,
            advalorem_tax: Optional[str] = None,
            advalorem_payer: Optional[str] = None,
            cost_scaling: Optional[str] = None,
            user_supplied_markups: Optional[str] = None,
    ) -> None:
        if not isinstance(downstream, ConductModel):
            raise TypeError(
                f"downstream must be a ConductModel instance, got "
                f"{type(downstream).__name__}."
            )
        if not isinstance(upstream, ConductModel):
            raise TypeError(
                f"upstream must be a ConductModel instance, got "
                f"{type(upstream).__name__}."
            )
        # Tax / vertical-integration config should NOT be on the inner
        # conducts in a Vertical; those belong to the combined model.
        for tier_name, tier in (('downstream', downstream), ('upstream', upstream)):
            for field in (
                    'vertical_integration', 'unit_tax', 'advalorem_tax',
                    'advalorem_payer', 'cost_scaling', 'user_supplied_markups',
            ):
                if getattr(tier, field, None) is not None:
                    raise TypeError(
                        f"{field} must be set on the Vertical(...) wrapper, "
                        f"not on the inner {tier_name} ConductModel. Move "
                        f"{field}={getattr(tier, field)!r} to Vertical(...)."
                    )
        self.downstream = downstream
        self.upstream = upstream
        self.vertical_integration = vertical_integration
        self.unit_tax = unit_tax
        self.advalorem_tax = advalorem_tax
        self.advalorem_payer = advalorem_payer
        self.cost_scaling = cost_scaling
        self.user_supplied_markups = user_supplied_markups
        self._validate_shared_config()

    def _validate_shared_config(self) -> None:
        if self.advalorem_tax is not None and self.advalorem_payer is None:
            raise TypeError(
                "advalorem_payer must be 'firm' or 'consumer' when "
                "advalorem_tax is supplied."
            )
        if self.advalorem_payer is not None and self.advalorem_payer not in {
                'firm', 'consumer', 'firms', 'consumers'}:
            raise TypeError(
                f"advalorem_payer must be 'firm' or 'consumer', got "
                f"{self.advalorem_payer!r}."
            )

    def __repr__(self) -> str:
        return (
            f'Vertical(downstream={self.downstream!r}, '
            f'upstream={self.upstream!r}, '
            f'vertical_integration={self.vertical_integration!r})'
        )
