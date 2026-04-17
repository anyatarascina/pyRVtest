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

from typing import Optional, Union

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

        .. deprecated:: v0.4
            Prefer Problem-level taxes:
            ``Problem(..., unit_tax='col', advalorem_tax='col',
            advalorem_payer='firm')``. The model-level fields remain
            for backward compatibility but emit a DeprecationWarning.
    unit_tax_salient, advalorem_tax_salient : bool, optional
        v0.4: opt-out flags for Problem-level taxes at the vertical
        level. Default ``True``; set to ``False`` to make this Vertical
        ignore the Problem-level tax (salience-test mechanism).
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
            cost_scaling: Optional[Union[str, float, int]] = None,
            user_supplied_markups: Optional[str] = None,
            unit_tax_salient: bool = True,
            advalorem_tax_salient: bool = True,
    ) -> None:
        if not isinstance(downstream, ConductModel):
            raise TypeError(
                f"Expected the downstream argument to Vertical(...) to be a "
                f"ConductModel instance (e.g., Bertrand, Cournot, Monopoly). "
                f"Received {type(downstream).__name__}. "
                f"Fix: wrap the downstream specification in the appropriate "
                f"ConductModel subclass: downstream must be a ConductModel."
            )
        if not isinstance(upstream, ConductModel):
            raise TypeError(
                f"Expected the upstream argument to Vertical(...) to be a "
                f"ConductModel instance (e.g., Monopoly, Bertrand, Cournot). "
                f"Received {type(upstream).__name__}. "
                f"Fix: wrap the upstream specification in the appropriate "
                f"ConductModel subclass: upstream must be a ConductModel."
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
                        f"Expected {field!r} to be set on the Vertical(...) "
                        f"wrapper, not on the inner {tier_name} ConductModel "
                        f"(shared config belongs to the combined vertical model). "
                        f"Received {field}={getattr(tier, field)!r} on the "
                        f"{tier_name} tier. "
                        f"Fix: move {field}={getattr(tier, field)!r} to "
                        f"Vertical(...) as a keyword argument."
                    )
        self.downstream = downstream
        self.upstream = upstream
        self.vertical_integration = vertical_integration
        self.unit_tax = unit_tax
        self.advalorem_tax = advalorem_tax
        self.advalorem_payer = advalorem_payer
        self.cost_scaling = cost_scaling
        self.user_supplied_markups = user_supplied_markups
        # v0.4 OQ 14: per-model salience flags for Problem-level taxes.
        self.unit_tax_salient = unit_tax_salient
        self.advalorem_tax_salient = advalorem_tax_salient
        self._validate_shared_config()

    def _validate_shared_config(self) -> None:
        if isinstance(self.cost_scaling, bool):
            raise TypeError(
                f"Expected cost_scaling on Vertical(...) to be a column-name "
                f"string or a numeric scalar. "
                f"Received bool ({self.cost_scaling!r}). "
                f"Fix: pass a numeric scalar (e.g. 0.5) or a column name."
            )
        if self.cost_scaling is not None and not isinstance(
                self.cost_scaling, (str, float, int)):
            raise TypeError(
                f"Expected cost_scaling on Vertical(...) to be a column-name "
                f"string, a numeric scalar, or None. "
                f"Received {type(self.cost_scaling).__name__}. "
                f"Fix: pass a column name like cost_scaling='lambda_col' or a "
                f"scalar like cost_scaling=0.5."
            )
        if self.advalorem_tax is not None and self.advalorem_payer is None:
            raise TypeError(
                "Expected advalorem_payer to be 'firm' or 'consumer' when "
                "advalorem_tax is supplied on Vertical(...). "
                "Received advalorem_payer=None. "
                "Fix: set advalorem_payer='firm' or 'consumer'."
            )
        if self.advalorem_payer is not None and self.advalorem_payer not in {
                'firm', 'consumer', 'firms', 'consumers'}:
            raise TypeError(
                f"Expected advalorem_payer to be 'firm' or 'consumer' (or None). "
                f"Received {self.advalorem_payer!r}. "
                f"Fix: pass advalorem_payer='firm' or 'consumer'."
            )
        # v0.4 OQ 14: per-model salience flags must be booleans.
        if not isinstance(self.unit_tax_salient, bool):
            raise TypeError(
                f"Expected unit_tax_salient on Vertical(...) to be True or "
                f"False. "
                f"Received {type(self.unit_tax_salient).__name__} "
                f"({self.unit_tax_salient!r}). "
                f"Fix: pass unit_tax_salient=True (default) or False."
            )
        if not isinstance(self.advalorem_tax_salient, bool):
            raise TypeError(
                f"Expected advalorem_tax_salient on Vertical(...) to be "
                f"True or False. "
                f"Received {type(self.advalorem_tax_salient).__name__} "
                f"({self.advalorem_tax_salient!r}). "
                f"Fix: pass advalorem_tax_salient=True (default) or False."
            )

    def __repr__(self) -> str:
        return (
            f'Vertical(downstream={self.downstream!r}, '
            f'upstream={self.upstream!r}, '
            f'vertical_integration={self.vertical_integration!r})'
        )
