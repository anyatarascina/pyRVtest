"""Demand-adjustment first-stage correction (DMSS 2024 Appendix C eq. 77).

Placeholder for v0.4 step 4. Will host a SINGLE implementation generic
over DemandBackend, replacing the two parallel paths
`Problem._compute_analytical_demand_adjustment` and
`Problem._compute_demand_adjustment_gradient`. These are the paths
that had the three b3b08a3 bugs (sign error, missing concentration
adjustment, updated_W vs W).
"""

__all__: list[str] = []
