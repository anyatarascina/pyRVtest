"""PyBLPBackend: wraps a pyblp.ProblemResults behind the DemandBackend protocol.

Placeholder for v0.4 step 3. Will encapsulate the private-attribute access
currently in `_compute_demand_adjustment_gradient` (reads/writes of
`_sigma`, `_pi`, `_beta`, `_rho`, `_delta`) behind a class so the rest of
pyRVtest doesn't depend on PyBLP internals.
"""

__all__: list[str] = []
