"""Testing-pipeline stages: markups, orthogonalize, demand adjustment, test engine.

Placeholder for v0.4 step 8 (split of `Problem.solve`). Each module in
this subpackage corresponds to a stage of the RV testing pipeline:

  markups.py           - build_markups, evaluate_foc
  passthrough.py       - standalone build_passthrough (step 11)
  orthogonalize.py     - residualize_and_absorb (step 8)
  endogenous_cost.py   - IV correction for scale economies
  demand_adjustment.py - generic over DemandBackend (step 4)
  test_engine.py       - RV, F, MCS, psi (step 8)

Examples
--------
>>> from pyRVtest import solve
>>> solve.__all__
[]
>>> # build_passthrough is re-exported at the package root, not here:
>>> from pyRVtest import build_passthrough
>>> callable(build_passthrough)
True
"""

__all__: list[str] = []
