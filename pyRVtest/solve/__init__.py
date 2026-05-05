"""Testing-pipeline stages: markups, orthogonalize, demand adjustment, test engine.

Each module in
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

from typing import List

# Use ``List[str]`` (typing) rather than ``list[str]`` (PEP 585) so the
# package-level import chain works on Python 3.7/3.8, matching the
# ``python_requires='>=3.7'`` declaration in setup.py.
__all__: List[str] = []
