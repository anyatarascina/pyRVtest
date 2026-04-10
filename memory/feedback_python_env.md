---
name: Python environment
description: Always use the pyRVenv conda environment when running Python commands
type: feedback
---

Always invoke Python via `/home/md/anaconda3/envs/pyRVenv/bin/python` (or `conda run -n pyRVenv python`). pyblp and pyRVtest dependencies are installed there, not in the system Python.

**Why:** pyblp is installed in `/home/md/anaconda3/envs/pyRVenv/lib/python3.14/site-packages`. Running with system `python` or `python3` fails with ModuleNotFoundError.

**How to apply:** Every `Bash` call that runs Python should use `/home/md/anaconda3/envs/pyRVenv/bin/python` or prefix with `conda run -n pyRVenv`.
