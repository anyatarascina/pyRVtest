"""Shared fixtures for pyRVtest tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the package is importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
