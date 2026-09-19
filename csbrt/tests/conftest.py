"""Shared pytest configuration for the csbrt test suite.

Makes the ``csbrt`` package importable from ``csbrt/src`` without requiring a
full ``pip install`` (the scientific stack is not installed in CI/developer
environments).
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
