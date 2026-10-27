"""Import shim for ``step1_diagnose_llama.py`` (module lives under EigenAttn_base/diagnose)."""
from __future__ import annotations

import sys
from pathlib import Path

_DIAG = Path(__file__).resolve().parents[1] / "EigenAttn_base" / "diagnose"
if str(_DIAG) not in sys.path:
    sys.path.insert(0, str(_DIAG))

from step1 import *  # noqa: F401,F403
from step1 import _unpatch as unpatch  # noqa: F401
