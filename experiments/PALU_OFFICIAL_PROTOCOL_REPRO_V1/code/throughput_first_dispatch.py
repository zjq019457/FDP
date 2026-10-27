#!/usr/bin/env python3
"""Independent memory-safe throughput dispatcher.

It coexists with the legacy conservative dispatcher. It never stops or
restarts existing workers; it only launches a missing quality/calibration cell
when projected memory plus a fixed safety margin fits.
"""
import sys
from pathlib import Path

ROOT = Path("/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1")
sys.path.insert(0, str(ROOT / "code"))
import official_tier_dispatch as base

base.WORKER_PEAK_MIB = 28672
base.RESERVED_MARGIN_MIB = 12288

if __name__ == "__main__":
    base.main()
