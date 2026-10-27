"""Forbidden-method stub for the CC topology policy family.

``utils/cache_utils.py`` is the authoritative Qwen DRM file and is kept byte-identical,
so its unconditional ``from bridgekv.cc_topology_policies import ...`` must resolve.
The CC sources were intentionally not exported in the authoritative bundle, and
protocol LLAMA_DRM_A100_4GPU_V1 forbids CC methods outright.

Exporting an empty ``CC_TOPOLOGY_METHODS`` makes every ``in CC_TOPOLOGY_METHODS``
branch in cache_utils statically unreachable, so the dispatchers below act as
tripwires rather than fallbacks: reaching them means a forbidden method leaked into
the run and the run must fail loudly instead of silently degrading.
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet

PROTOCOL_ID = "LLAMA_DRM_A100_4GPU_V1"

CC_TOPOLOGY_METHODS: FrozenSet[str] = frozenset()

FORBIDDEN_REASON = (
    "CC topology methods are forbidden under protocol LLAMA_DRM_A100_4GPU_V1 "
    "(LLaMA + DRM methods only). The CC sources are not part of the authoritative bundle."
)


class ForbiddenCCMethodError(RuntimeError):
    pass


def dispatch_cc_topology_policy(method: str, *args: Any, **kwargs: Any):
    raise ForbiddenCCMethodError(f"{FORBIDDEN_REASON} Requested method: {method!r}")


def load_cc_gate_config(path: Any = None) -> Dict[str, Any]:
    raise ForbiddenCCMethodError(f"{FORBIDDEN_REASON} Requested cc_gate_config_path: {path!r}")
