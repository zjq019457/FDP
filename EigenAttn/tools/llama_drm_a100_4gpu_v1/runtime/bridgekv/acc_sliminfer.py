"""Forbidden-method stub for the ACC-DRM SlimInfer policy family.

Same rationale as ``cc_topology_policies``: the authoritative ``utils/cache_utils.py``
imports these names unconditionally, the ACC sources were not exported in the
authoritative bundle, and protocol LLAMA_DRM_A100_4GPU_V1 restricts the run to the
five approved LLaMA DRM methods plus Full-KV.

``ACC_DRM_METHODS`` is empty so the ACC branches in cache_utils are unreachable; the
dispatcher raises so that any leak fails the run rather than silently changing
selection semantics.
"""
from __future__ import annotations

from typing import Any, FrozenSet

PROTOCOL_ID = "LLAMA_DRM_A100_4GPU_V1"

ACC_DRM_METHODS: FrozenSet[str] = frozenset()

FORBIDDEN_REASON = (
    "ACC-DRM SlimInfer methods are forbidden under protocol LLAMA_DRM_A100_4GPU_V1 "
    "(LLaMA + DRM methods only). The ACC sources are not part of the authoritative bundle."
)


class ForbiddenACCMethodError(RuntimeError):
    pass


def dispatch_acc_drm_policy(method: str, *args: Any, **kwargs: Any):
    raise ForbiddenACCMethodError(f"{FORBIDDEN_REASON} Requested method: {method!r}")
