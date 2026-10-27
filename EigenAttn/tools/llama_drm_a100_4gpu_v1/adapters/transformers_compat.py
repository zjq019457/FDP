#!/usr/bin/env python3
"""Compatibility shim for running the slim DRM LLaMA modeling file on transformers 4.57.3.

The authoritative ``runtime/models/modeling_llama.py`` was exported against an older
transformers release that still exposed a handful of private/typing helpers from
``transformers.utils``.  Importing this module *before* the modeling file patches the
missing symbols back onto the relevant transformers namespaces so that the DRM /
SlimInfer hooks can be imported unchanged.

Nothing here changes model numerics: only typing helpers and re-exports are added.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List, Tuple

PROTOCOL_ID = "LLAMA_DRM_A100_4GPU_V1"

_APPLIED: Dict[str, str] = {}


def _record(name: str, action: str) -> None:
    _APPLIED[name] = action


def _install_loss_kwargs() -> None:
    """`transformers.utils.LossKwargs` was removed in >=4.55; recreate it as a TypedDict."""
    import transformers.utils as tu

    if hasattr(tu, "LossKwargs"):
        _record("transformers.utils.LossKwargs", "already_present")
        return

    LossKwargs = None
    for mod_name in (
        "transformers.utils.generic",
        "transformers.modeling_utils",
        "transformers.loss.loss_utils",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        candidate = getattr(mod, "LossKwargs", None)
        if candidate is not None:
            LossKwargs = candidate
            _record("transformers.utils.LossKwargs", f"reexported_from:{mod_name}")
            break

    if LossKwargs is None:
        from typing import Optional

        try:
            from typing import TypedDict
        except ImportError:  # pragma: no cover - py<3.8 never happens here
            from typing_extensions import TypedDict  # type: ignore

        class LossKwargs(TypedDict, total=False):  # type: ignore[no-redef]
            """Keyword arguments forwarded to the loss function.

            Mirrors the removed upstream definition: `num_items_in_batch` is the only
            documented field, and `total=False` keeps every key optional so that
            `Unpack[KwargsForCausalLM]` behaves exactly as before.
            """

            num_items_in_batch: Optional[int]

        _record("transformers.utils.LossKwargs", "synthesized_typeddict")

    tu.LossKwargs = LossKwargs
    try:
        import transformers.utils.generic as tug

        if not hasattr(tug, "LossKwargs"):
            tug.LossKwargs = LossKwargs
    except Exception:
        pass


def _install_optional_reexports() -> None:
    """Backfill symbols that moved between transformers releases.

    Every entry is best-effort: a missing source simply leaves the target untouched so
    that the real ImportError surfaces at the original call site instead of here.
    """
    moves: List[Tuple[str, str, str, str]] = [
        # (target_module, target_name, source_module, source_name)
        ("transformers.utils", "is_torchdynamo_compiling", "transformers.utils.import_utils", "is_torchdynamo_compiling"),
        ("transformers.cache_utils", "Cache", "transformers.cache_utils", "Cache"),
        ("transformers.modeling_flash_attention_utils", "_flash_attention_forward",
         "transformers.modeling_flash_attention_utils", "_flash_attention_forward"),
    ]
    for target_mod, target_name, source_mod, source_name in moves:
        try:
            tmod = importlib.import_module(target_mod)
        except Exception:
            continue
        if hasattr(tmod, target_name):
            continue
        try:
            smod = importlib.import_module(source_mod)
        except Exception:
            continue
        value = getattr(smod, source_name, None)
        if value is not None:
            setattr(tmod, target_name, value)
            _record(f"{target_mod}.{target_name}", f"reexported_from:{source_mod}")


def _install_cache_no_arg_init() -> None:
    """Let ``Cache.__init__()`` be called with no arguments again.

    transformers >=4.55 requires exactly one of ``layers`` / ``layer_class_to_replicate``
    and raises otherwise. The authoritative ``SlimInferCache(Cache)`` calls a bare
    ``super().__init__()`` because it manages every tensor itself: it overrides
    ``update()`` and ``get_seq_length()`` and stores KV in its own paged structures, so it
    needs the base only for inert bookkeeping.

    The wrapper defaults a no-arg call to ``layers=[]``, which is the pre-4.55 behaviour:
    empty base state and no dynamic layer auto-creation. Calls that already pass either
    argument are forwarded untouched, so ``DynamicCache`` and friends are unaffected.
    """
    import transformers.cache_utils as tcu

    Cache = getattr(tcu, "Cache", None)
    if Cache is None:
        return
    if getattr(Cache.__init__, "_llama_drm_no_arg_shim", False):
        _record("transformers.cache_utils.Cache.__init__", "already_patched")
        return

    original_init = Cache.__init__

    try:
        import inspect

        parameters = inspect.signature(original_init).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "layer_class_to_replicate" not in parameters:
        _record("transformers.cache_utils.Cache.__init__", "no_patch_needed")
        return

    def patched_init(self, layers=None, layer_class_to_replicate=None, *args, **kwargs):
        if layers is None and layer_class_to_replicate is None:
            layers = []
        return original_init(self, layers, layer_class_to_replicate, *args, **kwargs)

    patched_init._llama_drm_no_arg_shim = True  # type: ignore[attr-defined]
    patched_init.__doc__ = original_init.__doc__
    Cache.__init__ = patched_init  # type: ignore[method-assign]
    _record("transformers.cache_utils.Cache.__init__", "no_arg_defaults_to_empty_layers")


def apply_compat() -> Dict[str, str]:
    """Idempotently patch transformers so the DRM modeling files import cleanly."""
    _install_loss_kwargs()
    _install_optional_reexports()
    _install_cache_no_arg_init()
    return dict(_APPLIED)


def compat_report() -> Dict[str, Any]:
    import transformers
    import torch

    try:
        import flash_attn  # noqa: F401

        flash_available = True
    except Exception:
        flash_available = False

    return {
        "protocol_id": PROTOCOL_ID,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "flash_attn_available": flash_available,
        "recommended_backend": "flash_attention_2" if flash_available else "sdpa",
        "patches": apply_compat(),
    }


apply_compat()
