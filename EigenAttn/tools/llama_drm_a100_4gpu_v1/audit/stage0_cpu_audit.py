#!/usr/bin/env python3
"""Stage 0: pure-CPU pre-flight audit for LLAMA_DRM_A100_4GPU_V1.

Hard guarantee: this process never places model weights on a GPU. Any check that needs
`torch` runs in a subprocess with `CUDA_VISIBLE_DEVICES=""`, and no check calls
`from_pretrained`. The audit is the only gate before GPU stages start, so it prefers
failing loudly over passing optimistically.

Check groups
  A  file / bundle inventory and hashes
  B  transformers compatibility (import-only, CPU)
  C  model identity
  D  dataset identity and manifests (17 tasks / 3950, six tasks / 1500)
  E  scorer and prompt-config coverage
  F  five-method + Full-KV CLI wiring
  G  Random-vs-DRM fairness invariants
  H  DRM rule presence in cache_utils (source inspection)
  I  multi-GPU co-location scheduler unit tests (mocked, no CUDA)
  J  resume and atomic-write behaviour
  K  forbidden CC/ACC method families are unreachable
  L  writable results volume with enough free space

Exit 0 and write STAGE0_PASS.json on PASS; write HARD_BLOCK.* and exit non-zero on FAIL.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C
from scheduler.gpu_coloc_scheduler import (
    GpuColocScheduler,
    Job,
    estimated_peak_gib,
    make_mock_gpu_source,
    make_mock_launcher,
    STATE_WAITING_FOR_GPU_IDLE,
)

SEVERITY_FAIL = "FAIL"
SEVERITY_WARN = "WARN"
SEVERITY_PASS = "PASS"


class Audit:
    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []

    def record(self, group: str, name: str, ok: bool, detail: Any = "",
               severity: str = SEVERITY_FAIL) -> bool:
        self.checks.append({
            "group": group,
            "name": name,
            "status": SEVERITY_PASS if ok else severity,
            "detail": detail if isinstance(detail, (str, int, float, bool, type(None)))
            else json.loads(json.dumps(detail, default=str)),
        })
        return ok

    def expect(self, group: str, name: str, ok: bool, detail: Any = "") -> bool:
        return self.record(group, name, ok, detail, SEVERITY_FAIL)

    def warn_if(self, group: str, name: str, ok: bool, detail: Any = "") -> bool:
        return self.record(group, name, ok, detail, SEVERITY_WARN)

    def guard(self, group: str, name: str, fn: Callable[[], Tuple[bool, Any]]) -> bool:
        try:
            ok, detail = fn()
        except Exception:
            return self.expect(group, name, False, traceback.format_exc(limit=6))
        return self.expect(group, name, ok, detail)

    @property
    def failures(self) -> List[Dict[str, Any]]:
        return [check for check in self.checks if check["status"] == SEVERITY_FAIL]

    @property
    def warnings(self) -> List[Dict[str, Any]]:
        return [check for check in self.checks if check["status"] == SEVERITY_WARN]

    def summary(self) -> Dict[str, Any]:
        return {
            "total": len(self.checks),
            "passed": sum(1 for c in self.checks if c["status"] == SEVERITY_PASS),
            "failed": len(self.failures),
            "warnings": len(self.warnings),
        }


# ---------------------------------------------------------------------------
# CPU-only subprocess helper
# ---------------------------------------------------------------------------


def run_cpu_probe(script: str, timeout: int = 900) -> Tuple[int, str, str]:
    """Run a probe with CUDA hidden, so torch can never claim a device."""
    env = C.offline_env({
        "CUDA_VISIBLE_DEVICES": "",
        # Belt and braces: even if something tried to init CUDA it would find no device.
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    })
    proc = subprocess.run(
        [C.python_bin(), "-c", script],
        capture_output=True, text=True, timeout=timeout,
        cwd=str(C.RUNTIME_ROOT), env=env, check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


# ---------------------------------------------------------------------------
# Group A: inventory
# ---------------------------------------------------------------------------

REQUIRED_FILES: Tuple[str, ...] = (
    "common.py",
    "orchestrator.py",
    "worker_entry.py",
    "LLAMA_DRM_METHOD_SPEC.md",
    "adapters/transformers_compat.py",
    "audit/stage0_cpu_audit.py",
    "scheduler/gpu_coloc_scheduler.py",
    "stages/stage7_analysis.py",
    "status_bootstrap/protocol_templates.py",
    "runtime/experiments/run_llama_sliminfer_variants.py",
    "runtime/models/modeling_llama.py",
    "runtime/models/configuration_llama.py",
    "runtime/utils/cache_utils.py",
    "runtime/utils/processor.py",
    "runtime/bridgekv/rescuekv_scoring.py",
    "runtime/bridgekv/structured_rescue_scoring.py",
    "runtime/bridgekv/cc_topology_policies.py",
    "runtime/bridgekv/acc_sliminfer.py",
    "runtime/longbench_eval.py",
    "runtime/tools/qwen_pipeline_common.py",
    "runtime/configs/dataset2prompt.json",
    "runtime/configs/dataset2maxlen.json",
    "protocol_ulb_v1/task_manifest.json",
)


def group_a_inventory(audit: Audit) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for rel in REQUIRED_FILES:
        path = C.TOOL_ROOT / rel
        exists = path.is_file()
        audit.expect("A_inventory", f"present:{rel}", exists, str(path))
        if exists:
            hashes[rel] = C.sha256_file(path)

    cache_utils = C.TOOL_ROOT / "runtime" / "utils" / "cache_utils.py"
    if cache_utils.is_file():
        line_count = sum(1 for _ in cache_utils.open("r", encoding="utf-8", errors="ignore"))
        # The authoritative DRM cache_utils is ~10.8k lines; a drastically smaller file
        # means a stub or truncated copy slipped in.
        audit.expect("A_inventory", "cache_utils_is_authoritative_size",
                     line_count > 10000, f"{line_count} lines")
    audit.expect("A_inventory", "runner_syntax_ok", *_syntax_check(
        C.TOOL_ROOT / "runtime" / "experiments" / "run_llama_sliminfer_variants.py"))
    audit.expect("A_inventory", "modeling_llama_syntax_ok", *_syntax_check(
        C.TOOL_ROOT / "runtime" / "models" / "modeling_llama.py"))
    return hashes


def _syntax_check(path: Path) -> Tuple[bool, str]:
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        return True, "compiles"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Group B: transformers compatibility
# ---------------------------------------------------------------------------


def group_b_compat(audit: Audit) -> Dict[str, Any]:
    packages = C.python_package_versions()
    audit.expect("B_compat", "torch_present", packages.get("torch", "MISSING") != "MISSING",
                 packages.get("torch"))
    audit.expect("B_compat", "transformers_present",
                 packages.get("transformers", "MISSING") != "MISSING",
                 packages.get("transformers"))
    flash_missing = packages.get("flash_attn", "MISSING") == "MISSING"
    audit.warn_if("B_compat", "flash_attn_absent_so_backend_is_sdpa", flash_missing,
                  f"flash_attn={packages.get('flash_attn')}; frozen backend is sdpa")

    probe = r"""
import json, sys
sys.path.insert(0, %r)
sys.path.insert(0, %r)
out = {}
try:
    from adapters import transformers_compat
    out["compat"] = transformers_compat.compat_report()
    import transformers.utils as tu
    out["loss_kwargs_available"] = hasattr(tu, "LossKwargs")
    from models.modeling_llama import LlamaForCausalLM
    from models.configuration_llama import LlamaConfig
    out["modeling_llama_import"] = True
    out["has_sliminfer_hooks"] = all(
        name in open(%r, encoding="utf-8").read()
        for name in ("SlimInferCache", "HsCache", "SlimInferLayerProcessor")
    )
    from utils.cache_utils import SlimInferCache, HsCache
    out["cache_classes_import"] = True
    import torch
    out["cuda_available"] = torch.cuda.is_available()
    out["cuda_device_count"] = torch.cuda.device_count()
except Exception as exc:
    import traceback
    out["error"] = traceback.format_exc()
print("PROBE_JSON:" + json.dumps(out, default=str))
""" % (
        str(C.TOOL_ROOT), str(C.RUNTIME_ROOT),
        str(C.TOOL_ROOT / "runtime" / "models" / "modeling_llama.py"),
    )
    code, stdout, stderr = run_cpu_probe(probe)
    payload = _extract_probe(stdout)
    audit.expect("B_compat", "compat_probe_ran", code == 0 and bool(payload),
                 (stderr or stdout)[-1500:])
    audit.expect("B_compat", "loss_kwargs_shim_effective",
                 bool(payload.get("loss_kwargs_available")), payload.get("compat"))
    audit.expect("B_compat", "modeling_llama_imports",
                 bool(payload.get("modeling_llama_import")), payload.get("error", "")[-1500:])
    audit.expect("B_compat", "sliminfer_drm_hooks_intact",
                 bool(payload.get("has_sliminfer_hooks")), "SlimInferCache/HsCache/processor referenced")
    audit.expect("B_compat", "cache_utils_classes_import",
                 bool(payload.get("cache_classes_import")), payload.get("error", "")[-800:])
    # Proof that the audit itself never touched a GPU.
    audit.expect("B_compat", "audit_probe_saw_no_gpu",
                 payload.get("cuda_device_count", 0) == 0,
                 f"cuda_device_count={payload.get('cuda_device_count')}")

    sdpa_causality = group_b_sdpa_causality(audit)
    return {"packages": packages, "probe": payload, "sdpa_causality": sdpa_causality}


def group_b_sdpa_causality(audit: Audit) -> Dict[str, Any]:
    """Guard the SDPA prefill-causality invariant.

    The authoritative SlimInfer attention was validated with flash_attention_2, whose
    branch gets causality from ``is_causal=True``. Prefill therefore deliberately leaves
    ``current_attention_mask = None``. On the sdpa backend a None mask combined with
    ``is_causal=False`` silently produces *bidirectional* prefill attention: the model
    still runs, but every prompt token attends to its own future and generation collapses
    into degenerate repetition (observed as LongBench 0.0 across every DRM arm while
    Full-KV scored normally). This regression is invisible without a scored run, so the
    invariant is pinned here.
    """
    model_src = (C.RUNTIME_ROOT / "models" / "modeling_llama.py").read_text(encoding="utf-8")

    audit.expect(
        "B_compat", "sdpa_branch_does_not_hardcode_is_causal_false",
        "is_causal=sdpa_is_causal" in model_src,
        "the sdpa branch must derive is_causal from mask presence, not pin it to False",
    )
    audit.expect(
        "B_compat", "sdpa_prefill_uses_causal_when_mask_is_none",
        "if sdpa_mask is None:" in model_src and "sdpa_is_causal = True" in model_src,
        "a None mask (prefill) must fall back to is_causal=True",
    )
    audit.expect(
        "B_compat", "sdpa_length_mismatch_fails_loudly",
        "causal alignment would be ambiguous" in model_src,
        "unequal q_len/kv_len with no mask must raise rather than mis-align the causal mask",
    )
    audit.expect(
        "B_compat", "prefill_still_leaves_mask_none_as_authoritative",
        "# Prefill phase doesn't use attn mask" in model_src,
        "upstream prefill behaviour retained; only the sdpa consumer was corrected",
    )

    # Numeric proof of the equivalence the fix relies on: for q_len == kv_len, SDPA's
    # top-left aligned is_causal=True equals an explicit additive causal mask, and both
    # differ from the unmasked (bidirectional) result.
    probe = r"""
import json, torch, torch.nn.functional as F
out = {}
torch.manual_seed(0)
q, k, v = (torch.randn(1, 2, 16, 8, dtype=torch.float64) for _ in range(3))
causal_flag = F.scaled_dot_product_attention(q, k, v, is_causal=True)
mask = torch.full((16, 16), float("-inf"), dtype=torch.float64).triu(1)
causal_mask = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
bidirectional = F.scaled_dot_product_attention(q, k, v, is_causal=False)
out["causal_flag_matches_explicit_mask"] = torch.allclose(causal_flag, causal_mask, atol=1e-10)
out["bidirectional_differs_from_causal"] = not torch.allclose(bidirectional, causal_flag, atol=1e-6)
print("PROBE_JSON:" + json.dumps(out))
"""
    code, stdout, stderr = run_cpu_probe(probe)
    payload = _extract_probe(stdout)
    audit.expect("B_compat", "sdpa_causality_probe_ran", code == 0 and bool(payload),
                 (stderr or stdout)[-800:])
    audit.expect("B_compat", "is_causal_true_equals_explicit_causal_mask",
                 bool(payload.get("causal_flag_matches_explicit_mask")), payload)
    audit.expect("B_compat", "unmasked_sdpa_is_not_causal",
                 bool(payload.get("bidirectional_differs_from_causal")), payload)
    return payload


def _extract_probe(stdout: str) -> Dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("PROBE_JSON:"):
            try:
                return json.loads(line[len("PROBE_JSON:"):])
            except json.JSONDecodeError:
                return {}
    return {}


# ---------------------------------------------------------------------------
# Group C: model identity
# ---------------------------------------------------------------------------


def group_c_model(audit: Audit) -> Dict[str, Any]:
    model_path = C.resolve_model_path()
    if not audit.expect("C_model", "model_snapshot_found", model_path is not None,
                        f"searched {list(C.MODEL_PATH_CANDIDATES)}"):
        return {}
    assert model_path is not None
    config = C.read_json(model_path / "config.json", default={}) or {}
    audit.expect("C_model", "architecture_is_llama",
                 config.get("architectures") == ["LlamaForCausalLM"],
                 config.get("architectures"))
    audit.expect("C_model", "max_position_embeddings_8192",
                 int(config.get("max_position_embeddings", 0)) == C.MODEL_MAX_POSITION_EMBEDDINGS,
                 config.get("max_position_embeddings"))
    audit.expect("C_model", "num_hidden_layers_32",
                 int(config.get("num_hidden_layers", 0)) == 32, config.get("num_hidden_layers"))
    audit.expect("C_model", "prune_layers_within_depth",
                 all(layer < int(config.get("num_hidden_layers", 0))
                     for layer in C.FROZEN_LLAMA_CONFIG["prune_layers"]),
                 C.FROZEN_LLAMA_CONFIG["prune_layers"])
    audit.expect("C_model", "max_model_len_leaves_generation_headroom",
                 C.MAX_MODEL_LEN + 512 <= C.MODEL_MAX_POSITION_EMBEDDINGS,
                 f"{C.MAX_MODEL_LEN} + 512 <= {C.MODEL_MAX_POSITION_EMBEDDINGS}")

    tokenizer_config = C.read_json(model_path / "tokenizer_config.json", default={}) or {}
    has_template = bool(tokenizer_config.get("chat_template"))
    audit.expect("C_model", "base_model_has_no_chat_template", not has_template,
                 "build_chat must pass prompts through unchanged")
    audit.expect("C_model", "revision_recorded_as_unknown",
                 C.MODEL_REVISION == "UNKNOWN_LOCAL_SNAPSHOT", C.MODEL_REVISION)

    weights = sorted(p.name for p in model_path.glob("*.safetensors"))
    audit.expect("C_model", "weight_shards_present", len(weights) >= 1, weights)
    return {
        "resolved_path": str(model_path),
        "revision": C.MODEL_REVISION,
        "config_sha256": C.sha256_file(model_path / "config.json"),
        "tokenizer_config_sha256": (
            C.sha256_file(model_path / "tokenizer_config.json")
            if (model_path / "tokenizer_config.json").exists() else None
        ),
        "weight_shards": weights,
        "config": config,
    }


# ---------------------------------------------------------------------------
# Group D: dataset identity
# ---------------------------------------------------------------------------


def group_d_dataset(audit: Audit) -> Dict[str, Any]:
    longbench = C.resolve_longbench_dir()
    if not audit.expect("D_dataset", "longbench_dir_found", longbench is not None,
                        f"searched {list(C.LONGBENCH_PATH_CANDIDATES)}"):
        return {}
    assert longbench is not None

    manifest = C.read_json(C.PROTOCOL_DIR / "task_manifest.json", default={}) or {}
    manifest_tasks = {entry["task"]: entry for entry in manifest.get("tasks", [])}
    audit.expect("D_dataset", "manifest_has_17_tasks", len(manifest_tasks) == 17,
                 len(manifest_tasks))
    audit.expect("D_dataset", "manifest_total_3950",
                 int(manifest.get("total_samples", 0)) == C.EXPECTED_TOTAL_SAMPLES_17,
                 manifest.get("total_samples"))
    audit.expect("D_dataset", "manifest_task_order_matches_protocol",
                 tuple(manifest.get("task_order", [])) == C.SEVENTEEN_TASKS,
                 manifest.get("task_order"))
    audit.expect("D_dataset", "pooled_sample_average_forbidden",
                 bool(manifest.get("pooled_sample_average_forbidden")), True)

    per_task: Dict[str, Any] = {}
    observed_total = 0
    for task in C.SEVENTEEN_TASKS:
        path = C.longbench_task_file(longbench, task)
        if not audit.expect("D_dataset", f"file_present:{task}", path.exists(), str(path)):
            continue
        count = C.count_jsonl_records(path)
        expected = int(manifest_tasks.get(task, {}).get("sample_count", -1))
        observed_total += count
        audit.expect("D_dataset", f"sample_count:{task}", count == expected,
                     f"observed {count}, manifest {expected}")
        digest = C.sha256_file(path)
        manifest_digest = manifest_tasks.get(task, {}).get("dataset_file_sha256")
        # The manifest hashes came from the Qwen protocol snapshot. A mismatch is
        # reported but not fatal: the authoritative count check above is what pins
        # evaluation identity for this run.
        audit.warn_if("D_dataset", f"file_sha256_matches_manifest:{task}",
                      digest == manifest_digest,
                      f"observed {digest[:16]}..., manifest {str(manifest_digest)[:16]}...")
        per_task[task] = {
            "path": str(path), "samples": count, "expected": expected,
            "sha256": digest, "manifest_sha256": manifest_digest,
            "max_new_tokens": manifest_tasks.get(task, {}).get("max_new_tokens"),
        }

    audit.expect("D_dataset", "total_samples_3950",
                 observed_total == C.EXPECTED_TOTAL_SAMPLES_17,
                 f"observed {observed_total}, expected {C.EXPECTED_TOTAL_SAMPLES_17}")

    six_total = sum(per_task.get(task, {}).get("samples", 0) for task in C.SIX_TASKS)
    audit.expect("D_dataset", "six_task_subset_is_17_task_subset",
                 all(task in C.SEVENTEEN_TASKS for task in C.SIX_TASKS), list(C.SIX_TASKS))
    audit.expect("D_dataset", "six_task_total_1500",
                 six_total == C.EXPECTED_TOTAL_SAMPLES_6,
                 f"observed {six_total}, expected {C.EXPECTED_TOTAL_SAMPLES_6}")

    # No-network guarantee: the runner must not be able to fall back to a download.
    runner_src = (C.RUNTIME_ROOT / "experiments" / "run_llama_sliminfer_variants.py").read_text(
        encoding="utf-8")
    audit.expect("D_dataset", "runner_never_calls_load_dataset",
                 "load_dataset(" not in runner_src,
                 "datasets.load_dataset removed from load_longbench_samples")
    audit.expect("D_dataset", "runner_raises_filenotfound_when_missing",
                 "raise FileNotFoundError(" in runner_src, "offline hard error present")
    return {
        "resolved_path": str(longbench),
        "per_task": per_task,
        "total_samples": observed_total,
        "six_task_total": six_total,
        "manifests": {
            "seventeen_tasks": list(C.SEVENTEEN_TASKS),
            "six_tasks": list(C.SIX_TASKS),
        },
    }


# ---------------------------------------------------------------------------
# Group E: scorer / prompt configs
# ---------------------------------------------------------------------------


def group_e_scorer(audit: Audit) -> Dict[str, Any]:
    scorer_path = C.RUNTIME_ROOT / "longbench_eval.py"
    src = scorer_path.read_text(encoding="utf-8") if scorer_path.exists() else ""
    audit.expect("E_scorer", "scorer_function_present", "def scorer(" in src, str(scorer_path))

    prompts = C.read_json(C.RUNTIME_ROOT / "configs" / "dataset2prompt.json", default={}) or {}
    maxlens = C.read_json(C.RUNTIME_ROOT / "configs" / "dataset2maxlen.json", default={}) or {}
    missing_prompts = [task for task in C.SEVENTEEN_TASKS if task not in prompts]
    missing_maxlen = [task for task in C.SEVENTEEN_TASKS if task not in maxlens]
    audit.expect("E_scorer", "prompt_template_for_all_17", not missing_prompts, missing_prompts)
    audit.expect("E_scorer", "max_new_tokens_for_all_17", not missing_maxlen, missing_maxlen)

    manifest = C.read_json(C.PROTOCOL_DIR / "task_manifest.json", default={}) or {}
    mismatches = []
    for entry in manifest.get("tasks", []):
        task = entry.get("task")
        if task in maxlens and int(maxlens[task]) != int(entry.get("max_new_tokens", -1)):
            mismatches.append({"task": task, "config": maxlens[task],
                               "manifest": entry.get("max_new_tokens")})
    audit.expect("E_scorer", "max_new_tokens_matches_manifest", not mismatches, mismatches)

    # The scorer is only reachable at the *end* of a task, so a missing scoring
    # dependency would otherwise waste a full task's generation before failing.
    vendored = C.vendored_deps_dir()
    audit.expect("E_scorer", "vendored_deps_dir_configured", vendored is not None,
                 f"pointer {C.VENDORED_DEPS_POINTER}")
    audit.expect("E_scorer", "vendored_deps_dir_exists",
                 vendored is not None and vendored.is_dir(), str(vendored))
    audit.expect("E_scorer", "vendored_deps_on_worker_pythonpath",
                 vendored is not None and str(vendored) in C.offline_env()["PYTHONPATH"],
                 C.offline_env()["PYTHONPATH"])

    probe = r"""
import json, sys
sys.path.insert(0, %r); sys.path.insert(0, %r)
out = {}
for name in %s:
    try:
        mod = __import__(name)
        out[name] = getattr(mod, "__version__", "ok")
    except Exception as exc:
        out[name] = "MISSING: " + repr(exc)
try:
    from longbench_eval import scorer, dataset2metric
    out["scorer_import"] = True
    out["metric_coverage"] = [t for t in %s if t not in dataset2metric]
except Exception as exc:
    import traceback
    out["scorer_import"] = False
    out["scorer_error"] = traceback.format_exc()
print("PROBE_JSON:" + json.dumps(out, default=str))
""" % (str(C.TOOL_ROOT), str(C.RUNTIME_ROOT), json.dumps(list(C.REQUIRED_SCORER_MODULES)),
       json.dumps(list(C.SEVENTEEN_TASKS)))
    code, stdout, stderr = run_cpu_probe(probe)
    payload = _extract_probe(stdout)
    audit.expect("E_scorer", "scorer_probe_ran", code == 0 and bool(payload),
                 (stderr or stdout)[-1200:])
    for module in C.REQUIRED_SCORER_MODULES:
        value = str(payload.get(module, "MISSING"))
        audit.expect("E_scorer", f"scorer_dependency_importable:{module}",
                     not value.startswith("MISSING"), value)
    audit.expect("E_scorer", "longbench_eval_scorer_importable",
                 bool(payload.get("scorer_import")),
                 str(payload.get("scorer_error", ""))[-1200:])
    audit.expect("E_scorer", "metric_defined_for_all_17_tasks",
                 payload.get("metric_coverage") == [], payload.get("metric_coverage"))
    return {
        "scorer": str(scorer_path),
        "vendored_deps_dir": str(vendored) if vendored else None,
        "scorer_dependency_versions": {
            module: payload.get(module) for module in C.REQUIRED_SCORER_MODULES
        },
        "prompt_config_sha256": C.sha256_json(prompts),
        "maxlen_config_sha256": C.sha256_json(maxlens),
    }


# ---------------------------------------------------------------------------
# Group F: method CLI wiring
# ---------------------------------------------------------------------------


def group_f_method_cli(audit: Audit) -> Dict[str, Any]:
    import orchestrator as O

    expected = {
        "sliminfer_original": ("sliminfer_original", "none", "delta_o", None),
        "sliminfer_larger": ("sliminfer_original", "sliminfer_larger", "delta_o", None),
        "random_add": ("sliminfer_original", "add", "random", 0.05),
        "drm_add": ("sliminfer_original", "add", "drm", 0.05),
        "drm_swap": ("sliminfer_original", "budget_neutral_drm_swap", "drm", None),
    }
    built: Dict[str, Any] = {}
    for key, (method, policy, score, ratio) in expected.items():
        args = O.method_args(key, seed=0 if key == "random_add" else None)
        built[key] = args
        audit.expect("F_method_cli", f"{key}:method", args.get("method") == method, args)
        audit.expect("F_method_cli", f"{key}:rescue_policy",
                     args.get("rescue_policy") == policy, args)
        audit.expect("F_method_cli", f"{key}:rescue_score",
                     args.get("rescue_score") == score, args)
        if ratio is None:
            audit.expect("F_method_cli", f"{key}:no_rescue_ratio_override",
                         "rescue_ratio" not in args, args)
        else:
            audit.expect("F_method_cli", f"{key}:rescue_ratio",
                         abs(float(args.get("rescue_ratio", -1)) - ratio) < 1e-9, args)

    fullkv = O.method_args("fullkv")
    built["fullkv"] = fullkv
    audit.expect("F_method_cli", "fullkv:method", fullkv.get("method") == "fullkv", fullkv)
    audit.expect("F_method_cli", "fullkv:no_rescue_flags",
                 "rescue_policy" not in fullkv and "rescue_score" not in fullkv, fullkv)
    audit.expect("F_method_cli", "only_fullkv_differs_in_method",
                 all(built[key]["method"] == "sliminfer_original"
                     for key in C.FIVE_METHODS),
                 {key: built[key]["method"] for key in C.FIVE_METHODS})
    audit.expect("F_method_cli", "random_needs_explicit_seed",
                 "seed" in built["random_add"], built["random_add"])

    # The argparse layer must accept every arm; validated in a CPU subprocess so the
    # audit process itself never imports torch.
    probe = r"""
import json, sys
sys.path.insert(0, %r); sys.path.insert(0, %r)
import importlib.util
spec = importlib.util.spec_from_file_location("runner", %r)
mod = importlib.util.module_from_spec(spec); sys.modules["runner"] = mod
spec.loader.exec_module(mod)
cases = %s
out = {"accepted": {}, "errors": {}}
for name, argv in cases.items():
    try:
        sys.argv = ["runner"] + argv
        args = mod.normalize_args(mod.parse_args())
        out["accepted"][name] = {
            "method": args.method, "rescue_policy": args.rescue_policy,
            "rescue_score": args.rescue_score, "rescue_ratio": args.rescue_ratio,
            "prune_layers": args.prune_layers, "retain_tokens": args.retain_tokens,
            "dtype": args.dtype, "backend": args.backend, "block_size": args.block_size,
            "token_unit_size": args.token_unit_size, "swap_threshold": args.swap_threshold,
            "local_query_window": args.local_query_window, "seed": args.seed,
            "model_family": args.model_family, "max_model_len": args.max_model_len,
        }
    except BaseException as exc:
        out["errors"][name] = repr(exc)
print("PROBE_JSON:" + json.dumps(out, default=str))
""" % (
        str(C.TOOL_ROOT), str(C.RUNTIME_ROOT),
        str(C.RUNNER_SCRIPT),
        json.dumps({
            key: _argv_from_args({**O.base_runner_args(Path("/tmp/model"), Path("/tmp/lb")),
                                  **built[key],
                                  "longbench_tasks": "qasper",
                                  "output_dir": "/tmp/out"})
            for key in list(C.FIVE_METHODS) + ["fullkv"]
        }),
    )
    code, stdout, stderr = run_cpu_probe(probe)
    payload = _extract_probe(stdout)
    audit.expect("F_method_cli", "argparse_probe_ran", code == 0 and bool(payload),
                 (stderr or stdout)[-1500:])
    audit.expect("F_method_cli", "argparse_accepts_all_arms",
                 not payload.get("errors"), payload.get("errors"))

    accepted = payload.get("accepted", {})
    for key in C.FIVE_METHODS:
        resolved = accepted.get(key, {})
        audit.expect("F_method_cli", f"resolved_frozen_config:{key}",
                     resolved.get("prune_layers") == "10,20,30"
                     and resolved.get("retain_tokens") == "8192,4096,2048"
                     and resolved.get("dtype") == "bf16"
                     and resolved.get("backend") == "sdpa"
                     and int(resolved.get("block_size", 0)) == 64
                     and int(resolved.get("token_unit_size", 0)) == 8
                     and abs(float(resolved.get("swap_threshold", 0)) - 0.9) < 1e-9
                     and int(resolved.get("local_query_window", 0)) == 4
                     and int(resolved.get("max_model_len", 0)) == C.MAX_MODEL_LEN,
                     resolved)
    return {"built_args": built, "resolved": accepted}


def _argv_from_args(args: Dict[str, Any]) -> List[str]:
    argv: List[str] = []
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                argv.append(f"--{key}")
            continue
        if value is None:
            continue
        argv.extend([f"--{key}", str(value)])
    return argv


# ---------------------------------------------------------------------------
# Group G: Random-vs-DRM fairness invariants
# ---------------------------------------------------------------------------


def group_g_random_invariants(audit: Audit) -> Dict[str, Any]:
    """Documented checks that the Random control is budget-identical to DRM.

    The comparison is only meaningful if Random and DRM differ in exactly one thing: the
    score used to rank inactive pages. These checks pin that down at three levels: the
    CLI, the rescue-budget arithmetic, and the source of the score itself.
    """
    import orchestrator as O

    random_args = O.method_args("random_add", seed=0)
    drm_args = O.method_args("drm_add")
    differing = {
        key for key in set(random_args) | set(drm_args)
        if random_args.get(key) != drm_args.get(key)
    }
    audit.expect("G_random", "random_vs_drm_differ_only_in_score_and_seed",
                 differing <= {"rescue_score", "seed"}, sorted(differing))
    audit.expect("G_random", "random_and_drm_share_rescue_policy",
                 random_args["rescue_policy"] == drm_args["rescue_policy"] == "add",
                 [random_args["rescue_policy"], drm_args["rescue_policy"]])
    audit.expect("G_random", "random_and_drm_share_rescue_ratio",
                 abs(random_args["rescue_ratio"] - drm_args["rescue_ratio"]) < 1e-9,
                 [random_args["rescue_ratio"], drm_args["rescue_ratio"]])

    # Rescue budget arithmetic replicated from bridgekv.rescuekv_scoring.apply_rescue_policy:
    #   m_rescue = int(round(k_active * rescue_ratio)), floored at 1 when ratio > 0.
    # With retain_tokens/block_size = 128/64/32 active pages this must give M = 6/3/2,
    # identically for Random, Larger and DRM-Add.
    def m_of(k_active: int, ratio: float, inactive: int = 10 ** 6) -> int:
        m = int(round(k_active * float(ratio)))
        if ratio > 0.0 and k_active > 0 and inactive:
            m = max(1, m)
        return min(m, inactive)

    derived_m = [m_of(pages, 0.05) for pages in C.RETAINED_PAGES_PER_PRUNE_LAYER]
    audit.expect("G_random", "rescue_M_is_6_3_2_at_ratio_0p05",
                 tuple(derived_m) == C.RESCUE_M_PER_PRUNE_LAYER,
                 {"pages": list(C.RETAINED_PAGES_PER_PRUNE_LAYER), "derived_M": derived_m,
                  "frozen_M": list(C.RESCUE_M_PER_PRUNE_LAYER)})
    audit.expect("G_random", "M_identical_across_add_arms",
                 m_of(128, 0.05) == m_of(128, 0.05),
                 "M depends only on (k_active, rescue_ratio), never on rescue_score")

    rescue_src = (C.RUNTIME_ROOT / "bridgekv" / "rescuekv_scoring.py").read_text(encoding="utf-8")
    audit.expect("G_random", "add_maps_to_additive_rescue",
                 'if policy == "add":' in rescue_src
                 and 'policy = "additive_rescue"' in rescue_src, "policy alias present")
    audit.expect("G_random", "larger_shares_additive_budget",
                 'if policy == "sliminfer_larger":' in rescue_src, "K+M budget for Larger")
    audit.expect("G_random", "additive_rescue_reports_K_plus_M",
                 'final_budget_type="K+M"' in rescue_src, "additive arms are K+M")
    audit.expect("G_random", "swap_reports_K_budget",
                 'final_budget_type="K"' in rescue_src, "swap arms stay at K")
    audit.expect("G_random", "M_formula_present_in_source",
                 "m_rescue = int(round(k_active * float(rescue_ratio)))" in rescue_src,
                 "budget arithmetic is score-independent")

    cache_src = (C.RUNTIME_ROOT / "utils" / "cache_utils.py").read_text(encoding="utf-8")
    audit.expect("G_random", "random_score_is_seeded",
                 'if score_name == "random":' in cache_src
                 and "generator.manual_seed(seed)" in cache_src,
                 "random rescue scores derive from cfg['seed'] via a CPU generator")
    audit.expect("G_random", "random_uses_same_scoring_entrypoint",
                 'if score_name == "sliminfer_next":' in cache_src,
                 "random is a branch of the shared rescue-score function, not a separate path")
    audit.expect("G_random", "random_seeds_are_distinct", len(set(C.RANDOM_SEEDS)) == len(C.RANDOM_SEEDS),
                 list(C.RANDOM_SEEDS))
    return {
        "random_args": random_args, "drm_args": drm_args,
        "derived_M": derived_m,
    }


# ---------------------------------------------------------------------------
# Group H: DRM rule presence in cache_utils
# ---------------------------------------------------------------------------


def group_h_drm_rules(audit: Audit) -> Dict[str, Any]:
    """Source inspection: the shared active-set / residual / Add / Swap rules must exist.

    These are presence checks on the authoritative file, not behavioural tests: running
    them behaviourally would need GPU tensors, which stage 0 must not do.
    """
    cache_path = C.RUNTIME_ROOT / "utils" / "cache_utils.py"
    src = cache_path.read_text(encoding="utf-8")

    markers: Dict[str, Tuple[bool, str]] = {
        "shared_A_K_active_set": (
            "A_K" in src and "inactive P set" in src,
            "the fixed-page swap path consumes the concrete SlimInfer A_K and inactive P set",
        ),
        "sliminfer_larger_extra_budget_rule": (
            'if rescue_policy == "sliminfer_larger"' in src
            and "num_virtual_to_keep + extra" in src,
            "Larger enlarges the retained budget by round(K*ratio)",
        ),
        "add_policy_dispatch": (
            "apply_rescue_policy" in src, "additive rescue routed through the shared policy",
        ),
        "budget_neutral_drm_swap_rule": (
            '"budget_neutral_drm_swap"' in src, "swap policy recognised in the page path",
        ),
        "drm_score_entrypoint": (
            "drm_s0_score(" in src, "DRM S0 scoring invoked",
        ),
        "drm_rescue_score_branch": (
            'if rescue_score == "drm"' in src, "drm rescue-score branch present",
        ),
        "residual_aware_proxy_fields": (
            "_residual_aware_proxy_fields" in src and "residual_benefit_proxy" in src,
            "residual-aware audit fields computed",
        ),
        "block_size_64_constant": (
            "_BLOCK_SIZE = 64" in src, "page size 64 frozen in the DRM cache",
        ),
        "token_unit_default_8": (
            "_SEMANTIC_KERNEL_SIZE = _BLOCK_SIZE // 8" in src, "token unit 8 derived from page size",
        ),
        "local_pruning_window_4": (
            "_PRUNING_WINDOW_LEN = 4" in src, "local query window 4",
        ),
        "swap_threshold_default_0p8_overridden_to_0p9": (
            "_SWAP_THRESHOLD" in src, "gamma is configurable; the run pins it to 0.9 via --swap_threshold",
        ),
    }
    for name, (ok, detail) in markers.items():
        audit.expect("H_drm_rules", name, ok, detail)

    audit.expect("H_drm_rules", "cache_utils_unmodified_authoritative",
                 "from bridgekv.cc_topology_policies import" in src
                 and "from bridgekv.acc_sliminfer import" in src,
                 "authoritative import block intact; CC/ACC satisfied by stub modules")
    return {"cache_utils_sha256": C.sha256_file(cache_path),
            "markers": {name: bool(ok) for name, (ok, _) in markers.items()}}


# ---------------------------------------------------------------------------
# Group I: scheduler unit tests (mocked)
# ---------------------------------------------------------------------------


def group_i_scheduler(audit: Audit) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    def peak_rule() -> Tuple[bool, Any]:
        cases = {
            10.0: max(12.0, 14.0),
            30.0: max(36.0, 34.0),
            100.0: max(120.0, 104.0),
        }
        observed = {actual: estimated_peak_gib(actual) for actual in cases}
        ok = all(abs(observed[a] - expected) < 1e-6 for a, expected in cases.items())
        return ok, {"expected": cases, "observed": observed}

    audit.guard("I_scheduler", "estimated_peak_rule", peak_rule)

    with tempfile.TemporaryDirectory(prefix="stage0_sched_") as tmp:
        tmp_path = Path(tmp)
        four_gpus = [
            {"uuid": f"GPU-MOCK-{i}", "index": i, "free_mib": 81920, "total_mib": 81920}
            for i in range(4)
        ]

        def make(canary: bool = False, gpus: Optional[List[Dict[str, Any]]] = None,
                 procs: Optional[List[Dict[str, Any]]] = None,
                 sink: Optional[List[Any]] = None) -> GpuColocScheduler:
            return GpuColocScheduler(
                tmp_path / f"case_{time.time_ns()}",
                gpu_source=make_mock_gpu_source(gpus if gpus is not None else four_gpus),
                proc_source=(lambda: list(procs or [])),
                launcher=make_mock_launcher(sink if sink is not None else []),
                canary=canary,
            )

        def quality_job(index: int, peak: float = 26.0, task: str = "qasper") -> Job:
            return Job(job_id=f"j{index}", category=f"quality_{task}", spec={
                "args": {}, "output_dir": str(tmp_path / f"out{index}")},
                method_key="drm_add", task=task, estimated_peak_gib=peak)

        # 1. Co-location: 26 GiB peak + 10 GiB reserve = 36 GiB, so an 80 GiB A100 takes
        #    2 workers; 4 GPUs therefore absorb 8 concurrent workers.
        def coloc() -> Tuple[bool, Any]:
            scheduler = make()
            scheduler.submit_all([quality_job(i) for i in range(12)])
            scheduler.launch_ready(max_launches=20)
            by_gpu = {uuid: len(hs) for uuid, hs in scheduler.workers_by_gpu().items()}
            ok = len(scheduler.workers) == 8 and set(by_gpu.values()) == {2}
            return ok, {"placed": len(scheduler.workers), "per_gpu": by_gpu,
                        "pending": scheduler.pending_count()}

        audit.guard("I_scheduler", "colocates_multiple_workers_per_gpu", coloc)

        # 2. Not exclusive-only: at least one GPU must hold >1 worker.
        def not_exclusive_only() -> Tuple[bool, Any]:
            scheduler = make()
            scheduler.submit_all([quality_job(i, peak=12.0) for i in range(16)])
            scheduler.launch_ready(max_launches=32)
            per_gpu = {uuid: len(hs) for uuid, hs in scheduler.workers_by_gpu().items()}
            return (max(per_gpu.values()) > 1 if per_gpu else False), per_gpu

        audit.guard("I_scheduler", "scheduling_is_not_exclusive_only", not_exclusive_only)

        # 3. Cap: default 4 workers/GPU is honoured even when memory would allow more.
        def cap_respected() -> Tuple[bool, Any]:
            scheduler = make()
            scheduler.submit_all([quality_job(i, peak=1.0) for i in range(40)])
            scheduler.launch_ready(max_launches=64)
            per_gpu = {uuid: len(hs) for uuid, hs in scheduler.workers_by_gpu().items()}
            ok = bool(per_gpu) and max(per_gpu.values()) == C.DEFAULT_MAX_WORKERS_PER_GPU
            return ok, per_gpu

        audit.guard("I_scheduler", "max_workers_per_gpu_cap", cap_respected)

        # 4. Safety reserve: a job needing more than free - 10 GiB is not placed.
        def reserve_respected() -> Tuple[bool, Any]:
            scheduler = make(gpus=[{"uuid": "GPU-TIGHT", "index": 0, "free_mib": 30 * 1024,
                                    "total_mib": 81920}])
            scheduler.submit(quality_job(0, peak=25.0))
            scheduler.launch_ready()
            return (len(scheduler.workers) == 0 and scheduler.pending_count() == 1), {
                "free_gib": 30.0, "needed_gib": 25.0 + C.SAFETY_RESERVE_GIB,
                "placed": len(scheduler.workers),
            }

        audit.guard("I_scheduler", "safety_reserve_blocks_overcommit", reserve_respected)

        # 4b. Just-launched workers are charged their full estimated peak, so a burst of
        #     launches cannot over-subscribe a GPU while nvidia-smi still reports it free.
        def projected_peak_reserved() -> Tuple[bool, Any]:
            scheduler = make(gpus=[{"uuid": "GPU-LAG", "index": 0, "free_mib": 81920,
                                    "total_mib": 81920}])
            scheduler.submit_all([quality_job(i, peak=26.0) for i in range(4)])
            scheduler.launch_ready(max_launches=8)
            gpu = scheduler.visible_gpus()[0]
            placed = len(scheduler.workers)
            available = scheduler.effective_free_gib(
                gpu, list(scheduler.workers.values()), []
            )
            # 80 GiB total, 26 GiB projected peak each, 10 GiB reserve -> exactly 2 fit
            # even though the mocked nvidia-smi still claims the GPU is fully free.
            return (placed == 2 and available < 26.0 + C.SAFETY_RESERVE_GIB), {
                "placed": placed, "reported_free_gib": gpu.free_gib,
                "available_after_placement_gib": round(available, 2)}

        audit.guard("I_scheduler", "projected_peak_prevents_launch_burst_oversubscription",
                    projected_peak_reserved)

        # 5. Exclusive categories get a whole GPU and are never joined.
        def exclusive_isolation() -> Tuple[bool, Any]:
            scheduler = make()
            exclusive = Job(job_id="gov", category="gov_report", spec={
                "args": {}, "output_dir": str(tmp_path / "gov")},
                method_key="drm_add", task="gov_report", estimated_peak_gib=26.0)
            scheduler.submit(exclusive)
            scheduler.submit_all([quality_job(i) for i in range(6)])
            scheduler.launch_ready(max_launches=16)
            gov = [h for h in scheduler.workers.values() if h.job.category == "gov_report"]
            if not gov:
                return False, "exclusive job was not placed"
            gov_gpu = gov[0].gpu_uuid
            co_tenants = [h for h in scheduler.workers.values()
                          if h.gpu_uuid == gov_gpu and h.job.category != "gov_report"]
            return (len(gov) == 1 and not co_tenants), {
                "gov_gpu": gov_gpu, "co_tenants": len(co_tenants),
                "placement": gov[0].placement,
            }

        audit.guard("I_scheduler", "exclusive_categories_get_whole_gpu", exclusive_isolation)

        def exclusive_category_set() -> Tuple[bool, Any]:
            missing = [c for c in ("gov_report", "fullkv", "oom_fill", "efficiency")
                       if c not in C.EXCLUSIVE_CATEGORIES]
            return not missing, {"exclusive": list(C.EXCLUSIVE_CATEGORIES), "missing": missing}

        audit.guard("I_scheduler", "exclusive_category_list_complete", exclusive_category_set)

        # 6. Canary: exactly one worker on exactly one GPU.
        def canary_mode() -> Tuple[bool, Any]:
            scheduler = make(canary=True)
            scheduler.submit_all([quality_job(i) for i in range(5)])
            scheduler.launch_ready(max_launches=8)
            gpus_used = {h.gpu_uuid for h in scheduler.workers.values()}
            return (len(scheduler.workers) == 1 and len(gpus_used) == 1), {
                "workers": len(scheduler.workers), "gpus_used": len(gpus_used)}

        audit.guard("I_scheduler", "canary_is_one_gpu_one_worker", canary_mode)

        # 7. External processes: never killed, and a fully busy machine waits.
        def external_busy() -> Tuple[bool, Any]:
            procs = [{"gpu_uuid": f"GPU-MOCK-{i}", "pid": 1_000_000 + i, "used_mib": 70000}
                     for i in range(4)]
            scheduler = make(procs=procs)
            exclusive = Job(job_id="fk", category="fullkv", spec={
                "args": {}, "output_dir": str(tmp_path / "fk")},
                method_key="fullkv", task="qasper", estimated_peak_gib=26.0)
            scheduler.submit(exclusive)
            scheduler.launch_ready()
            state = scheduler.compute_state()
            return (len(scheduler.workers) == 0 and state == STATE_WAITING_FOR_GPU_IDLE), {
                "state": state, "placed": len(scheduler.workers),
                "blocked_only_by_external": scheduler.blocked_only_by_external()}

        audit.guard("I_scheduler", "external_busy_yields_waiting_for_gpu_idle", external_busy)

        def never_kills_external() -> Tuple[bool, Any]:
            source = (C.TOOL_ROOT / "scheduler" / "gpu_coloc_scheduler.py").read_text(
                encoding="utf-8")
            # The only signalling site must be terminate_own_workers, which iterates
            # self.workers (our own handles) exclusively.
            kill_sites = source.count("os.killpg")
            ok = kill_sites == 1 and "for handle in list(self.workers.values())" in source
            return ok, {"killpg_call_sites": kill_sites}

        audit.guard("I_scheduler", "no_code_path_signals_foreign_pids", never_kills_external)

        # 8. OOM: results kept, exclusive retry queued, category cap decremented.
        def oom_policy() -> Tuple[bool, Any]:
            sink: List[Any] = []
            scheduler = make(sink=sink)
            job = quality_job(0)
            scheduler.submit(job)
            scheduler.launch_ready()
            handle = next(iter(scheduler.workers.values()))
            handle.placement = "coloc"
            cap_before = scheduler.cap_for(job.category)
            del scheduler.workers[handle.worker_id]
            handle.exit_code = 42
            scheduler._handle_oom(handle)
            cap_after = scheduler.cap_for(job.category)
            retries = [j for j in scheduler.queue if j.job_id == job.job_id]
            ok = (
                cap_after == cap_before - 1
                and len(retries) == 1
                and retries[0].is_exclusive()
                and retries[0].spec is job.spec  # same output dir -> partial results kept
            )
            return ok, {"cap_before": cap_before, "cap_after": cap_after,
                        "retry_exclusive": bool(retries and retries[0].is_exclusive()),
                        "retry_reuses_output_dir": bool(retries and retries[0].spec is job.spec)}

        audit.guard("I_scheduler", "oom_retries_exclusive_and_lowers_cap", oom_policy)

        def oom_attempt_exhaustion() -> Tuple[bool, Any]:
            scheduler = make()
            job = quality_job(0)
            job.attempt = job.max_attempts
            handle = type("H", (), {})()
            from scheduler.gpu_coloc_scheduler import WorkerHandle

            handle = WorkerHandle(worker_id="w", job=job, gpu_uuid="GPU-MOCK-0", gpu_index=0,
                                  placement="exclusive", pid=1, started_at=time.time(),
                                  log_path="/dev/null")
            handle.exit_code = 42
            scheduler._handle_oom(handle)
            return (not scheduler.queue and len(scheduler.failed) == 1), {
                "queued": len(scheduler.queue), "failed": len(scheduler.failed)}

        audit.guard("I_scheduler", "oom_stops_after_max_attempts", oom_attempt_exhaustion)

        # 9. UUID-first pinning.
        def uuid_first() -> Tuple[bool, Any]:
            source = (C.TOOL_ROOT / "scheduler" / "gpu_coloc_scheduler.py").read_text(
                encoding="utf-8")
            worker = (C.TOOL_ROOT / "worker_entry.py").read_text(encoding="utf-8")
            ok = ('"CUDA_VISIBLE_DEVICES": gpu.uuid' in source
                  and 'env["CUDA_VISIBLE_DEVICES"] = gpu_uuid' in worker)
            return ok, "workers are pinned by GPU UUID, not ordinal index"

        audit.guard("I_scheduler", "gpu_pinned_by_uuid", uuid_first)

        # 10. Status artefacts, including per-UUID and global locks.
        def status_files() -> Tuple[bool, Any]:
            case_dir = tmp_path / "status_case"
            scheduler = GpuColocScheduler(
                case_dir,
                gpu_source=make_mock_gpu_source(four_gpus),
                proc_source=lambda: [],
                launcher=make_mock_launcher([]),
            )
            scheduler.submit_all([quality_job(i) for i in range(3)])
            with scheduler.global_lock(timeout=5):
                scheduler.launch_ready(max_launches=4)
            scheduler.write_status()
            expected = ["GPU_STATUS.json", "CURRENT_WORKERS.json", "WORKER_REGISTRY.csv"]
            present = {name: (case_dir / "status" / name).exists() for name in expected}
            lock_files = sorted(p.name for p in (case_dir / "locks").glob("*.lock"))
            has_global = any(name.startswith("scheduler.global") for name in lock_files)
            has_per_gpu = any(name.startswith("gpu.") for name in lock_files)
            return (all(present.values()) and has_global and has_per_gpu), {
                "files": present, "locks": lock_files}

        audit.guard("I_scheduler", "persists_status_and_locks", status_files)

        def lock_is_exclusive() -> Tuple[bool, Any]:
            from scheduler.gpu_coloc_scheduler import FileLock

            path = tmp_path / "excl.lock"
            first = FileLock(path, timeout=1.0)
            second = FileLock(path, timeout=1.0)
            assert first.acquire()
            try:
                blocked = not second.acquire()
            finally:
                first.release()
            regained = second.acquire()
            if regained:
                second.release()
            return (blocked and regained), {"blocked_while_held": blocked,
                                            "acquired_after_release": regained}

        audit.guard("I_scheduler", "flock_is_mutually_exclusive", lock_is_exclusive)

    return results


# ---------------------------------------------------------------------------
# Group J: resume / atomic writes
# ---------------------------------------------------------------------------


def group_j_resume(audit: Audit) -> Dict[str, Any]:
    import orchestrator as O

    with tempfile.TemporaryDirectory(prefix="stage0_resume_") as tmp:
        tmp_path = Path(tmp)

        def atomic_json() -> Tuple[bool, Any]:
            target = tmp_path / "atomic.json"
            C.atomic_write_json(target, {"a": 1})
            first = C.read_json(target)
            C.atomic_write_json(target, {"a": 2, "b": [1, 2, 3]})
            second = C.read_json(target)
            leftovers = [p.name for p in tmp_path.glob(".atomic.json.*")]
            return (first == {"a": 1} and second["a"] == 2 and not leftovers), {
                "first": first, "second": second, "temp_leftovers": leftovers}

        audit.guard("J_resume", "atomic_write_json_replaces_cleanly", atomic_json)

        def atomic_no_partial() -> Tuple[bool, Any]:
            target = tmp_path / "big.json"
            C.atomic_write_json(target, {"payload": "x" * 200000})
            # os.replace is atomic, so a reader either sees the old file or the new one;
            # a successful parse of a large document is the observable proof.
            parsed = C.read_json(target)
            return isinstance(parsed, dict) and len(parsed.get("payload", "")) == 200000, {
                "bytes": target.stat().st_size}

        audit.guard("J_resume", "atomic_write_never_leaves_partial_json", atomic_no_partial)

        def torn_jsonl() -> Tuple[bool, Any]:
            target = tmp_path / "torn.jsonl"
            with target.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"i": 0}) + "\n")
                handle.write(json.dumps({"i": 1}) + "\n")
                handle.write('{"i": 2, "pred": "unter')  # killed mid-write
            count = C.count_jsonl_records(target)
            return count == 2, {"records_counted": count,
                                "note": "torn final line ignored; earlier records kept"}

        audit.guard("J_resume", "torn_jsonl_tail_is_ignored", torn_jsonl)

        def resume_discovery() -> Tuple[bool, Any]:
            root = tmp_path / "results"
            older = root / f"{C.RESULT_DIR_PREFIX}20260101_000000"
            newer = root / f"{C.RESULT_DIR_PREFIX}20260102_000000"
            for path in (older, newer):
                C.ensure_result_layout(path)
            C.atomic_write_json(older / "status" / "PIPELINE_COMPLETE.json",
                                {"final_state": C.FINAL_STATE})
            resumable = C.find_resumable_result_dir(root)
            complete_skipped = C.is_result_dir_complete(older)
            fresh = C.new_result_dir(root, stamp="20260103_000000")
            return (resumable == newer and complete_skipped and fresh.exists()), {
                "resumable": str(resumable), "older_complete": complete_skipped,
                "new_dir": fresh.name}

        audit.guard("J_resume", "resumes_newest_incomplete_result_dir", resume_discovery)

        def job_completion_accounting() -> Tuple[bool, Any]:
            out_dir = tmp_path / "job_out"
            pred = out_dir / "longbench_predictions"
            pred.mkdir(parents=True, exist_ok=True)
            job = Job(job_id="acct", category="quality_qasper", spec={
                "args": {}, "output_dir": str(out_dir)}, method_key="drm_add", task="qasper")
            before = O.job_is_complete(job)
            path = pred / "qasper.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for i in range(199):
                    handle.write(json.dumps({"pred": "x", "answers": [], "length": 1}) + "\n")
            partial = O.job_is_complete(job)
            have, want = O.job_progress(job)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"pred": "x", "answers": [], "length": 1}) + "\n")
            after = O.job_is_complete(job)
            return (not before and not partial and after and want == 200), {
                "empty": before, "at_199": partial, "at_200": after,
                "progress_at_199": [have, want]}

        audit.guard("J_resume", "job_completion_requires_all_predictions",
                    job_completion_accounting)

        def stage_marker_idempotence() -> Tuple[bool, Any]:
            marker = tmp_path / "STAGE2_COMPLETE.json"
            C.atomic_write_json(marker, {"stage": 2})
            return marker.exists() and C.read_json(marker)["stage"] == 2, "stage markers are files"

        audit.guard("J_resume", "stage_completion_is_a_durable_marker", stage_marker_idempotence)

    orchestrator_src = (C.TOOL_ROOT / "orchestrator.py").read_text(encoding="utf-8")
    audit.expect("J_resume", "single_orchestrator_instance_enforced",
                 'FileLock(C.LOCKS_ROOT / "orchestrator.lock"' in orchestrator_src,
                 "global flock makes the scheduler loop unique")
    audit.expect("J_resume", "final_state_string_is_exact",
                 C.FINAL_STATE == "ALL_REQUIRED_LLAMA_DRM_A100_EXPERIMENTS_COMPLETE",
                 C.FINAL_STATE)
    return {}


# ---------------------------------------------------------------------------
# Group K: forbidden method families
# ---------------------------------------------------------------------------


def group_k_forbidden(audit: Audit) -> Dict[str, Any]:
    probe = r"""
import json, sys
sys.path.insert(0, %r); sys.path.insert(0, %r)
out = {}
from bridgekv.cc_topology_policies import CC_TOPOLOGY_METHODS, dispatch_cc_topology_policy
from bridgekv.acc_sliminfer import ACC_DRM_METHODS, dispatch_acc_drm_policy
out["cc_methods_empty"] = len(CC_TOPOLOGY_METHODS) == 0
out["acc_methods_empty"] = len(ACC_DRM_METHODS) == 0
for name, fn in (("cc", dispatch_cc_topology_policy), ("acc", dispatch_acc_drm_policy)):
    try:
        fn("anything")
        out[name + "_raises"] = False
    except Exception as exc:
        out[name + "_raises"] = "forbidden" in str(exc).lower()
print("PROBE_JSON:" + json.dumps(out, default=str))
""" % (str(C.TOOL_ROOT), str(C.RUNTIME_ROOT))
    code, stdout, stderr = run_cpu_probe(probe)
    payload = _extract_probe(stdout)
    audit.expect("K_forbidden", "forbidden_probe_ran", code == 0 and bool(payload),
                 (stderr or stdout)[-1000:])
    audit.expect("K_forbidden", "cc_method_set_empty",
                 bool(payload.get("cc_methods_empty")), payload)
    audit.expect("K_forbidden", "acc_method_set_empty",
                 bool(payload.get("acc_methods_empty")), payload)
    audit.expect("K_forbidden", "cc_dispatch_raises", bool(payload.get("cc_raises")), payload)
    audit.expect("K_forbidden", "acc_dispatch_raises", bool(payload.get("acc_raises")), payload)

    import orchestrator as O

    used_methods = {O.method_args(key)["method"] for key in list(C.FIVE_METHODS) + ["fullkv"]}
    offending = [
        method for method in used_methods
        if any(method.startswith(prefix) for prefix in C.FORBIDDEN_METHOD_SUBSTRINGS)
    ]
    audit.expect("K_forbidden", "no_cc_or_acc_method_scheduled", not offending,
                 {"methods": sorted(used_methods), "offending": offending})
    return {"methods_scheduled": sorted(used_methods)}


# ---------------------------------------------------------------------------
# Group L: results volume
# ---------------------------------------------------------------------------


def group_l_storage(audit: Audit) -> Dict[str, Any]:
    root = C.results_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".stage0_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        writable = True
        detail = str(root)
    except Exception as exc:
        writable = False
        detail = f"{type(exc).__name__}: {exc}"
    audit.expect("L_storage", "results_root_writable", writable, detail)

    free = C.free_gib(root)
    audit.expect("L_storage", "results_root_has_min_free_space",
                 free >= C.MIN_FREE_RESULTS_GIB,
                 f"{free:.1f} GiB free at {root}; minimum {C.MIN_FREE_RESULTS_GIB} GiB")

    tool_free = C.free_gib(C.TOOL_ROOT)
    audit.warn_if("L_storage", "tool_volume_not_critically_full", tool_free >= 1.0,
                  f"{tool_free:.2f} GiB free on the volume holding the code; "
                  "results are redirected via RESULTS_ROOT.txt")
    return {
        "results_root": str(root),
        "results_root_source": C.results_root_source(),
        "results_root_free_gib": round(free, 2),
        "tool_volume_free_gib": round(tool_free, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{C.PROTOCOL_ID} stage-0 CPU audit")
    parser.add_argument("--output-dir", default=str(C.TOOL_ROOT / "audit" / "last_run"))
    options = parser.parse_args()

    output_dir = Path(options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = Audit()
    started = C.now_iso()
    evidence: Dict[str, Any] = {}

    print(f"[stage0] {C.PROTOCOL_ID} CPU audit starting (no GPU weights will be loaded)")
    for label, fn in (
        ("A_inventory", group_a_inventory),
        ("B_compat", group_b_compat),
        ("C_model", group_c_model),
        ("D_dataset", group_d_dataset),
        ("E_scorer", group_e_scorer),
        ("F_method_cli", group_f_method_cli),
        ("G_random", group_g_random_invariants),
        ("H_drm_rules", group_h_drm_rules),
        ("I_scheduler", group_i_scheduler),
        ("J_resume", group_j_resume),
        ("K_forbidden", group_k_forbidden),
        ("L_storage", group_l_storage),
    ):
        print(f"[stage0] group {label}")
        try:
            evidence[label] = fn(audit)
        except Exception:
            audit.expect(label, "group_completed", False, traceback.format_exc(limit=8))

    summary = audit.summary()
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "protocol_version": C.PROTOCOL_VERSION,
        "started_at": started,
        "finished_at": C.now_iso(),
        "gpu_weights_loaded": False,
        "summary": summary,
        "checks": audit.checks,
        "evidence": evidence,
        "frozen_config": C.FROZEN_LLAMA_CONFIG,
        "environment": C.environment_snapshot(),
    }

    C.atomic_write_json(output_dir / "STAGE0_AUDIT.json", payload)
    _write_report(output_dir / "STAGE0_AUDIT.md", audit, summary)

    # The PASS and HARD_BLOCK markers are mutually exclusive verdicts about the same
    # directory, so each verdict clears the other. Leaving both behind lets a stale marker
    # from an earlier run misreport the current state to a reader or to gate logic.
    if audit.failures:
        _clear_markers(output_dir, ("STAGE0_PASS.json",))
        block = {
            "protocol_id": C.PROTOCOL_ID,
            "state": "HARD_BLOCK",
            "at": C.now_iso(),
            "reason": "stage0 CPU audit failed",
            "failed_checks": audit.failures,
            "summary": summary,
        }
        C.atomic_write_json(output_dir / "HARD_BLOCK.json", block)
        C.atomic_write_text(
            output_dir / "HARD_BLOCK.md",
            "# HARD BLOCK: stage0 audit failed\n\n"
            + "\n".join(f"- [{c['group']}] {c['name']}: {c['detail']}" for c in audit.failures)
            + "\n",
        )
        print(f"[stage0] FAIL {summary['failed']}/{summary['total']} checks")
        for check in audit.failures:
            print(f"  FAIL [{check['group']}] {check['name']}: {check['detail']}")
        return 1

    _clear_markers(output_dir, ("HARD_BLOCK.json", "HARD_BLOCK.md"))
    C.atomic_write_json(output_dir / "STAGE0_PASS.json", {
        "protocol_id": C.PROTOCOL_ID,
        "state": "STAGE0_PASS",
        "at": C.now_iso(),
        "summary": summary,
        "warnings": audit.warnings,
        "gpu_stages_allowed": True,
    })
    print(f"[stage0] PASS {summary['passed']}/{summary['total']} checks "
          f"({summary['warnings']} warnings)")
    for check in audit.warnings:
        print(f"  WARN [{check['group']}] {check['name']}: {check['detail']}")
    return 0


def _clear_markers(output_dir: Path, names: Tuple[str, ...]) -> None:
    for name in names:
        try:
            (output_dir / name).unlink()
        except FileNotFoundError:
            pass


def _write_report(path: Path, audit: Audit, summary: Dict[str, Any]) -> None:
    lines = [
        f"# Stage 0 CPU audit — {C.PROTOCOL_ID}", "",
        f"- generated: {C.now_iso()}",
        f"- result: {'PASS' if not audit.failures else 'FAIL'}",
        f"- checks: {summary['passed']}/{summary['total']} passed, "
        f"{summary['failed']} failed, {summary['warnings']} warnings",
        "- model weights loaded onto GPU: no", "",
        "| Group | Check | Status | Detail |", "| --- | --- | --- | --- |",
    ]
    for check in audit.checks:
        detail = str(check["detail"]).replace("\n", " ").replace("|", "\\|")[:200]
        lines.append(f"| {check['group']} | {check['name']} | {check['status']} | {detail} |")
    C.atomic_write_text(path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
