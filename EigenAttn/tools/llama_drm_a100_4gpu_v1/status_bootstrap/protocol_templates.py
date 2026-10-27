#!/usr/bin/env python3
"""Protocol document templates written into each result directory at bootstrap.

The documents are regenerated on every orchestrator start so that a resumed run always
carries an accurate identity record (resolved paths, hashes, environment) rather than a
stale copy from the first launch.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C


def _method_table() -> str:
    header = "| Arm | --method | rescue_policy | rescue_score | rescue_ratio | seeded |"
    lines = [header, "| --- | --- | --- | --- | --- | --- |"]
    for key, spec in C.METHOD_CLI_MAP.items():
        lines.append(
            "| {label} | {method} | {policy} | {score} | {ratio} | {seed} |".format(
                label=spec["label"],
                method=spec["method"],
                policy=spec.get("rescue_policy", "-"),
                score=spec.get("rescue_score", "-"),
                ratio=spec.get("rescue_ratio", "-"),
                seed="yes" if spec.get("needs_seed") else "no",
            )
        )
    return "\n".join(lines)


def protocol_md(model_path: Optional[Path], longbench_dir: Optional[Path]) -> str:
    cfg = C.FROZEN_LLAMA_CONFIG
    return f"""# {C.PROTOCOL_ID}

- protocol version: {C.PROTOCOL_VERSION}
- generated: {C.now_iso()}

## Model identity

- name: {C.MODEL_NAME}
- resolved path: `{model_path or 'NOT_FOUND'}`
- revision: `{C.MODEL_REVISION}`
- max_position_embeddings: {cfg['max_position_embeddings']}
- prompt truncation length (`--max_model_len`): {cfg['max_model_len']}
- chat template: none (base model). Prompts are fed verbatim; no template is invented.

The local snapshot carries no Hugging Face revision metadata. `{C.MODEL_REVISION}` is
recorded literally: a fabricated commit hash would make this identity record wrong.

## Dataset identity

- LongBench directory: `{longbench_dir or 'NOT_FOUND'}`
- tasks: 17 official tasks, {C.EXPECTED_TOTAL_SAMPLES_17} samples total
- six-task subset: {', '.join(C.SIX_TASKS)} ({C.EXPECTED_TOTAL_SAMPLES_6} samples)
- downloads are disabled; a missing task file is a hard error, never a silent fetch

## Frozen configuration

| Knob | Value |
| --- | --- |
| dtype | {cfg['dtype']} |
| batch size | {cfg['batch_size']} |
| decoding | greedy (do_sample={cfg['do_sample']}, num_beams={cfg['num_beams']}) |
| block size | {cfg['block_size']} |
| token unit | {cfg['token_unit_size']} |
| gamma (swap threshold) | {cfg['gamma_swap_threshold']} |
| local query window | {cfg['local_query_window']} |
| prune layers | {cfg['prune_layers']} |
| retain tokens | {cfg['retain_tokens']} |
| rescue ratio (main DRM-Add) | {cfg['rescue_ratio_main']} |
| rescue M per prune layer | {list(C.RESCUE_M_PER_PRUNE_LAYER)} |
| attention backend | {cfg['backend']} ({cfg['backend_reason']}) |

Retained pages per prune layer are {list(C.RETAINED_PAGES_PER_PRUNE_LAYER)}
(retain_tokens / block_size), so rescue_ratio {cfg['rescue_ratio_main']} yields
M = {list(C.RESCUE_M_PER_PRUNE_LAYER)}.

## Method CLI mapping

{_method_table()}

## Scoring

Each task is scored independently with the official LongBench scorer, then the macro
average is the arithmetic mean over task scores. Pooled sample averaging is forbidden.

## Forbidden

- CC and ACC method families (their modules are absent and their dispatchers raise).
- Any network access: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / HF_DATASETS_OFFLINE are
  set in every subprocess.
- Killing GPU processes that this pipeline did not start.
"""


def scheduling_md() -> str:
    return f"""# GPU co-location policy ({C.PROTOCOL_ID})

## Co-location, not one task per GPU

A GPU accepts an additional worker while both hold:

- `free_memory >= estimated_peak + {C.SAFETY_RESERVE_GIB} GiB` safety reserve
- `workers_on_gpu < cap` (default cap {C.DEFAULT_MAX_WORKERS_PER_GPU} per GPU)

`estimated_peak = max(actual_peak * {C.PEAK_MULTIPLIER}, actual_peak + {C.PEAK_HEADROOM_GIB} GiB)`,
where `actual_peak` is the largest peak measured so far (seeded by the canary stage).

## Canary

Stage 1 runs a single worker on a single GPU. Its measured peak sizes every later
co-location decision.

## Exclusive categories

{chr(10).join(f'- `{category}`' for category in C.EXCLUSIVE_CATEGORIES)}

These get an entire GPU: their memory profile (Full-KV, OOM fill, long context) or their
timing measurement (efficiency) is invalidated by a co-tenant. `gov_report` is exclusive
because its 512-token generations dominate both memory and runtime.

## OOM handling

1. Partial predictions are kept; the runner resumes per sample.
2. The job is requeued as exclusive.
3. The co-location cap for that category is decremented.
4. After `max_attempts` the job is recorded as FAILED_OOM_EXHAUSTED.

## External processes

Foreign PIDs are never signalled. When every visible GPU is held by foreign work the
scheduler reports `WAITING_FOR_GPU_IDLE` and idles until capacity appears.

## Identity

GPUs are tracked by `nvidia-smi` UUID, never by ordinal index, and workers are pinned
with `CUDA_VISIBLE_DEVICES=<uuid>`.
"""


def resume_md() -> str:
    return f"""# Resume policy ({C.PROTOCOL_ID})

- A single orchestrator instance is enforced by an flock on `locks/orchestrator.lock`.
- `orchestrator.py --resume` adopts the newest result directory whose
  `status/PIPELINE_COMPLETE.json` does not report `{C.FINAL_STATE}`.
- Stage completion is a file: `status/STAGE<N>_COMPLETE.json`.
- Job completion is derived from artefacts, not memory: a quality job is complete when
  `longbench_predictions/<task>.jsonl` holds every expected sample.
- Sample-level resume is the runner's own behaviour; existing predictions are appended
  to, never recomputed.
- All status writes are atomic (temp file + `os.replace`), so a kill never leaves a
  reader with a half-written JSON document.
- systemd restarts the unit on any exit; an incomplete stage exits non-zero on purpose.
"""


def write_all(result_dir: Path, *, model_path: Optional[Path] = None,
              longbench_dir: Optional[Path] = None) -> List[str]:
    protocol_dir = Path(result_dir) / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    documents = {
        "LLAMA_DRM_PROTOCOL.md": protocol_md(model_path, longbench_dir),
        "GPU_COLOCATION_POLICY.md": scheduling_md(),
        "RESUME_POLICY.md": resume_md(),
    }
    for name, text in documents.items():
        C.atomic_write_text(protocol_dir / name, text)
        written.append(name)

    identity: Dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "protocol_version": C.PROTOCOL_VERSION,
        "generated_at": C.now_iso(),
        "model": {
            "name": C.MODEL_NAME,
            "family": C.MODEL_FAMILY,
            "resolved_path": str(model_path) if model_path else None,
            "revision": C.MODEL_REVISION,
            "config_sha256": (
                C.sha256_file(Path(model_path) / "config.json")
                if model_path and (Path(model_path) / "config.json").exists() else None
            ),
        },
        "longbench": {
            "resolved_path": str(longbench_dir) if longbench_dir else None,
            "tasks_17": list(C.SEVENTEEN_TASKS),
            "tasks_6": list(C.SIX_TASKS),
            "expected_samples_17": C.EXPECTED_TOTAL_SAMPLES_17,
            "expected_samples_6": C.EXPECTED_TOTAL_SAMPLES_6,
        },
        "frozen_config": C.FROZEN_LLAMA_CONFIG,
        "method_cli_map": C.METHOD_CLI_MAP,
        "scheduling": {
            "max_workers_per_gpu": C.DEFAULT_MAX_WORKERS_PER_GPU,
            "safety_reserve_gib": C.SAFETY_RESERVE_GIB,
            "peak_multiplier": C.PEAK_MULTIPLIER,
            "peak_headroom_gib": C.PEAK_HEADROOM_GIB,
            "exclusive_categories": list(C.EXCLUSIVE_CATEGORIES),
        },
    }
    C.atomic_write_json(protocol_dir / "PROTOCOL_IDENTITY.json", identity)
    written.append("PROTOCOL_IDENTITY.json")
    return written
