#!/usr/bin/env python3
"""Shared primitives for the LLAMA_DRM_A100_4GPU_V1 unattended pipeline.

Import-safe on a CPU-only process: no torch import, no CUDA init, no network. The
stage-0 auditor depends on that property.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROTOCOL_ID = "LLAMA_DRM_A100_4GPU_V1"
PROTOCOL_VERSION = "1.0.0"

TOOL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOL_ROOT.parents[1]
RUNTIME_ROOT = TOOL_ROOT / "runtime"
ADAPTERS_ROOT = TOOL_ROOT / "adapters"
PROTOCOL_DIR = TOOL_ROOT / "protocol_ulb_v1"
LOCKS_ROOT = TOOL_ROOT / "locks"

RUNNER_SCRIPT = RUNTIME_ROOT / "experiments" / "run_llama_sliminfer_variants.py"
WORKER_ENTRY = TOOL_ROOT / "worker_entry.py"
ORCHESTRATOR = TOOL_ROOT / "orchestrator.py"
STAGE0_AUDIT = TOOL_ROOT / "audit" / "stage0_cpu_audit.py"

PYTHON_BIN = "/home/bai-project/a1804298693/.conda/envs/llamaqkv/bin/python"

RESULT_DIR_PREFIX = "llama_drm_a100_4gpu_v1_"
RESULT_DIR_RE = re.compile(r"^llama_drm_a100_4gpu_v1_(\d{8}_\d{6})(_\d{2})?$")

FINAL_STATE = "ALL_REQUIRED_LLAMA_DRM_A100_EXPERIMENTS_COMPLETE"

# ---------------------------------------------------------------------------
# Results root
# ---------------------------------------------------------------------------

#: The repo lives on a root filesystem that is effectively full, so results must be
#: written to a data volume. Resolution order is explicit and auditable rather than
#: implicit: env override, then a pinned pointer file, then tool-local `results/`.
RESULTS_ROOT_POINTER = TOOL_ROOT / "RESULTS_ROOT.txt"
RESULTS_ROOT_ENV = "LLAMA_DRM_RESULTS_ROOT"
DEFAULT_RESULTS_ROOT = TOOL_ROOT / "results"

#: A full 17-task x 6-arm sweep writes several GiB of predictions and traces; refuse to
#: start rather than dying halfway through with ENOSPC.
MIN_FREE_RESULTS_GIB = 20.0


def results_root() -> Path:
    env_value = os.environ.get(RESULTS_ROOT_ENV, "").strip()
    if env_value:
        return Path(env_value).expanduser()
    if RESULTS_ROOT_POINTER.exists():
        pinned = RESULTS_ROOT_POINTER.read_text(encoding="utf-8").strip()
        pinned = "\n".join(
            line for line in pinned.splitlines() if line.strip() and not line.lstrip().startswith("#")
        ).strip()
        if pinned:
            return Path(pinned).expanduser()
    return DEFAULT_RESULTS_ROOT


def results_root_source() -> str:
    if os.environ.get(RESULTS_ROOT_ENV, "").strip():
        return f"env:{RESULTS_ROOT_ENV}"
    if RESULTS_ROOT_POINTER.exists():
        return f"pointer:{RESULTS_ROOT_POINTER}"
    return "default:tool_local_results"


#: The LongBench scorer needs jieba / rouge / fuzzywuzzy, which are absent from the
#: llamaqkv environment and cannot be installed (no network). They are vendored to a data
#: volume and prepended to PYTHONPATH instead of being installed into the shared env.
VENDORED_DEPS_POINTER = TOOL_ROOT / "VENDORED_DEPS.txt"
VENDORED_DEPS_ENV = "LLAMA_DRM_VENDORED_DEPS"
REQUIRED_SCORER_MODULES: Tuple[str, ...] = ("jieba", "rouge", "fuzzywuzzy")


def vendored_deps_dir() -> Optional[Path]:
    env_value = os.environ.get(VENDORED_DEPS_ENV, "").strip()
    if env_value:
        return Path(env_value).expanduser()
    if VENDORED_DEPS_POINTER.exists():
        for line in VENDORED_DEPS_POINTER.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return Path(line).expanduser()
    return None


def free_gib(path: os.PathLike | str) -> float:
    """Free space on the filesystem holding `path` (nearest existing ancestor)."""
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(str(target))
    return usage.free / GIB


# ---------------------------------------------------------------------------
# Model / dataset identity
# ---------------------------------------------------------------------------

MODEL_PATH_CANDIDATES: Tuple[str, ...] = (
    "/mnt/d/jiaqizhang/llama3",
    "/data/jiaqizhang/llama3",
    "/d/jiaqizhang/llama3",
    "/home/bai-project/jiaqizhang/llama3",
)

LONGBENCH_PATH_CANDIDATES: Tuple[str, ...] = (
    "/home/bai-project/jiaqizhang/EigenAttn/data/LongBench",
    "/home/bai-project/jiaqizhang/data/LongBench",
    "/home/bai-project/jiaqizhang/LongBench",
)

#: The local snapshot carries no HF revision metadata. Inventing one would make the
#: identity matrix silently wrong, so the unknown value is explicit and propagated.
MODEL_REVISION = "UNKNOWN_LOCAL_SNAPSHOT"
MODEL_NAME = "Llama-3-8B"
MODEL_FAMILY = "llama"

MODEL_MAX_POSITION_EMBEDDINGS = 8192
#: Prompts are middle-truncated to this length so prompt + max_new_tokens (<=512)
#: stays inside Llama-3's 8192-position RoPE range.
MAX_MODEL_LEN = 7500

# ---------------------------------------------------------------------------
# Frozen LLaMA DRM configuration
# ---------------------------------------------------------------------------

GIB = 1024 ** 3

FROZEN_LLAMA_CONFIG: Dict[str, Any] = {
    "model_family": MODEL_FAMILY,
    "model_name": MODEL_NAME,
    "model_revision": MODEL_REVISION,
    "dtype": "bf16",
    "batch_size": 1,
    "greedy": True,
    "do_sample": False,
    "num_beams": 1,
    "block_size": 64,
    "token_unit_size": 8,
    "gamma_swap_threshold": 0.9,
    "local_query_window": 4,
    "prune_layers": [10, 20, 30],
    "retain_tokens": [8192, 4096, 2048],
    "num_hidden_layers": 32,
    "rescue_ratio_main": 0.05,
    "rescue_m_per_prune_layer": [6, 3, 2],
    "backend": "sdpa",
    "backend_reason": "flash_attn is not installed in the llamaqkv environment",
    "max_model_len": MAX_MODEL_LEN,
    "max_position_embeddings": MODEL_MAX_POSITION_EMBEDDINGS,
    "preserve_first_block": True,
    "block_strategy": "fixed",
    "scoring": "token_unit_qk",
}

#: retain_tokens / block_size = retained pages per prune layer (128/64/32). At
#: rescue_ratio=0.05 with round-half-up this yields the frozen M = 6/3/2.
RETAINED_PAGES_PER_PRUNE_LAYER: Tuple[int, ...] = (8192 // 64, 4096 // 64, 2048 // 64)
RESCUE_M_PER_PRUNE_LAYER: Tuple[int, ...] = (6, 3, 2)

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

SEVENTEEN_TASKS: Tuple[str, ...] = (
    "passage_retrieval_en", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
    "musique", "gov_report", "qmsum", "multi_news", "vcsum", "passage_count",
    "trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p",
)

SIX_TASKS: Tuple[str, ...] = (
    "qasper", "hotpotqa", "2wikimqa", "musique", "gov_report", "repobench-p",
)

EXPECTED_TOTAL_SAMPLES_17 = 3950
EXPECTED_TOTAL_SAMPLES_6 = 1500

# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

#: Authoritative CLI mapping. Every arm except Full-KV uses --method sliminfer_original;
#: arms differ only in rescue_policy / rescue_score / rescue_ratio.
METHOD_CLI_MAP: Dict[str, Dict[str, Any]] = {
    "fullkv": {
        "label": "Full-KV", "method": "fullkv", "is_drm": False, "exclusive_gpu": True,
    },
    "sliminfer_original": {
        "label": "SlimInfer-Original", "method": "sliminfer_original",
        "rescue_policy": "none", "rescue_score": "delta_o", "is_drm": False,
    },
    "sliminfer_larger": {
        "label": "SlimInfer-Larger", "method": "sliminfer_original",
        "rescue_policy": "sliminfer_larger", "rescue_score": "delta_o", "is_drm": False,
    },
    "random_add": {
        "label": "Random-Add", "method": "sliminfer_original",
        "rescue_policy": "add", "rescue_score": "random", "rescue_ratio": 0.05,
        "is_drm": False, "needs_seed": True,
    },
    "drm_add": {
        "label": "DRM-Add", "method": "sliminfer_original",
        "rescue_policy": "add", "rescue_score": "drm", "rescue_ratio": 0.05,
        "is_drm": True,
    },
    "drm_swap": {
        "label": "DRM-Swap", "method": "sliminfer_original",
        "rescue_policy": "budget_neutral_drm_swap", "rescue_score": "drm",
        "is_drm": True,
    },
}

FIVE_METHODS: Tuple[str, ...] = (
    "sliminfer_original", "sliminfer_larger", "random_add", "drm_add", "drm_swap",
)

RANDOM_SEEDS: Tuple[int, ...] = (0, 1, 2)

FORBIDDEN_METHOD_SUBSTRINGS: Tuple[str, ...] = ("cc_", "acc_")

# ---------------------------------------------------------------------------
# GPU / scheduling constants
# ---------------------------------------------------------------------------

SAFETY_RESERVE_GIB = 10.0
PEAK_MULTIPLIER = 1.20
PEAK_HEADROOM_GIB = 4.0
DEFAULT_MAX_WORKERS_PER_GPU = 4
CANARY_WORKERS = 1

#: Categories that must never share a GPU: long-context / large-generation arms, the
#: Full-KV memory baseline, OOM fills, and efficiency measurements whose timings are
#: invalidated by contention.
EXCLUSIVE_CATEGORIES: Tuple[str, ...] = (
    "gov_report", "long_context_32k", "fullkv", "oom_fill", "efficiency",
)

LONG_CONTEXT_THRESHOLD_TOKENS = 32768

# ---------------------------------------------------------------------------
# Offline environment
# ---------------------------------------------------------------------------


def diskf_workspace() -> Path:
    """Unified F-disk workspace for all bulky I/O (results/logs/tmp/cache)."""
    pointer = TOOL_ROOT / "DISKF_WORKSPACE.txt"
    env_value = os.environ.get("LLAMA_DRM_DISKF_WORKSPACE", "").strip()
    if env_value:
        return Path(env_value).expanduser()
    if pointer.exists():
        for line in pointer.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return Path(line).expanduser()
    return Path("/diskF/jiaqizhang/llama_drm_a100_4gpu_v1")


def ensure_diskf_workspace() -> Path:
    """Create F-disk subdirs so ENOSPC on `/` cannot claim tmp/cache/logs."""
    root = diskf_workspace()
    for sub in ("results", "logs", "tmp", "cache/huggingface", "cache/torch",
                "exports", "workdir"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def offline_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Environment for a worker subprocess with every network path disabled."""
    env = dict(os.environ)
    ws = ensure_diskf_workspace()
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "DISABLE_TELEMETRY": "1",
        "NO_PROXY": "*",
        "no_proxy": "*",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "PYTHONHASHSEED": "0",
        "PROTOCOL_ID": PROTOCOL_ID,
        # Keep all expandable I/O off the near-full root filesystem.
        "TMPDIR": str(ws / "tmp"),
        "TMP": str(ws / "tmp"),
        "TEMP": str(ws / "tmp"),
        "HF_HOME": str(ws / "cache" / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(ws / "cache" / "huggingface" / "hub"),
        "TRANSFORMERS_CACHE": str(ws / "cache" / "huggingface" / "transformers"),
        "TORCH_HOME": str(ws / "cache" / "torch"),
        "XDG_CACHE_HOME": str(ws / "cache"),
        "LLAMA_DRM_RESULTS_ROOT": str(ws / "results"),
        "LLAMA_DRM_DISKF_WORKSPACE": str(ws),
    })
    for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                     "ALL_PROXY", "all_proxy"):
        env.pop(proxy_var, None)
    pythonpath = [str(TOOL_ROOT), str(RUNTIME_ROOT)]
    vendored = vendored_deps_dir()
    if vendored is not None:
        pythonpath.append(str(vendored))
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


OFFLINE_FLAG_KEYS: Tuple[str, ...] = (
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
)

# ---------------------------------------------------------------------------
# Time / hashing / atomic IO
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: os.PathLike | str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                  separators=(",", ":")))


def atomic_write_text(path: os.PathLike | str, text: str) -> Path:
    """Temp file in the target directory + os.replace, so readers never see a partial file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.",
                                    suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def atomic_write_json(path: os.PathLike | str, payload: Any, *, sort_keys: bool = True) -> Path:
    return atomic_write_text(
        path, json.dumps(payload, indent=2, sort_keys=sort_keys, ensure_ascii=False) + "\n"
    )


def atomic_write_csv(path: os.PathLike | str, rows: Sequence[Dict[str, Any]],
                     fieldnames: Sequence[str]) -> Path:
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return atomic_write_text(path, buffer.getvalue())


def read_json(path: os.PathLike | str, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def append_jsonl(path: os.PathLike | str, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: os.PathLike | str) -> List[Any]:
    target = Path(path)
    if not target.exists():
        return []
    out: List[Any] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn final line is expected after a hard kill; earlier records stay valid.
                break
    return out


def count_jsonl_records(path: os.PathLike | str) -> int:
    return len(read_jsonl(path))


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def resolve_model_path(candidates: Iterable[str] = MODEL_PATH_CANDIDATES) -> Optional[Path]:
    """First candidate that looks like a complete local HF snapshot."""
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_dir() or not (path / "config.json").exists():
            continue
        if any(path.glob("*.safetensors")) or any(path.glob("*.bin")):
            return path.resolve()
    return None


def resolve_longbench_dir(candidates: Iterable[str] = LONGBENCH_PATH_CANDIDATES) -> Optional[Path]:
    """First candidate containing all 17 official LongBench task files."""
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_dir():
            continue
        if all((path / f"{task}.jsonl").exists() for task in SEVENTEEN_TASKS):
            return path.resolve()
    return None


def longbench_task_file(longbench_dir: os.PathLike | str, task: str) -> Path:
    return Path(longbench_dir) / f"{task}.jsonl"


def python_bin() -> str:
    if Path(PYTHON_BIN).exists():
        return PYTHON_BIN
    fallback = shutil.which("python3") or shutil.which("python")
    if fallback:
        return fallback
    raise RuntimeError("No usable python interpreter found")


# ---------------------------------------------------------------------------
# Result directory discovery
# ---------------------------------------------------------------------------

RESULT_SUBDIRS: Tuple[str, ...] = (
    "status", "logs", "runs", "analysis", "reports", "protocol", "audit", "locks", "specs",
)


def list_result_dirs(root: Optional[os.PathLike | str] = None) -> List[Path]:
    base = Path(root) if root is not None else results_root()
    if not base.is_dir():
        return []
    dirs = [p for p in base.iterdir() if p.is_dir() and RESULT_DIR_RE.match(p.name)]
    return sorted(dirs, key=lambda p: p.name)


def is_result_dir_complete(result_dir: os.PathLike | str) -> bool:
    payload = read_json(Path(result_dir) / "status" / "PIPELINE_COMPLETE.json", default={}) or {}
    return payload.get("final_state") == FINAL_STATE


def find_resumable_result_dir(root: Optional[os.PathLike | str] = None) -> Optional[Path]:
    """Newest incomplete result dir, so --resume continues instead of forking a new run."""
    for candidate in reversed(list_result_dirs(root)):
        if not is_result_dir_complete(candidate):
            return candidate
    return None


def new_result_dir(root: Optional[os.PathLike | str] = None,
                   stamp: Optional[str] = None) -> Path:
    base = Path(root) if root is not None else results_root()
    stamp = stamp or now_stamp()
    target = base / f"{RESULT_DIR_PREFIX}{stamp}"
    suffix = 0
    while target.exists():
        suffix += 1
        target = base / f"{RESULT_DIR_PREFIX}{stamp}_{suffix:02d}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def ensure_result_layout(result_dir: os.PathLike | str) -> Path:
    root = Path(result_dir)
    for sub in RESULT_SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------


def host_info() -> Dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "user": os.environ.get("USER", ""),
        "cwd": os.getcwd(),
    }


def nvidia_smi_query(fields: Sequence[str], query: str = "--query-gpu") -> List[Dict[str, str]]:
    """Parse `nvidia-smi --query-* --format=csv,noheader,nounits`; [] when unavailable."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return []
    cmd = [binary, f"{query}={','.join(fields)}", "--format=csv,noheader,nounits"]
    try:
        raw = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    if raw.returncode != 0:
        return []
    rows: List[Dict[str, str]] = []
    for line in raw.stdout.strip().splitlines():
        parts = [part.strip() for part in line.strip().split(",")]
        if len(parts) == len(fields):
            rows.append(dict(zip(fields, parts)))
    return rows


def python_package_versions(python: Optional[str] = None) -> Dict[str, str]:
    """Query versions out-of-process so the CPU auditor never imports torch itself."""
    interpreter = python or python_bin()
    script = (
        "import json\n"
        "out={}\n"
        "for name in ('torch','transformers','numpy','datasets','accelerate','flash_attn'):\n"
        "    try:\n"
        "        out[name]=getattr(__import__(name),'__version__','unknown')\n"
        "    except Exception:\n"
        "        out[name]='MISSING'\n"
        "try:\n"
        "    import torch\n"
        "    out['cuda_version']=str(torch.version.cuda)\n"
        "except Exception:\n"
        "    out['cuda_version']='MISSING'\n"
        "print(json.dumps(out))\n"
    )
    try:
        proc = subprocess.run(
            [interpreter, "-c", script], capture_output=True, text=True, timeout=600,
            env=offline_env({"CUDA_VISIBLE_DEVICES": ""}), check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        pass
    return {}


def environment_snapshot() -> Dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "captured_at": now_iso(),
        "host": host_info(),
        "python_bin": python_bin(),
        "packages": python_package_versions(),
        "gpus": nvidia_smi_query(
            ["index", "uuid", "name", "memory.total", "memory.used", "memory.free"]
        ),
        "results_root": str(results_root()),
        "results_root_source": results_root_source(),
        "results_root_free_gib": round(free_gib(results_root()), 2),
        "offline_flags": {key: offline_env()[key] for key in OFFLINE_FLAG_KEYS},
    }
