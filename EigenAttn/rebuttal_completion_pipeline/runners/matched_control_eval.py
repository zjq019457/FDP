#!/usr/bin/env python3
"""Matched-control eval with phased GPU-parallel execution.

Phases:
  build      — truncate FDP basis + compress_only (disk-serialized via flock)
  wikitext2  — PPL
  c4         — PPL
  zero_shot  — ZS bundle hs_arc | wg_piqa
  cleanup    — delete safetensors after all evals
  all        — legacy sequential (build+evals+cleanup)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
LEGACY = ROOT / "rebuttal_legacy_reproduction"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "runners"))
from common import (  # noqa: E402
    ARTIFACT_ROOT,
    C4_TEXTS,
    MODEL,
    PYTHON_BIN,
    TASK_YAML,
    WIKITEXT,
    artifact_dir,
    atomic_write_json,
    evaluation_hash,
    now_iso,
)
from kv_accounting import load_low_rank, realized_kv_bytes  # noqa: E402
from truncate_basis_to_ranks import truncate_basis  # noqa: E402

FDP_MODEL = Path("/diskF/jiaqizhang/rebuttal_completion_op_artifacts/operating_points/moderate/fdp/final_model")
FDP_BASIS = Path(
    "/diskF/jiaqizhang/rebuttal_completion_op_artifacts/operating_points/moderate/fdp/"
    "adaptive_basis/eigen_basis_moderate_perlayer_eb.pt"
)
# Transient MC weights stay on diskF (now has headroom); cold archives live on diskD via symlink.
MC_ROOT = Path("/diskF/jiaqizhang/rebuttal_completion_op_artifacts/matched_control_models")
COMPRESS_LOCK = Path("/diskF/jiaqizhang/rebuttal_completion_op_artifacts/.mc_compress.lock")
MIN_FREE_BYTES = 18 * (1024**3)


def _disk_free(path: Path) -> int:
    st = os.statvfs(path)
    return int(st.f_bavail * st.f_frsize)


def _with_compress_lock(fn):
    import fcntl
    import time

    COMPRESS_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with COMPRESS_LOCK.open("w") as lf:
        while True:
            try:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(3)
                continue
            # Also wait for free disk under lock so only one builder checks/space.
            check = MC_ROOT if MC_ROOT.exists() else Path("/diskF/jiaqizhang")
            while _disk_free(check) < MIN_FREE_BYTES:
                time.sleep(10)
            try:
                return fn()
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def _pairs_from_schedule(sched: dict) -> List[Tuple[int, int]]:
    return [(int(L["k_proj"]["rank"]), int(L["v_proj"]["rank"])) for L in sched["schedule"]]


def _model_ready(model_dir: Path) -> bool:
    return (model_dir / "model.safetensors.index.json").exists() or any(model_dir.glob("model-*.safetensors"))


def _compress(pairs: List[Tuple[int, int]], out_model: Path, work: Path) -> Path:
    def _do():
        work.mkdir(parents=True, exist_ok=True)
        basis_dst = work / "truncated_basis.pt"
        truncate_basis(FDP_BASIS, pairs, basis_dst)
        if out_model.exists():
            shutil.rmtree(out_model)
        out_model.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "EigenAttn") + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        cmd = [
            PYTHON_BIN, "-u", str(ROOT / "EigenAttn" / "main_eigen_attn.py"),
            "--model", MODEL, "--net", "Llama-3-8b", "--dtype", "bf16",
            "--error_budget", "0.06", "--avg_dim", "4096",
            "--seqlen", "2048", "--nsamples", "64", "--seed", "2",
            "--calib_dataset", "c4", "--compress_only", "--attn_implementation", "eager",
            "--eigen_basis_path", str(basis_dst),
            "--save_dir", str(out_model),
            "--output_dir", str(out_model.parent),
            "--cache_dir", str(ARTIFACT_ROOT / "HF_cache"),
        ]
        with (work / "compress_stdout.log").open("w") as fo, (work / "compress_stderr.log").open("w") as fe:
            cp = subprocess.run(cmd, cwd=str(ROOT / "EigenAttn"), env=env, stdout=fo, stderr=fe, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"compress failed rc={cp.returncode}; see {work}/compress_stderr.log")
        got = load_low_rank(out_model / "low_rank_config.json")
        if got != pairs:
            atomic_write_json(work / "RANK_MISMATCH.json", {"expected": pairs, "got": got})
            atomic_write_json(out_model / "low_rank_config.json", [[a, b] for a, b in pairs])
        return out_model

    return _with_compress_lock(_do)


def _run_ppl(model: Path, out: Path, dataset: str, method: str) -> dict:
    if (out / "ppl_aggregate.json").exists():
        return json.loads((out / "ppl_aggregate.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "EigenAttn") + os.pathsep + str(LEGACY) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        PYTHON_BIN, "-u", str(LEGACY / "runners" / "ppl_instrumented.py"),
        "--dataset", dataset, "--model", str(model), "--tok", MODEL,
        "--out_dir", str(out), "--dtype", "bf16", "--method", method,
        "--evaluation_protocol_hash", evaluation_hash(),
    ]
    if dataset == "wikitext2":
        cmd += ["--wikitext2_dir", WIKITEXT]
    else:
        cmd += ["--c4_texts", C4_TEXTS]
    subprocess.run(cmd, cwd=str(LEGACY), env=env, check=True)
    return json.loads((out / "ppl_aggregate.json").read_text())


def _run_zs(model: Path, out: Path, tasks: str, method: str) -> dict:
    if (out / "zero_shot_aggregate.json").exists():
        return json.loads((out / "zero_shot_aggregate.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "EigenAttn") + os.pathsep + str(LEGACY) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        PYTHON_BIN, "-u", str(LEGACY / "runners" / "zero_shot_instrumented.py"),
        "--model", str(model), "--tokenizer_model", MODEL,
        "--task_yaml_dir", TASK_YAML, "--tasks", tasks,
        "--out_dir", str(out), "--dtype", "bf16", "--net", "llama-3-8b",
        "--method", method, "--low_rank_config", str(model / "low_rank_config.json"),
    ]
    subprocess.run(cmd, cwd=str(LEGACY), env=env, check=True)
    return json.loads((out / "zero_shot_aggregate.json").read_text())


def _cleanup(model_dir: Path, art: Path) -> None:
    n = 0
    for p in model_dir.glob("model*.safetensors"):
        p.unlink()
        n += 1
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        idx.unlink()
        n += 1
    atomic_write_json(art / "WEIGHTS_DELETED_AFTER_EVAL.json", {
        "deleted_from": str(model_dir), "n_files": n, "reason": "disk_conservation",
        "generated_at": now_iso(),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--phase", default="all",
                    choices=["all", "build", "wikitext2", "c4", "zero_shot", "cleanup"])
    ap.add_argument("--bundle", default="all", choices=["all", "hs_arc", "wg_piqa"])
    ap.add_argument("--keep_model", action="store_true")
    # backward compat
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()
    if args.dataset:
        # map old flags
        if args.dataset == "all":
            args.phase = "all"
        elif args.dataset in ("wikitext2", "c4"):
            args.phase = args.dataset
        elif args.dataset == "zero_shot":
            args.phase = "zero_shot"

    art = artifact_dir(f"matched_controls/{args.control}", link_under=HERE / "artifacts" / f"MC_{args.control}")
    sched_path = art / "MATCHED_CONTROL_SCHEDULE.json"
    if not sched_path.exists():
        subprocess.check_call([
            PYTHON_BIN, str(HERE / "runners" / "matched_control_build.py"),
            "--control", args.control, "--anchor", "moderate",
        ])
    sched = json.loads(sched_path.read_text())
    pairs = _pairs_from_schedule(sched)
    kv = realized_kv_bytes(pairs)
    fdp_kv = realized_kv_bytes(load_low_rank(FDP_MODEL / "low_rank_config.json"))
    method = f"MC_{args.control}"
    MC_ROOT.mkdir(parents=True, exist_ok=True)
    model_dir = MC_ROOT / args.control / "final_model"
    work = art / "compress_work"

    if args.phase in ("build", "all"):
        if not _model_ready(model_dir):
            _compress(pairs, model_dir, work)
        atomic_write_json(art / "EVAL_MODEL_MANIFEST.json", {
            "control": args.control,
            "generated_at": now_iso(),
            "model_dir": str(model_dir),
            "basis_src": str(FDP_BASIS),
            "kv": kv,
            "fdp_kv": fdp_kv,
            "byte_gap_vs_fdp": kv["realized_kv_bytes"] - fdp_kv["realized_kv_bytes"],
            "rel_gap": (kv["realized_kv_bytes"] - fdp_kv["realized_kv_bytes"]) / max(fdp_kv["realized_kv_bytes"], 1),
            "build": "truncate_fdp_basis_then_compress_only",
        })
        if args.phase == "build":
            print(json.dumps({"ok": True, "phase": "build", "control": args.control, "kv": kv}, ensure_ascii=False))
            return

    if args.phase != "cleanup" and not _model_ready(model_dir):
        raise SystemExit(f"model not ready for phase={args.phase}: {model_dir}")

    results = {"control": args.control, "kv": kv, "phase": args.phase}
    if args.phase in ("wikitext2", "all"):
        results["wikitext2"] = _run_ppl(model_dir, art / "wikitext2_ppl", "wikitext2", method)
    if args.phase in ("c4", "all"):
        results["c4"] = _run_ppl(model_dir, art / "c4_ppl", "c4", method)
    if args.phase in ("zero_shot", "all"):
        if args.bundle in ("all", "hs_arc"):
            results["hs_arc"] = _run_zs(
                model_dir, art / "hs_arc",
                "hellaswag_localjsonl,arc_challenge_localjsonl,arc_easy_localjsonl", method,
            )
        if args.bundle in ("all", "wg_piqa"):
            results["wg_piqa"] = _run_zs(
                model_dir, art / "wg_piqa",
                "winogrande_localjsonl,piqa_localjsonl", method,
            )

    if args.phase == "cleanup" or (args.phase == "all" and not args.keep_model):
        _cleanup(model_dir, art)
        results["cleaned"] = True

    # refresh summary if pieces exist
    summary = {"generated_at": now_iso(), "control": args.control, "kv": kv}
    for name, path in [
        ("wikitext2", art / "wikitext2_ppl" / "ppl_aggregate.json"),
        ("c4", art / "c4_ppl" / "ppl_aggregate.json"),
        ("hs_arc", art / "hs_arc" / "zero_shot_aggregate.json"),
        ("wg_piqa", art / "wg_piqa" / "zero_shot_aggregate.json"),
    ]:
        if path.exists():
            summary[name] = json.loads(path.read_text())
    atomic_write_json(art / "CONTROL_EVAL_SUMMARY.json", summary)
    print(json.dumps({"ok": True, **{k: results.get(k) for k in ("phase", "control", "cleaned")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
