#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import traceback
from pathlib import Path

REPO = Path("/home/bai-project/jiaqizhang/EigenAttn")
EIGEN = REPO / "EigenAttn"
PY = "/home/bai-project/anaconda3/envs/eigen/bin/python"
MODEL = Path("/diskD/jiaqizhang/Llama-3")
WIKI = Path("/home/bai-project/jiaqizhang/wikitext2")
C4 = Path("/diskD/jiaqizhang/llama3_8b_eb045_eval_minimal-20260502T170035Z-3-001/llama3_8b_eb045_eval_minimal/data/c4_ppl_selected_texts.jsonl")
BUILDER = EIGEN / "tools/build_eigen_basis_llama_error_budget_offline.py"
MAIN = EIGEN / "main_eigen_attn.py"
STEP1 = EIGEN / "step1_diagnose_llama.py"
RUNNER = REPO / "rebuttal_legacy_reproduction/runners/ppl_instrumented.py"
LOCK = REPO / "rebuttal_legacy_reproduction/LEGACY_EVALUATION_PROTOCOL_LOCK.json"
CAMPAIGN = Path("/diskF/jiaqizhang/fdp125_final_campaign_v1")
SCRATCH_ROOT = Path("/diskE/jiaqizhang/fdp125_final_campaign_v1_scratch")
FULL_BYTES = 268_435_456
BASE_E = {"light": 0.045, "medium": 0.060, "aggressive": 0.090}
TARGET_B = {"light": 218_988_544, "medium": 198_377_472, "aggressive": 158_105_600}


def now() -> str:
    return dt.datetime.now().astimezone().isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_sha(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def run(name: str, cmd: list[str], cwd: Path, env: dict[str, str], logs: Path) -> dict:
    logs.mkdir(parents=True, exist_ok=True)
    out = logs / f"{name}.log"
    started = now()
    with out.open("w") as f:
        cp = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT)
    if cp.returncode:
        raise RuntimeError(f"{name} failed rc={cp.returncode}; {out}")
    return {"name": name, "command": cmd, "started_at": started, "completed_at": now(), "log": str(out)}


def derive_eb(step1: Path, e: float, weights: tuple[float, ...]) -> list[float]:
    import sys

    sys.path.insert(0, str(EIGEN))
    from realloc_rank_from_fragility import fragility_vector, minmax

    per_layer = json.loads(step1.read_text())["per_layer"]
    layers = sorted(int(x) for x in per_layer)
    raw, _ = fragility_vector(per_layer, layers, *weights, False)
    st = minmax(raw)
    eps = [e / (1.0 + 1.2 * x) for x in st]
    scale = e / (sum(eps) / len(eps))
    eps = [x * scale for x in eps]
    eps[0] = math.exp(0.75 * math.log(e) + 0.25 * math.log(eps[0]))
    return eps


def basis_and_model(method: str, tier: str, eb: list[float], gpu: int, work: Path, logs: Path, commands: list[dict]) -> tuple[Path, dict]:
    e = BASE_E[tier]
    basis_dir = work / "basis"
    model_dir = work / "final_model"
    basis_dir.mkdir(parents=True)
    eb_path = work / "per_layer_eb.json"
    atomic_json(eb_path, {"meta": {"method": method, "tier": tier}, "error_budget_by_layer": eb})
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": f"{EIGEN}:{REPO}:{env.get('PYTHONPATH','')}", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    prefix = f"fdp125_llama_{tier}_{method}"
    bcmd = [PY, "-u", str(BUILDER), "--model", str(MODEL), "--data_dir", str(WIKI), "--out_dir", str(basis_dir), "--out_prefix", prefix, "--per_layer_eb_json", str(eb_path), "--freeze_layer0_eb", "--layer0_eb", str(e), "--nsamples", "512", "--seqlen", "2048", "--dtype", "bf16", "--attn_implementation", "eager", "--device", "cuda:0", "--seed", "2"]
    commands.append(run(f"{method}_basis", bcmd, EIGEN, env, logs))
    basis = basis_dir / f"{prefix}_perlayer_eb.pt"
    if not basis.is_file():
        raise FileNotFoundError(basis)
    basis_sha = sha(basis)
    mcmd = [PY, "-u", str(MAIN), "--model", str(MODEL), "--net", "Llama-3-8b", "--dtype", "bf16", "--error_budget", str(e), "--avg_dim", "4096", "--seqlen", "2048", "--nsamples", "64", "--seed", "2", "--calib_dataset", "c4", "--compress_only", "--attn_implementation", "eager", "--eigen_basis_path", str(basis), "--save_dir", str(model_dir), "--output_dir", str(work), "--cache_dir", str(EIGEN / "HF_cache")]
    commands.append(run(f"{method}_materialize", mcmd, EIGEN, env, logs))
    rank_path = model_dir / "low_rank_config.json"
    ranks = json.loads(rank_path.read_text())
    meta = {"eb_vector": eb, "eb_sha": json_sha([round(x, 8) for x in eb]), "basis_sha": basis_sha, "ranks": ranks, "rank_sha": json_sha(ranks), "B_live": sum(sum(x) for x in ranks) * 4096}
    shutil.rmtree(basis_dir)
    return model_dir, meta


def evaluate(method: str, tier: str, model_dir: Path, gpu: int, result: Path, logs: Path, commands: list[dict]) -> dict:
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": f"{EIGEN}:{REPO}:{env.get('PYTHONPATH','')}", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    proto = sha(LOCK)
    wiki = result / "wiki"; c4 = result / "c4"
    wcmd = [PY, "-u", str(RUNNER), "--dataset", "wikitext2", "--model", str(model_dir), "--tok", str(MODEL), "--out_dir", str(wiki), "--wikitext2_dir", str(WIKI), "--dtype", "bf16", "--method", f"FDP125_LLAMA_{tier}_{method}", "--evaluation_protocol_hash", proto]
    ccmd = [PY, "-u", str(RUNNER), "--dataset", "c4", "--model", str(model_dir), "--tok", str(MODEL), "--c4_texts", str(C4), "--out_dir", str(c4), "--dtype", "bf16", "--method", f"FDP125_LLAMA_{tier}_{method}", "--evaluation_protocol_hash", proto]
    commands.append(run(f"{method}_wiki", wcmd, REPO, env, logs))
    commands.append(run(f"{method}_c4", ccmd, REPO, env, logs))
    wa = json.loads((wiki / "ppl_aggregate.json").read_text()); ca = json.loads((c4 / "ppl_aggregate.json").read_text())
    if (wa["used_windows"], wa["skipped_windows"]) != (141, 0) or (ca["used_windows"], ca["skipped_windows"]) != (256, 0):
        raise RuntimeError("window integrity failure")
    return {"wiki": wa, "c4": ca}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--tier", choices=BASE_E, required=True); ap.add_argument("--gpu", type=int, required=True); args = ap.parse_args()
    tier, gpu = args.tier, args.gpu
    scratch = SCRATCH_ROOT / f"llama_{tier}"
    out = CAMPAIGN / "results/llama" / tier
    if (out / "CORE_DONE.json").is_file():
        return
    if shutil.disk_usage("/diskE").free < 47 * 1024**3:
        atomic_json(out / "STATUS.json", {"status": "DISPATCH_PAUSED_LOW_DISK", "at": now()}); return
    if scratch.exists(): shutil.rmtree(scratch)
    scratch.mkdir(parents=True); out.mkdir(parents=True, exist_ok=True)
    commands: list[dict] = []
    atomic_json(out / "STATUS.json", {"status": "RUNNING", "tier": tier, "gpu": gpu, "at": now()})
    try:
        uniform_work = scratch / "uniform"
        uniform_model, uniform_meta = basis_and_model("uniform", tier, [BASE_E[tier]] * 32, gpu, uniform_work, scratch / "logs", commands)
        step1 = scratch / "step1.json"
        env = dict(os.environ); env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": f"{EIGEN}:{REPO}:{env.get('PYTHONPATH','')}"})
        scmd = [PY, "-u", str(STEP1), "--base", str(MODEL), "--cmp", str(uniform_model), "--tok", str(MODEL), "--dtype", "bf16", "--seqlen", "512", "--num_seqs", "2", "--out", str(step1), "--gpu", "0", "--attn_implementation", "eager"]
        commands.append(run("step1", scmd, EIGEN, env, scratch / "logs"))
        shutil.copy2(step1, out / "step1.json")
        schedules = {"parent": derive_eb(step1, BASE_E[tier], (1, 1, 1, 1, 1)), "fdp125": derive_eb(step1, BASE_E[tier], (1, 1, 0, 0, 1))}
        methods = {}
        for method in ["uniform", "parent", "fdp125"]:
            result = out / method; result.mkdir(parents=True, exist_ok=True)
            if method == "uniform":
                model_dir, meta = uniform_model, uniform_meta
            else:
                model_dir, meta = basis_and_model(method, tier, schedules[method], gpu, scratch / method, scratch / "logs", commands)
            quality = evaluate(method, tier, model_dir, gpu, result, scratch / "logs", commands)
            atomic_json(result / "per_layer_eb.json", {"error_budget_by_layer": meta["eb_vector"]})
            atomic_json(result / "low_rank_config.json", meta["ranks"])
            prov = {"status": "PASS", "model": "llama3_8b", "regime": tier, "method": method, "base_e": BASE_E[tier], "lambda": 1.2 if method != "uniform" else 0, "rho": 0.75 if method != "uniform" else None, "components": ["d1", "d2", "d3", "d4", "d5"] if method == "parent" else (["d1", "d2", "d5"] if method == "fdp125" else []), **meta, "full_resource_bytes": FULL_BYTES, "retained_ratio": meta["B_live"] / FULL_BYTES, "target_B_live": TARGET_B[tier], "resource_gap_pct": (meta["B_live"] - TARGET_B[tier]) / TARGET_B[tier] * 100, "quality": quality, "commands": commands, "completed_at": now()}
            atomic_json(result / "provenance.json", prov)
            methods[method] = {"B_live": meta["B_live"], "wiki_ppl": quality["wiki"]["ppl"], "c4_ppl": quality["c4"]["ppl"], "rank_sha": meta["rank_sha"], "basis_sha": meta["basis_sha"]}
            shutil.rmtree(model_dir)
        atomic_json(out / "CORE_DONE.json", {"status": "PASS", "model": "llama3_8b", "tier": tier, "methods": methods, "completed_at": now()})
        atomic_json(out / "STATUS.json", {"status": "DONE", "at": now()})
        shutil.rmtree(scratch)
    except Exception as exc:
        atomic_json(out / "FAILED_ENGINEERING.json", {"status": "FAILED_ENGINEERING", "error": repr(exc), "traceback": traceback.format_exc(), "commands": commands, "at": now()})
        if (scratch / "logs").exists():
            if (out / "failure_logs").exists(): shutil.rmtree(out / "failure_logs")
            shutil.copytree(scratch / "logs", out / "failure_logs")
        shutil.rmtree(scratch, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
