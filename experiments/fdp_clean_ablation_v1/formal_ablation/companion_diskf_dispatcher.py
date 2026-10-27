#!/usr/bin/env python3
"""
COMPANION DISPATCHER (DISK F SCRATCH)
=====================================
Leverages 52GB on /diskF to run additional concurrent jobs.
Targets states from LOO_CONTROLS_AUTHORITY.json in reverse order
so it works inward from the other end of the queue, avoiding overlap.
Scratch dir: /diskF/jiaqizhang/core_models_scratch_diskf
GPU Pool: GPU 1 and GPU 2 (sharing capacity safely)
"""
import os, sys, json, math, shutil, subprocess, time
from pathlib import Path
from datetime import datetime, timezone
import concurrent.futures

WORKSPACE   = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1")
FORMAL_DIR  = WORKSPACE / "formal_ablation"
PER_WIN_DIR = FORMAL_DIR / "CORE_FORMAL_PER_WINDOW"
SCRATCH_DIR = Path("/diskF/jiaqizhang/core_models_scratch_diskf")
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

MODEL_BASE  = "/diskD/jiaqizhang/Llama-3"
PY_BIN      = "/home/bai-project/anaconda3/envs/eigen/bin/python"
REPO        = Path("/home/bai-project/jiaqizhang/EigenAttn")
PPL_SCRIPT  = REPO / "rebuttal_legacy_reproduction/runners/ppl_instrumented.py"
ZS_SCRIPT   = REPO / "rebuttal_legacy_reproduction/runners/zero_shot_instrumented.py"
MAIN_SCRIPT = REPO / "EigenAttn/main_eigen_attn.py"
WIKI_DIR    = "/home/bai-project/jiaqizhang/wikitext2"
C4_TEXTS    = "/diskF/jiaqizhang/fdp_eb_orth_paper_v1_results/offline_calib/c4/c4_ppl_selected_texts.jsonl"
ZS_YAML_DIR = "/diskF/jiaqizhang/a1804298693/offline_datasets"
CACHE_DIR   = REPO / "EigenAttn/HF_cache"

def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def mkenv(gpu_id, extra=""):
    e = os.environ.copy()
    e["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    parts = [str(REPO / "EigenAttn")]
    if extra: parts.append(str(REPO / extra))
    parts.append(e.get("PYTHONPATH", ""))
    e["PYTHONPATH"] = ":".join(parts)
    return e

def run_single_state(sha, sinfo, gpu_id, worker_id):
    state_dir = PER_WIN_DIR / sha
    state_dir.mkdir(parents=True, exist_ok=True)
    done_file = state_dir / "STATE_EVAL_DONE.json"
    if done_file.exists():
        print(f"[{utc()}][FW{worker_id}] {sha[:12]} already done, skip.")
        return True

    model_dir = SCRATCH_DIR / f"model_fw{worker_id}_{sha[:12]}"
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{utc()}][FW{worker_id}][GPU {gpu_id}] (diskF) Starting {sinfo['tier']} {sinfo['config_name']} ({sha[:12]})...")

    try:
        # 1. Compress
        shards = list(model_dir.glob("model-*.safetensors"))
        if not shards:
            cmd = [
                PY_BIN, "-u", str(MAIN_SCRIPT),
                "--model", MODEL_BASE,
                "--net", "Llama-3-8b",
                "--dtype", "bf16",
                "--error_budget", "0.06",
                "--avg_dim", "4096",
                "--seqlen", "2048",
                "--nsamples", "64",
                "--seed", "2",
                "--calib_dataset", "c4",
                "--compress_only",
                "--attn_implementation", "eager",
                "--eigen_basis_path", sinfo["basis_pt"],
                "--save_dir", str(model_dir),
                "--output_dir", str(model_dir),
                "--cache_dir", str(CACHE_DIR),
            ]
            p = subprocess.run(cmd, cwd=str(REPO), env=mkenv(gpu_id),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                print(f"[{utc()}][FW{worker_id}] COMPRESS FAILED:\n{p.stdout.decode('utf-8', errors='ignore')[-2000:]}")
                return False
            shards = list(model_dir.glob("model-*.safetensors"))
            if not shards:
                return False

        # 2. WikiText-2 PPL
        wiki_out = state_dir / "wikitext"
        wiki_out.mkdir(parents=True, exist_ok=True)
        wiki_win = state_dir / "wiki_windows.jsonl"
        if not wiki_win.exists():
            wiki_cmd = [
                PY_BIN, "-u", str(PPL_SCRIPT),
                "--dataset", "wikitext2",
                "--model", str(model_dir),
                "--tok", MODEL_BASE,
                "--out_dir", str(wiki_out),
                "--seqlen", "2048",
                "--dtype", "bf16",
                "--device", "cuda:0",
                "--method", sinfo["config_name"],
                "--net", "llama",
                "--seed", "2",
                "--wikitext2_dir", WIKI_DIR,
                "--evaluation_protocol_hash",
                "da93ade0391d1886941e896092c016d08918defcb8c68eab7e315758209f0644",
            ]
            p = subprocess.run(wiki_cmd, cwd=str(REPO), env=mkenv(gpu_id, "rebuttal_legacy_reproduction"),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                print(f"[{utc()}][FW{worker_id}] WIKI FAILED:\n{p.stdout.decode('utf-8', errors='ignore')[-2000:]}")
                return False
            out_file = wiki_out / "ppl_windows.jsonl"
            if out_file.exists(): out_file.replace(wiki_win)
            else: return False

        # 3. C4 PPL
        c4_out = state_dir / "c4"
        c4_out.mkdir(parents=True, exist_ok=True)
        c4_win = state_dir / "c4_windows.jsonl"
        if not c4_win.exists():
            c4_cmd = [
                PY_BIN, "-u", str(PPL_SCRIPT),
                "--dataset", "c4",
                "--model", str(model_dir),
                "--tok", MODEL_BASE,
                "--c4_texts", C4_TEXTS,
                "--out_dir", str(c4_out),
                "--seqlen", "2048",
                "--dtype", "bf16",
                "--device", "cuda:0",
                "--method", sinfo["config_name"],
                "--net", "llama",
                "--seed", "2",
                "--evaluation_protocol_hash",
                "da93ade0391d1886941e896092c016d08918defcb8c68eab7e315758209f0644",
            ]
            p = subprocess.run(c4_cmd, cwd=str(REPO), env=mkenv(gpu_id, "rebuttal_legacy_reproduction"),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                print(f"[{utc()}][FW{worker_id}] C4 FAILED:\n{p.stdout.decode('utf-8', errors='ignore')[-2000:]}")
                return False
            out_file = c4_out / "ppl_windows.jsonl"
            if out_file.exists(): out_file.replace(c4_win)
            else: return False

        # 4. Zero-Shot
        zs_out = state_dir / "zeroshot"
        zs_out.mkdir(parents=True, exist_ok=True)
        zs_agg = zs_out / "zero_shot_aggregate.json"
        if not zs_agg.exists():
            zs_cmd = [
                PY_BIN, "-u", str(ZS_SCRIPT),
                "--model", str(model_dir),
                "--tokenizer_model", MODEL_BASE,
                "--task_yaml_dir", ZS_YAML_DIR,
                "--tasks", "hellaswag_localjsonl,arc_challenge_localjsonl,arc_easy_localjsonl,winogrande_localjsonl,piqa_localjsonl",
                "--out_dir", str(zs_out),
                "--net", "llama-3-8b",
                "--dtype", "bf16",
                "--method", sinfo["config_name"],
                "--cache_dir", str(CACHE_DIR),
                "--attn_implementation", "eager",
            ]
            p = subprocess.run(zs_cmd, cwd=str(REPO), env=mkenv(gpu_id, "rebuttal_legacy_reproduction"),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                print(f"[{utc()}][FW{worker_id}] ZS FAILED:\n{p.stdout.decode('utf-8', errors='ignore')[-2000:]}")
                return False

        # 5. Record
        w_lines = [json.loads(l) for l in wiki_win.read_text().splitlines() if l.strip()]
        w_nll = sum(float(l["total_nll"]) for l in w_lines) / sum(int(l["token_count"]) for l in w_lines)
        w_ppl = math.exp(w_nll)

        c_lines = [json.loads(l) for l in c4_win.read_text().splitlines() if l.strip()]
        c_nll = sum(float(l["total_nll"]) for l in c_lines) / sum(int(l["token_count"]) for l in c_lines)
        c_ppl = math.exp(c_nll)

        z_dict = json.loads(zs_agg.read_text())
        tasks_res = {tn: td.get("lm_eval_primary", td.get("accuracy_from_correct"))
                     for tn, td in z_dict.get("tasks", {}).items()}
        z_macro = sum(tasks_res.values()) / len(tasks_res) if tasks_res else None

        rec = {
            "sha": sha,
            "status": "DONE_VALID_EVALUATED",
            "tier": sinfo["tier"],
            "group": sinfo["group"],
            "config_name": sinfo["config_name"],
            "tot_rank": sinfo["tot_rank"],
            "b_live": sinfo["b_live"],
            "gap_pct": sinfo["gap_pct"],
            "wiki_ppl": w_ppl,
            "wiki_nll": w_nll,
            "wiki_windows": len(w_lines),
            "c4_ppl": c_ppl,
            "c4_nll": c_nll,
            "c4_windows": len(c_lines),
            "zero_shot_tasks": tasks_res,
            "zero_shot_macro": z_macro,
            "scorer_authority": "lm_eval_primary",
            "completed_at": utc(),
        }
        with open(done_file, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[{utc()}][FW{worker_id}] DONE {sinfo['tier']} {sinfo['config_name']} => Wiki={w_ppl:.4f} C4={c_ppl:.4f} ZS={z_macro:.4f}")
        return True

    finally:
        shutil.rmtree(model_dir, ignore_errors=True)
        print(f"[{utc()}][FW{worker_id}] Cleaned up {model_dir}")


if __name__ == "__main__":
    with open(FORMAL_DIR / "LOO_CONTROLS_AUTHORITY.json") as f:
        all_records = json.load(f)

    # Reverse order so we pull Aggressive / Control jobs from the tail
    states_to_run = {}
    for r in reversed(all_records):
        sha = r["sha"]
        if (PER_WIN_DIR / sha / "STATE_EVAL_DONE.json").exists():
            continue
        if sha not in states_to_run:
            states_to_run[sha] = r

    items = list(states_to_run.items())
    print(f"[{utc()}] (Companion diskF) {len(items)} states remaining in reverse queue.")

    if not items:
        print(f"[{utc()}] All done.")
        sys.exit(0)

    # Run 1 worker on GPU 1 (has 73GB free VRAM) with diskF scratch
    gpu_pool = [1]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futs = {executor.submit(run_single_state, sha, sinfo, gpu_pool[i % len(gpu_pool)], i): sha
                for i, (sha, sinfo) in enumerate(items)}
        for fut in concurrent.futures.as_completed(futs):
            sha = futs[fut]
            try: fut.result()
            except Exception as e: print(f"Exception: {e}")

    print("Companion finished.")
