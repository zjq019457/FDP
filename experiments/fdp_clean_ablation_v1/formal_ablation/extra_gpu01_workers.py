#!/usr/bin/env python3
"""
EXTRA WORKERS: Ultra No-Anchor states on GPU 0 and GPU 1
Runs immediately alongside fdp_formal_v4 session.
"""
import os, json, math, shutil, subprocess
from pathlib import Path
from datetime import datetime, timezone
import concurrent.futures

WORKSPACE = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1")
FORMAL_DIR = WORKSPACE / "formal_ablation"
PER_WIN_DIR = FORMAL_DIR / "CORE_FORMAL_PER_WINDOW"
SCRATCH_DIR = Path("/diskE/jiaqizhang/core_models_scratch")
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

def mkenv(gpu_id, extra_path=""):
    e = os.environ.copy()
    e["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    parts = [str(REPO / "EigenAttn")]
    if extra_path:
        parts.append(str(REPO / extra_path))
    parts.append(e.get("PYTHONPATH", ""))
    e["PYTHONPATH"] = ":".join(parts)
    return e

def run_state(sha, sinfo, gpu_id, worker_id):
    state_dir = PER_WIN_DIR / sha
    state_dir.mkdir(parents=True, exist_ok=True)
    done_file = state_dir / "STATE_EVAL_DONE.json"
    if done_file.exists():
        print(f"[{utc()}][EW{worker_id}] {sha[:12]} already done, skip.")
        return True

    model_dir = SCRATCH_DIR / f"model_ew{worker_id}_{sha[:12]}"
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{utc()}][EW{worker_id}][GPU {gpu_id}] Starting {sha[:12]} (Tier={sinfo['tier']}, Rank={sinfo['tot_rank']})...")

    try:
        # 1. Compress
        if not list(model_dir.glob("model-*.safetensors")):
            cmd = [PY_BIN, "-u", str(MAIN_SCRIPT),
                   "--model", MODEL_BASE, "--net", "Llama-3-8b",
                   "--dtype", "bf16", "--error_budget", "0.06",
                   "--avg_dim", "4096", "--seqlen", "2048",
                   "--nsamples", "64", "--seed", "2",
                   "--calib_dataset", "c4", "--compress_only",
                   "--attn_implementation", "eager",
                   "--eigen_basis_path", sinfo["basis_pt"],
                   "--save_dir", str(model_dir),
                   "--output_dir", str(model_dir),
                   "--cache_dir", str(CACHE_DIR)]
            print(f"[{utc()}][EW{worker_id}][GPU {gpu_id}] Materializing on /diskE...")
            p = subprocess.run(cmd, cwd=str(REPO), env=mkenv(gpu_id),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                print(f"[EW{worker_id}] COMPRESS FAILED:\n{p.stdout.decode('utf-8',errors='ignore')[-2000:]}")
                return False
            if not list(model_dir.glob("model-*.safetensors")):
                print(f"[EW{worker_id}] No safetensors after compress!")
                return False

        # 2. Wiki PPL
        wiki_out = state_dir / "wikitext"
        wiki_out.mkdir(exist_ok=True)
        wiki_win = state_dir / "wiki_windows.jsonl"
        if not wiki_win.exists():
            cmd = [PY_BIN, "-u", str(PPL_SCRIPT),
                   "--dataset", "wikitext2", "--model", str(model_dir),
                   "--tok", MODEL_BASE, "--out_dir", str(wiki_out),
                   "--seqlen", "2048", "--dtype", "bf16", "--device", "cuda:0",
                   "--method", sinfo["method"], "--net", "llama", "--seed", "2",
                   "--wikitext2_dir", WIKI_DIR,
                   "--evaluation_protocol_hash",
                   "da93ade0391d1886941e896092c016d08918defcb8c68eab7e315758209f0644"]
            print(f"[{utc()}][EW{worker_id}][GPU {gpu_id}] Running Wiki PPL...")
            p = subprocess.run(cmd, cwd=str(REPO), env=mkenv(gpu_id, "rebuttal_legacy_reproduction"),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                print(f"[EW{worker_id}] WIKI FAILED:\n{p.stdout.decode('utf-8',errors='ignore')[-2000:]}")
                return False
            out = wiki_out / "ppl_windows.jsonl"
            if out.exists(): out.replace(wiki_win)
            else:
                print(f"[EW{worker_id}] No ppl_windows.jsonl after wiki eval!")
                return False

        # 3. C4 PPL
        c4_out = state_dir / "c4"
        c4_out.mkdir(exist_ok=True)
        c4_win = state_dir / "c4_windows.jsonl"
        if not c4_win.exists():
            cmd = [PY_BIN, "-u", str(PPL_SCRIPT),
                   "--dataset", "c4", "--model", str(model_dir),
                   "--tok", MODEL_BASE, "--c4_texts", C4_TEXTS,
                   "--out_dir", str(c4_out), "--seqlen", "2048",
                   "--dtype", "bf16", "--device", "cuda:0",
                   "--method", sinfo["method"], "--net", "llama", "--seed", "2",
                   "--evaluation_protocol_hash",
                   "da93ade0391d1886941e896092c016d08918defcb8c68eab7e315758209f0644"]
            print(f"[{utc()}][EW{worker_id}][GPU {gpu_id}] Running C4 PPL...")
            p = subprocess.run(cmd, cwd=str(REPO), env=mkenv(gpu_id, "rebuttal_legacy_reproduction"),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                print(f"[EW{worker_id}] C4 FAILED:\n{p.stdout.decode('utf-8',errors='ignore')[-2000:]}")
                return False
            out = c4_out / "ppl_windows.jsonl"
            if out.exists(): out.replace(c4_win)
            else:
                print(f"[EW{worker_id}] No ppl_windows.jsonl after c4 eval!")
                return False

        # 4. Zero-Shot
        zs_out = state_dir / "zeroshot"
        zs_out.mkdir(exist_ok=True)
        zs_agg = zs_out / "zero_shot_aggregate.json"
        if not zs_agg.exists():
            cmd = [PY_BIN, "-u", str(ZS_SCRIPT),
                   "--model", str(model_dir), "--tokenizer_model", MODEL_BASE,
                   "--task_yaml_dir", ZS_YAML_DIR,
                   "--tasks", "hellaswag_localjsonl,arc_challenge_localjsonl,arc_easy_localjsonl,winogrande_localjsonl,piqa_localjsonl",
                   "--out_dir", str(zs_out), "--net", "llama-3-8b",
                   "--dtype", "bf16", "--method", sinfo["method"],
                   "--cache_dir", str(CACHE_DIR), "--attn_implementation", "eager"]
            print(f"[{utc()}][EW{worker_id}][GPU {gpu_id}] Running 5-Task ZS...")
            p = subprocess.run(cmd, cwd=str(REPO), env=mkenv(gpu_id, "rebuttal_legacy_reproduction"),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                print(f"[EW{worker_id}] ZS FAILED:\n{p.stdout.decode('utf-8',errors='ignore')[-2000:]}")
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
        rec = {"sha": sha, "status": "DONE_VALID_EVALUATED",
               "tier": sinfo["tier"], "tot_rank": sinfo["tot_rank"], "b_live": sinfo["b_live"],
               "wiki_ppl": w_ppl, "wiki_nll": w_nll, "wiki_windows": len(w_lines),
               "c4_ppl": c_ppl, "c4_nll": c_nll, "c4_windows": len(c_lines),
               "zero_shot_tasks": tasks_res, "zero_shot_macro": z_macro,
               "scorer_authority": "lm_eval_primary", "completed_at": utc()}
        with open(done_file, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[{utc()}][EW{worker_id}] DONE {sha[:12]} => Wiki={w_ppl:.4f} C4={c_ppl:.4f} ZS={z_macro:.4f}")
        return True
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)
        print(f"[{utc()}][EW{worker_id}] Cleaned up {model_dir}")

if __name__ == "__main__":
    # The 2 Ultra No-Anchor states not yet started by v4
    ultra_basis = "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/pass2_historical_fdp120/eigen_basis_llama3_8b_wikitext2_frag_eb120_perlayer_eb.pt"
    
    with open(FORMAL_DIR / "CORE_RESOURCE_AUTHORITY.json") as f:
        core_rows = json.load(f)
    
    target_shas = {"6827686f8a78", "b09aa779e03b"}
    states = {}
    for r in core_rows:
        for kind in ["natural", "matched"]:
            sha = r[f"{kind}_rank_sha"]
            if sha[:12] in target_shas or sha in target_shas:
                if (PER_WIN_DIR / sha / "STATE_EVAL_DONE.json").exists():
                    print(f"[{utc()}] {sha[:12]} already done, skip.")
                    continue
                states[sha] = {
                    "tier": r["tier"], "method": r["method"], "sha": sha,
                    "tot_rank": r[f"{kind}_tot_rank"], "b_live": r[f"{kind}_b_live"],
                    "basis_pt": ultra_basis
                }
    
    if not states:
        print(f"[{utc()}] Both ultra no-anchor states already done!")
    else:
        print(f"[{utc()}] Launching {len(states)} extra workers on GPU 0 and GPU 1...")
        gpu_map = [0, 1]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futs = {executor.submit(run_state, sha, sinfo, gpu_map[i % 2], i): sha
                    for i, (sha, sinfo) in enumerate(states.items())}
            for fut in concurrent.futures.as_completed(futs):
                sha = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"[{utc()}] EXCEPTION {sha[:12]}: {e}")
    print(f"[{utc()}] Extra workers finished.")
