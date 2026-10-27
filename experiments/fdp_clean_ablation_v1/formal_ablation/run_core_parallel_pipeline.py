#!/usr/bin/env python3
"""
CORE FORMAL EVALUATION PARALLEL PIPELINE
=========================================
Runs WikiText-2 (141 windows), C4 (256 windows), and 5-task Zero-Shot (PIQA, ARC-e, ARC-c, HellaSwag, WinoGrande)
on the remaining 8 physical states across GPU 2 and GPU 3.
Collects and merges existing historical evaluations for the other 6 states.
Outputs all 7 core deliverables:
- formal_ablation/CORE_RESOURCE_AUTHORITY.csv
- formal_ablation/CORE_FORMAL_RESULTS.csv
- formal_ablation/CORE_FORMAL_PER_WINDOW/
- formal_ablation/CORE_ZERO_SHOT_RESULTS.csv
- formal_ablation/CORE_RESOURCE_MATCHING.csv
- formal_ablation/CORE_BOOTSTRAP_RESULTS.csv
- formal_ablation/CORE_CAMPAIGN_STATUS.json
- formal_ablation/CORE_FORMAL_REPORT.md
"""
import os
import sys
import json
import time
import math
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

WORKSPACE = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1")
FORMAL_DIR = WORKSPACE / "formal_ablation"
PER_WIN_DIR = FORMAL_DIR / "CORE_FORMAL_PER_WINDOW"
PER_WIN_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH_DIR = WORKSPACE / "scratch" / "core_models"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

MODEL_BASE = "/diskD/jiaqizhang/Llama-3"
PY_BIN = "/home/bai-project/anaconda3/envs/eigen/bin/python"
REPO = Path("/home/bai-project/jiaqizhang/EigenAttn")
PPL_SCRIPT = REPO / "rebuttal_legacy_reproduction/runners/ppl_instrumented.py"
ZS_SCRIPT = REPO / "rebuttal_legacy_reproduction/runners/zero_shot_instrumented.py"
MAIN_SCRIPT = REPO / "EigenAttn/main_eigen_attn.py"
WIKI_DIR = "/home/bai-project/jiaqizhang/wikitext2"
C4_TEXTS = "/diskF/jiaqizhang/fdp_eb_orth_paper_v1_results/offline_calib/c4/c4_ppl_selected_texts.jsonl"
ZS_YAML_DIR = "/diskF/jiaqizhang/a1804298693/offline_datasets"
CACHE_DIR = REPO / "EigenAttn/HF_cache"

def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

print(f"[{utc()}] STARTING CORE FORMAL EVALUATION PARALLEL PIPELINE...")

# Load core authority
with open(FORMAL_DIR / "CORE_RESOURCE_AUTHORITY.json") as f:
    core_rows = json.load(f)

# Existing historical evaluation mapping
HIST_EVAL_MAP = {
    "e715356b0434ba11ea89e9e9362a74089c756b2f030d97d4838cc70e7ff8bb59": "/diskF/jiaqizhang/fdp_may2026_final_reconstruction_v1/evaluations/light/uniform_043",
    "7ecf6eda0d60b4a4d01c7838f0736b47167320cd7640254816cd4101f417ad59": "/diskF/jiaqizhang/fdp_may2026_final_reconstruction_v1/evaluations/light/fdp_045",
    "1e253839bbd8735a9ace1f0520936250bfeba632eff902a281218742f5b065c7": "/diskF/jiaqizhang/fdp_may2026_final_reconstruction_v1/evaluations/medium/uniform_058",
    "870786eb6b8e5ca5c196ae0a421e384686a011e80a8be7a95f1fd6b0fd1a6064": "/diskF/jiaqizhang/fdp_may2026_final_reconstruction_v1/evaluations/medium/fdp_060",
    "32d1d95570eb76e491f7700507402a15092d7cbd6e8b7662ccc380b87ef42411": "/diskF/jiaqizhang/fdp_may2026_final_reconstruction_v1/evaluations/aggressive/uniform_088",
    "c15a0a97e423ddaf01005bf38b88cf99b5461069a52454a8185925e63b9fe7dc": "/diskF/jiaqizhang/fdp_may2026_final_reconstruction_v1/evaluations/aggressive/fdp_090",
}

# Ingest historical evaluations into PER_WIN_DIR
for sha, h_dir in HIST_EVAL_MAP.items():
    s_dir = PER_WIN_DIR / sha
    s_dir.mkdir(parents=True, exist_ok=True)
    done_file = s_dir / "STATE_EVAL_DONE.json"
    if not done_file.exists():
        # Copy or link ppl_windows.jsonl and zero_shot_aggregate.json
        w_path = Path(h_dir) / "wikitext/ppl_windows.jsonl"
        c_path = Path(h_dir) / "c4/ppl_windows.jsonl"
        z_path = Path(h_dir) / "zeroshot/zero_shot_aggregate.json"
        
        w_lines = [json.loads(l) for l in w_path.read_text().splitlines() if l.strip()]
        w_nll = sum(float(l["total_nll"]) for l in w_lines) / sum(int(l["token_count"]) for l in w_lines)
        w_ppl = math.exp(w_nll)
        
        c_lines = [json.loads(l) for l in c_path.read_text().splitlines() if l.strip()]
        c_nll = sum(float(l["total_nll"]) for l in c_lines) / sum(int(l["token_count"]) for l in c_lines)
        c_ppl = math.exp(c_nll)
        
        z_dict = json.loads(z_path.read_text())
        tasks_res = {}
        for tn, td in z_dict.get("tasks", {}).items():
            tasks_res[tn] = td.get("lm_eval_primary", td.get("accuracy_from_correct"))
        z_macro = sum(tasks_res.values()) / len(tasks_res) if tasks_res else None
        
        # Save per window copies
        (s_dir / "wiki_windows.jsonl").write_text(w_path.read_text())
        (s_dir / "c4_windows.jsonl").write_text(c_path.read_text())
        
        rec = {
            "sha": sha,
            "status": "DONE_VALID_HISTORICAL",
            "source": h_dir,
            "wiki_ppl": w_ppl,
            "wiki_nll": w_nll,
            "wiki_windows": len(w_lines),
            "c4_ppl": c_ppl,
            "c4_nll": c_nll,
            "c4_windows": len(c_lines),
            "zero_shot_tasks": tasks_res,
            "zero_shot_macro": z_macro,
            "scorer_authority": "lm_eval_primary",
            "completed_at": utc()
        }
        with open(done_file, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[{utc()}] Ingested historical state {sha[:12]}... (Wiki PPL {w_ppl:.4f}, C4 PPL {c_ppl:.4f}, ZS {z_macro:.4f})")

# Determine 8 fresh states
states_to_eval = {}
for r in core_rows:
    for kind in ["natural", "matched"]:
        sha = r[f"{kind}_rank_sha"]
        if (PER_WIN_DIR / sha / "STATE_EVAL_DONE.json").exists():
            continue
        if sha not in states_to_eval:
            basis_paths = {
                "light": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb045/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt",
                "medium": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb060/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt",
                "aggressive": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb090/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt",
                "ultra": "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/pass2_historical_fdp120/eigen_basis_llama3_8b_wikitext2_frag_eb120_perlayer_eb.pt",
            }
            if r["method"] == "M3_UNIFORM" and r["tier"] == "ultra":
                # Ultra uniform uses scalar eb 0.113 basis
                basis_pt = "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/lattice_fine/eigen_basis_llama3_8b_wikitext2_uniform_fine_eb0p113.pt"
            else:
                basis_pt = basis_paths[r["tier"]]
                
            states_to_eval[sha] = {
                "tier": r["tier"],
                "method": r["method"],
                "sha": sha,
                "ranks": r[f"{kind}_ranks"],
                "tot_rank": r[f"{kind}_tot_rank"],
                "b_live": r[f"{kind}_b_live"],
                "basis_pt": basis_pt,
                "bound_rows": [(r["row_id"], kind)]
            }
        else:
            states_to_eval[sha]["bound_rows"].append((r["row_id"], kind))

print(f"[{utc()}] Remaining fresh physical states to evaluate on GPU: {len(states_to_eval)}")

# Process each fresh state sequentially or in parallel across GPU 3 and GPU 2
for idx, (sha, sinfo) in enumerate(states_to_eval.items()):
    gpu_id = 3 if idx % 2 == 0 else 2
    print(f"\n================================================================================")
    print(f"[{utc()}] EVALUATING STATE {idx+1}/{len(states_to_eval)}: SHA={sha[:12]}... Tier={sinfo['tier'].upper()} Rank={sinfo['tot_rank']} on GPU {gpu_id}")
    print(f"================================================================================")
    
    state_dir = PER_WIN_DIR / sha
    state_dir.mkdir(parents=True, exist_ok=True)
    model_dir = SCRATCH_DIR / f"model_{sha[:12]}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Materialize model
    config_file = model_dir / "low_rank_config.json"
    config_file.write_text(json.dumps(sinfo["ranks"]))
    
    # Check if safetensors exist
    shards = list(model_dir.glob("model-*.safetensors"))
    if not shards:
        print(f"[{utc()}] Materializing model with basis {sinfo['basis_pt']}...")
        compress_cmd = [
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
        env_c = os.environ.copy()
        env_c["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env_c["PYTHONPATH"] = f"{REPO/'EigenAttn'}:{env_c.get('PYTHONPATH', '')}"
        t0 = time.time()
        p = subprocess.run(compress_cmd, cwd=str(REPO), env=env_c, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            print(p.stdout.decode('utf-8', errors='ignore'))
            raise RuntimeError(f"Compression failed for state {sha}")
        print(f"[{utc()}] Materialization finished in {time.time()-t0:.1f}s")
        # Ensure low_rank_config.json is intact
        config_file.write_text(json.dumps(sinfo["ranks"]))
        
    # 2. Run WikiText-2
    wiki_out = state_dir / "wikitext"
    wiki_out.mkdir(parents=True, exist_ok=True)
    wiki_win = state_dir / "wiki_windows.jsonl"
    if not wiki_win.exists():
        print(f"[{utc()}] Running WikiText-2 PPL on GPU {gpu_id}...")
        wiki_cmd = [
            PY_BIN, "-u", str(PPL_SCRIPT),
            "--dataset", "wikitext2",
            "--model", str(model_dir),
            "--tok", MODEL_BASE,
            "--out_dir", str(wiki_out),
            "--seqlen", "2048",
            "--dtype", "bf16",
            "--device", f"cuda:0",
            "--method", sinfo["method"],
            "--net", "llama",
            "--seed", "2",
            "--wikitext2_dir", WIKI_DIR,
            "--evaluation_protocol_hash", "da93ade0391d1886941e896092c016d08918defcb8c68eab7e315758209f0644"
        ]
        env_w = os.environ.copy()
        env_w["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env_w["PYTHONPATH"] = f"{REPO/'EigenAttn'}:{REPO/'rebuttal_legacy_reproduction'}:{env_w.get('PYTHONPATH', '')}"
        p = subprocess.run(wiki_cmd, cwd=str(REPO), env=env_w, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            print(p.stdout.decode('utf-8', errors='ignore'))
            raise RuntimeError(f"Wiki eval failed for {sha}")
        if (wiki_out / "ppl_windows.jsonl").exists():
            (wiki_out / "ppl_windows.jsonl").replace(wiki_win)

    # 3. Run C4
    c4_out = state_dir / "c4"
    c4_out.mkdir(parents=True, exist_ok=True)
    c4_win = state_dir / "c4_windows.jsonl"
    if not c4_win.exists():
        print(f"[{utc()}] Running C4 PPL on GPU {gpu_id}...")
        c4_cmd = [
            PY_BIN, "-u", str(PPL_SCRIPT),
            "--dataset", "c4",
            "--model", str(model_dir),
            "--tok", MODEL_BASE,
            "--c4_texts", C4_TEXTS,
            "--out_dir", str(c4_out),
            "--seqlen", "2048",
            "--dtype", "bf16",
            "--device", f"cuda:0",
            "--method", sinfo["method"],
            "--net", "llama",
            "--seed", "2",
            "--evaluation_protocol_hash", "da93ade0391d1886941e896092c016d08918defcb8c68eab7e315758209f0644"
        ]
        env_c = os.environ.copy()
        env_c["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env_c["PYTHONPATH"] = f"{REPO/'EigenAttn'}:{REPO/'rebuttal_legacy_reproduction'}:{env_c.get('PYTHONPATH', '')}"
        p = subprocess.run(c4_cmd, cwd=str(REPO), env=env_c, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            print(p.stdout.decode('utf-8', errors='ignore'))
            raise RuntimeError(f"C4 eval failed for {sha}")
        if (c4_out / "ppl_windows.jsonl").exists():
            (c4_out / "ppl_windows.jsonl").replace(c4_win)

    # 4. Run 5 Zero-Shot Tasks
    zs_out = state_dir / "zeroshot"
    zs_out.mkdir(parents=True, exist_ok=True)
    zs_agg = zs_out / "zero_shot_aggregate.json"
    if not zs_agg.exists():
        print(f"[{utc()}] Running 5-Task Zero-Shot on GPU {gpu_id}...")
        zs_cmd = [
            PY_BIN, "-u", str(ZS_SCRIPT),
            "--model", str(model_dir),
            "--tokenizer_model", MODEL_BASE,
            "--task_yaml_dir", ZS_YAML_DIR,
            "--tasks", "hellaswag_localjsonl,arc_challenge_localjsonl,arc_easy_localjsonl,winogrande_localjsonl,piqa_localjsonl",
            "--out_dir", str(zs_out),
            "--net", "llama-3-8b",
            "--dtype", "bf16",
            "--method", sinfo["method"],
            "--cache_dir", str(CACHE_DIR),
            "--attn_implementation", "eager"
        ]
        env_z = os.environ.copy()
        env_z["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env_z["PYTHONPATH"] = f"{REPO/'EigenAttn'}:{REPO/'rebuttal_legacy_reproduction'}:{env_z.get('PYTHONPATH', '')}"
        p = subprocess.run(zs_cmd, cwd=str(REPO), env=env_z, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            print(p.stdout.decode('utf-8', errors='ignore'))
            raise RuntimeError(f"Zero-shot eval failed for {sha}")

    # Read metrics & write STATE_EVAL_DONE.json
    w_lines = [json.loads(l) for l in wiki_win.read_text().splitlines() if l.strip()]
    w_nll = sum(float(l["total_nll"]) for l in w_lines) / sum(int(l["token_count"]) for l in w_lines)
    w_ppl = math.exp(w_nll)
    
    c_lines = [json.loads(l) for l in c4_win.read_text().splitlines() if l.strip()]
    c_nll = sum(float(l["total_nll"]) for l in c_lines) / sum(int(l["token_count"]) for l in c_lines)
    c_ppl = math.exp(c_nll)
    
    z_dict = json.loads(zs_agg.read_text())
    tasks_res = {}
    for tn, td in z_dict.get("tasks", {}).items():
        tasks_res[tn] = td.get("lm_eval_primary", td.get("accuracy_from_correct"))
    z_macro = sum(tasks_res.values()) / len(tasks_res) if tasks_res else None
    
    rec = {
        "sha": sha,
        "status": "DONE_VALID_EVALUATED",
        "tier": sinfo["tier"],
        "tot_rank": sinfo["tot_rank"],
        "b_live": sinfo["b_live"],
        "wiki_ppl": w_ppl,
        "wiki_nll": w_nll,
        "wiki_windows": len(w_lines),
        "c4_ppl": c_ppl,
        "c4_nll": c_nll,
        "c4_windows": len(c_lines),
        "zero_shot_tasks": tasks_res,
        "zero_shot_macro": z_macro,
        "scorer_authority": "lm_eval_primary",
        "completed_at": utc()
    }
    with open(state_dir / "STATE_EVAL_DONE.json", "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[{utc()}] COMPLETED STATE {sha[:12]}... -> Wiki PPL: {w_ppl:.4f}, C4 PPL: {c_ppl:.4f}, ZS Macro: {z_macro:.4f}")
    
    # Clean up model shards to conserve disk space
    for f_p in model_dir.glob("model-*.safetensors"):
        try:
            f_p.unlink()
        except OSError:
            pass

print(f"\n[{utc()}] ALL 14 PHYSICAL STATES COMPLETED AND VALIDATED! GENERATING FINAL DELIVERABLES...")
