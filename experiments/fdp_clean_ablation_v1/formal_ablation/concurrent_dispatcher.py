#!/usr/bin/env python3
"""
CONCURRENT MULTI-GPU FORMAL WORKER DISPATCHER v4 (FIXED)
=========================================================
Key fixes vs v3:
  1. NEVER manually write low_rank_config.json before or after compress.
     main_eigen_attn.py writes low_rank_config.pt/.json itself - trust it.
  2. 4 workers (GPU3: W0,W1; GPU2: W2,W3) for maximum throughput.
  3. Scratch strictly on /diskE/jiaqizhang/core_models_scratch.
  4. try/finally guarantees cleanup even on error.
"""
import os
import json
import math
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import concurrent.futures

WORKSPACE = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1")
FORMAL_DIR = WORKSPACE / "formal_ablation"
PER_WIN_DIR = FORMAL_DIR / "CORE_FORMAL_PER_WINDOW"
SCRATCH_DIR = Path("/diskE/jiaqizhang/core_models_scratch")
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

MODEL_BASE = "/diskD/jiaqizhang/Llama-3"
PY_BIN = "/home/bai-project/anaconda3/envs/eigen/bin/python"
REPO = Path("/home/bai-project/jiaqizhang/EigenAttn")
PPL_SCRIPT = REPO / "rebuttal_legacy_reproduction/runners/ppl_instrumented.py"
ZS_SCRIPT  = REPO / "rebuttal_legacy_reproduction/runners/zero_shot_instrumented.py"
MAIN_SCRIPT = REPO / "EigenAttn/main_eigen_attn.py"
WIKI_DIR   = "/home/bai-project/jiaqizhang/wikitext2"
C4_TEXTS   = "/diskF/jiaqizhang/fdp_eb_orth_paper_v1_results/offline_calib/c4/c4_ppl_selected_texts.jsonl"
ZS_YAML_DIR = "/diskF/jiaqizhang/a1804298693/offline_datasets"
CACHE_DIR  = REPO / "EigenAttn/HF_cache"

def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def run_single_state(sha, sinfo, gpu_id, worker_id):
    state_dir = PER_WIN_DIR / sha
    state_dir.mkdir(parents=True, exist_ok=True)
    done_file = state_dir / "STATE_EVAL_DONE.json"
    if done_file.exists():
        print(f"[{utc()}][W{worker_id}] {sha[:12]} already done, skip.")
        return True

    model_dir = SCRATCH_DIR / f"model_w{worker_id}_{sha[:12]}"
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{utc()}][W{worker_id}][GPU {gpu_id}] Starting {sha[:12]} (Tier={sinfo['tier']}, Rank={sinfo['tot_rank']})...")

    def mkenv(extra_path=""):
        e = os.environ.copy()
        e["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        parts = [str(REPO / "EigenAttn")]
        if extra_path:
            parts.append(str(REPO / extra_path))
        parts.append(e.get("PYTHONPATH", ""))
        e["PYTHONPATH"] = ":".join(parts)
        return e

    try:
        # ── 1. Compress (materialize model weights) ──────────────────────
        shards = list(model_dir.glob("model-*.safetensors"))
        if not shards:
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
            print(f"[{utc()}][W{worker_id}][GPU {gpu_id}] Materializing on /diskE...")
            p = subprocess.run(compress_cmd, cwd=str(REPO), env=mkenv(),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            # Log first 2000 chars of output to help debug any failures
            out_text = p.stdout.decode("utf-8", errors="ignore")
            if p.returncode != 0:
                print(f"[{utc()}][W{worker_id}] COMPRESS FAILED (rc={p.returncode}):")
                print(out_text[-3000:])
                return False
            # Verify shards were actually written
            shards = list(model_dir.glob("model-*.safetensors"))
            if not shards:
                print(f"[{utc()}][W{worker_id}] COMPRESS OK but no safetensors found in {model_dir}")
                return False
            # IMPORTANT: Do NOT write or overwrite low_rank_config.json here.
            # main_eigen_attn.py has already written low_rank_config.pt/.json
            # to model_dir with the actual per-layer ranks used. Trust it.

        # ── 2. WikiText-2 PPL ────────────────────────────────────────────
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
                "--method", sinfo["method"],
                "--net", "llama",
                "--seed", "2",
                "--wikitext2_dir", WIKI_DIR,
                "--evaluation_protocol_hash",
                "da93ade0391d1886941e896092c016d08918defcb8c68eab7e315758209f0644",
            ]
            print(f"[{utc()}][W{worker_id}][GPU {gpu_id}] Running WikiText-2 PPL...")
            p = subprocess.run(wiki_cmd, cwd=str(REPO),
                               env=mkenv("rebuttal_legacy_reproduction"),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                print(f"[{utc()}][W{worker_id}] WIKI EVAL FAILED:")
                print(p.stdout.decode("utf-8", errors="ignore")[-3000:])
                return False
            out_file = wiki_out / "ppl_windows.jsonl"
            if out_file.exists():
                out_file.replace(wiki_win)
            else:
                print(f"[{utc()}][W{worker_id}] WIKI eval succeeded but no ppl_windows.jsonl")
                return False

        # ── 3. C4 PPL ────────────────────────────────────────────────────
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
                "--method", sinfo["method"],
                "--net", "llama",
                "--seed", "2",
                "--evaluation_protocol_hash",
                "da93ade0391d1886941e896092c016d08918defcb8c68eab7e315758209f0644",
            ]
            print(f"[{utc()}][W{worker_id}][GPU {gpu_id}] Running C4 PPL...")
            p = subprocess.run(c4_cmd, cwd=str(REPO),
                               env=mkenv("rebuttal_legacy_reproduction"),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                print(f"[{utc()}][W{worker_id}] C4 EVAL FAILED:")
                print(p.stdout.decode("utf-8", errors="ignore")[-3000:])
                return False
            out_file = c4_out / "ppl_windows.jsonl"
            if out_file.exists():
                out_file.replace(c4_win)
            else:
                print(f"[{utc()}][W{worker_id}] C4 eval succeeded but no ppl_windows.jsonl")
                return False

        # ── 4. 5-Task Zero-Shot ──────────────────────────────────────────
        zs_out = state_dir / "zeroshot"
        zs_out.mkdir(parents=True, exist_ok=True)
        zs_agg = zs_out / "zero_shot_aggregate.json"
        if not zs_agg.exists():
            zs_cmd = [
                PY_BIN, "-u", str(ZS_SCRIPT),
                "--model", str(model_dir),
                "--tokenizer_model", MODEL_BASE,
                "--task_yaml_dir", ZS_YAML_DIR,
                "--tasks",
                "hellaswag_localjsonl,arc_challenge_localjsonl,arc_easy_localjsonl,winogrande_localjsonl,piqa_localjsonl",
                "--out_dir", str(zs_out),
                "--net", "llama-3-8b",
                "--dtype", "bf16",
                "--method", sinfo["method"],
                "--cache_dir", str(CACHE_DIR),
                "--attn_implementation", "eager",
            ]
            print(f"[{utc()}][W{worker_id}][GPU {gpu_id}] Running 5-Task Zero-Shot...")
            p = subprocess.run(zs_cmd, cwd=str(REPO),
                               env=mkenv("rebuttal_legacy_reproduction"),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                print(f"[{utc()}][W{worker_id}] ZS EVAL FAILED:")
                print(p.stdout.decode("utf-8", errors="ignore")[-3000:])
                return False

        # ── 5. Aggregate metrics & write DONE ───────────────────────────
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
            "completed_at": utc(),
        }
        with open(done_file, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[{utc()}][W{worker_id}] DONE {sha[:12]} => Wiki={w_ppl:.4f} C4={c_ppl:.4f} ZS={z_macro:.4f}")
        return True

    finally:
        shutil.rmtree(model_dir, ignore_errors=True)
        print(f"[{utc()}][W{worker_id}] Cleaned up {model_dir}")


if __name__ == "__main__":
    with open(FORMAL_DIR / "CORE_RESOURCE_AUTHORITY.json") as f:
        core_rows = json.load(f)

    basis_paths = {
        "light":      "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb045/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt",
        "medium":     "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb060/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt",
        "aggressive": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb090/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt",
        "ultra":      "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/pass2_historical_fdp120/eigen_basis_llama3_8b_wikitext2_frag_eb120_perlayer_eb.pt",
    }

    states_to_run = {}
    for r in core_rows:
        for kind in ["natural", "matched"]:
            sha = r[f"{kind}_rank_sha"]
            if (PER_WIN_DIR / sha / "STATE_EVAL_DONE.json").exists():
                continue
            if sha not in states_to_run:
                if r["method"] == "M3_UNIFORM" and r["tier"] == "ultra":
                    b_pt = "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/lattice_fine/eigen_basis_llama3_8b_wikitext2_uniform_fine_eb0p113.pt"
                else:
                    b_pt = basis_paths[r["tier"]]
                states_to_run[sha] = {
                    "tier":      r["tier"],
                    "method":    r["method"],
                    "sha":       sha,
                    "tot_rank":  r[f"{kind}_tot_rank"],
                    "b_live":    r[f"{kind}_b_live"],
                    "basis_pt":  b_pt,
                }

    items = list(states_to_run.items())
    print(f"[{utc()}] {len(items)} states to evaluate (SCRATCH=/diskE).")
    for i, (sha, s) in enumerate(items):
        print(f"  [{i}] {sha[:12]} tier={s['tier']} method={s['method']}")

    if not items:
        print(f"[{utc()}] All states already done!")
    else:
        # 4 workers: GPU3 gets slots 0,1; GPU2 gets slots 2,3
        gpu_slots = [3, 2, 3, 2]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futs = {
                executor.submit(run_single_state, sha, sinfo,
                                gpu_slots[i % len(gpu_slots)], i): sha
                for i, (sha, sinfo) in enumerate(items)
            }
            for fut in concurrent.futures.as_completed(futs):
                sha = futs[fut]
                try:
                    ok = fut.result()
                    if not ok:
                        print(f"[{utc()}] WARNING: {sha[:12]} returned False (see errors above)")
                except Exception as e:
                    print(f"[{utc()}] EXCEPTION for {sha[:12]}: {e}")

    print(f"[{utc()}] All concurrent tasks finished.")
