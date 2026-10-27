#!/usr/bin/env python3
import os, json, math, shutil, subprocess
from pathlib import Path
from datetime import datetime, timezone
import concurrent.futures

WORKSPACE   = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1")
FORMAL_DIR  = WORKSPACE / "formal_ablation"
PER_WIN_DIR = FORMAL_DIR / "CORE_FORMAL_PER_WINDOW"
SCRATCH_DIR = Path("/diskF/jiaqizhang/core_models_ultra_scratch")
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
ULTRA_BASIS = "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/pass2_historical_fdp120/eigen_basis_llama3_8b_wikitext2_frag_eb120_perlayer_eb.pt"

def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def mkenv(gpu_id, extra=""):
    e = os.environ.copy()
    e["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    parts = [str(REPO / "EigenAttn")]
    if extra:
        parts.append(str(REPO / extra))
    parts.append(e.get("PYTHONPATH", ""))
    e["PYTHONPATH"] = ":".join(parts)
    return e

def run_state(sha, sinfo, gpu_id, wid):
    state_dir = PER_WIN_DIR / sha
    state_dir.mkdir(parents=True, exist_ok=True)
    done_file = state_dir / "STATE_EVAL_DONE.json"
    if done_file.exists():
        print("[{}][UW{}] {} already done.".format(utc(), wid, sha[:12]))
        return True

    model_dir = SCRATCH_DIR / "model_uw{}_{}" .format(wid, sha[:12])
    model_dir.mkdir(parents=True, exist_ok=True)
    print("[{}][UW{}][GPU {}] Starting {} Tier={} Rank={}".format(
        utc(), wid, gpu_id, sha[:12], sinfo['tier'], sinfo['tot_rank']))

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
            print("[{}][UW{}][GPU {}] Materializing on /diskF...".format(utc(), wid, gpu_id))
            p = subprocess.run(cmd, cwd=str(REPO), env=mkenv(gpu_id),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                txt = p.stdout.decode("utf-8", errors="ignore")
                print("[UW{}] COMPRESS FAILED rc={}:".format(wid, p.returncode))
                print(txt[-2000:])
                return False
            if not list(model_dir.glob("model-*.safetensors")):
                print("[UW{}] No safetensors found!".format(wid))
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
            print("[{}][UW{}][GPU {}] Running Wiki PPL...".format(utc(), wid, gpu_id))
            p = subprocess.run(cmd, cwd=str(REPO),
                               env=mkenv(gpu_id, "rebuttal_legacy_reproduction"),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                print("[UW{}] WIKI FAILED:".format(wid))
                print(p.stdout.decode("utf-8", errors="ignore")[-2000:])
                return False
            out = wiki_out / "ppl_windows.jsonl"
            if out.exists():
                out.replace(wiki_win)
            else:
                print("[UW{}] No wiki ppl_windows.jsonl".format(wid))
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
            print("[{}][UW{}][GPU {}] Running C4 PPL...".format(utc(), wid, gpu_id))
            p = subprocess.run(cmd, cwd=str(REPO),
                               env=mkenv(gpu_id, "rebuttal_legacy_reproduction"),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                print("[UW{}] C4 FAILED:".format(wid))
                print(p.stdout.decode("utf-8", errors="ignore")[-2000:])
                return False
            out = c4_out / "ppl_windows.jsonl"
            if out.exists():
                out.replace(c4_win)
            else:
                print("[UW{}] No c4 ppl_windows.jsonl".format(wid))
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
            print("[{}][UW{}][GPU {}] Running 5-Task ZS...".format(utc(), wid, gpu_id))
            p = subprocess.run(cmd, cwd=str(REPO),
                               env=mkenv(gpu_id, "rebuttal_legacy_reproduction"),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                print("[UW{}] ZS FAILED:".format(wid))
                print(p.stdout.decode("utf-8", errors="ignore")[-2000:])
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
        print("[{}][UW{}] DONE {} Wiki={:.4f} C4={:.4f} ZS={:.4f}".format(
            utc(), wid, sha[:12], w_ppl, c_ppl, z_macro or 0))
        return True
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)
        print("[{}][UW{}] Cleaned up {}".format(utc(), wid, model_dir))

if __name__ == "__main__":
    with open(FORMAL_DIR / "CORE_RESOURCE_AUTHORITY.json") as f:
        core_rows = json.load(f)

    states = {}
    for r in core_rows:
        for kind in ["natural", "matched"]:
            sha = r["{}_rank_sha".format(kind)]
            if sha[:12] in ("6827686f8a78", "b09aa779e03b"):
                if (PER_WIN_DIR / sha / "STATE_EVAL_DONE.json").exists():
                    print("[{}] {} already done.".format(utc(), sha[:12]))
                    continue
                if sha not in states:
                    states[sha] = {
                        "tier": r["tier"], "method": r["method"], "sha": sha,
                        "tot_rank": r["{}_tot_rank".format(kind)],
                        "b_live": r["{}_b_live".format(kind)],
                        "basis_pt": ULTRA_BASIS
                    }

    if not states:
        print("[{}] Both ultra no-anchor states already done!".format(utc()))
    else:
        print("[{}] Launching {} ultra no-anchor workers GPU0+GPU1 scratch=/diskF".format(utc(), len(states)))
        gpu_map = [0, 1]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futs = {executor.submit(run_state, sha, sinfo, gpu_map[i % 2], i): sha
                    for i, (sha, sinfo) in enumerate(states.items())}
            for fut in concurrent.futures.as_completed(futs):
                try:
                    fut.result()
                except Exception as e:
                    print("[{}] EXCEPTION: {}".format(utc(), e))
    print("[{}] Ultra GPU01 workers finished.".format(utc()))
