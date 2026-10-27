#!/usr/bin/env python3
import os
import sys
import json
import time
import subprocess
from pathlib import Path

def log(msg):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}", flush=True)

def main():
    log("=== ULTRA PIPELINE SEQUENCER ===")
    
    pass1_meta = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/pass1_uniform/final_model/pass1_meta.json")
    while not pass1_meta.is_file():
        log("Waiting for Pass 1 to finish...")
        time.sleep(15)
        
    log("Pass 1 verified complete! Starting Step 2.2: Step-1 Diagnostic...")
    step1_out = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles/step1_llama3_8b_eb120.json")
    cmd_step1 = [
        "/home/bai-project/anaconda3/envs/eigen/bin/python",
        "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/run_step1_ultra.py"
    ]
    res = subprocess.run(cmd_step1)
    if res.returncode != 0 or not step1_out.is_file():
        log("ERROR: Step 1 diagnostic failed!")
        sys.exit(1)
        
    log("Step 2.2 complete! Starting Step 2.3: Derive Ultra EB Schedule...")
    cmd_derive = [
        "/home/bai-project/anaconda3/envs/eigen/bin/python",
        "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/derive_ultra_eb.py"
    ]
    res = subprocess.run(cmd_derive)
    if res.returncode != 0:
        log("ERROR: Derive EB failed!")
        sys.exit(1)
        
    log("Step 2.3 complete! Starting Step 2.4: Historical Pass-2 Rank Search at e=0.120...")
    eb_schedule = "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/per_layer_eb_ultra_120.json"
    fdp_ranks_out = "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/fdp120_ranks.json"
    cmd_pass2 = [
        "/home/bai-project/anaconda3/envs/eigen/bin/python",
        "/home/bai-project/jiaqizhang/EigenAttn/experimental/fdp_eb_orth_paper_v1/official_baselines/official_faithful_oldfdp_rank_only.py",
        "--model", "/diskD/jiaqizhang/Llama-3",
        "--data_dir", "/diskD/jiaqizhang/Llama-3",
        "--scalar_eb", "0.120",
        "--per_layer_eb_json", eb_schedule,
        "--out", fdp_ranks_out,
        "--nsamples", "64",
        "--seqlen", "2048",
        "--avg_dim", "32",
        "--seed", "2"
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "3"
    env["PYTHONPATH"] = f"/diskF/jiaqizhang/fdp_lowrank_operator_zoo_v1/repos/eigenattention_upstream:{env.get('PYTHONPATH', '')}"
    res = subprocess.run(cmd_pass2, env=env)
    if res.returncode != 0:
        log("ERROR: Historical Pass 2 rank search failed!")
        sys.exit(1)
        
    log("=== ULTRA PIPELINE COMPLETE! ===")
    with open(fdp_ranks_out) as f:
        rep = json.load(f)
    log(f"FDP-120 Ranks: B_live={rep.get('B_live')}, SHA={rep.get('rank_sha')}")

if __name__ == "__main__":
    main()
