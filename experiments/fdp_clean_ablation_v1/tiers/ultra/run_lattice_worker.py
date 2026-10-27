#!/usr/bin/env python3
"""Run Uniform comparator lattice evaluations for a list of error budgets on a given GPU."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eb_list", required=True, help="Comma-separated list of error budgets, e.g. '0.108,0.110,0.112'")
    ap.add_argument("--gpu", required=True, help="GPU ID to use (0, 1, 2, or 3)")
    ap.add_argument("--out_dir", default="/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/lattice")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ebs = [float(x.strip()) for x in args.eb_list.split(",") if x.strip()]
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Starting lattice worker on GPU {args.gpu} for budgets: {ebs}", flush=True)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONPATH"] = f"/diskF/jiaqizhang/fdp_lowrank_operator_zoo_v1/repos/eigenattention_upstream:{env.get('PYTHONPATH', '')}"

    script_path = "/home/bai-project/jiaqizhang/EigenAttn/experimental/fdp_eb_orth_paper_v1/official_baselines/official_eigenattn_rank_only.py"

    for eb in ebs:
        out_file = out_dir / f"uniform_eb_{eb:.5f}.json"
        if out_file.is_file():
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] eb={eb:.5f} already exists at {out_file}, skipping.", flush=True)
            continue
            
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Running official Uniform rank-only for eb={eb:.5f} on GPU {args.gpu}...", flush=True)
        cmd = [
            "/home/bai-project/anaconda3/envs/eigen/bin/python",
            script_path,
            "--model", "/diskD/jiaqizhang/Llama-3",
            "--data_dir", "/diskD/jiaqizhang/Llama-3",
            "--scalar_eb", str(eb),
            "--out", str(out_file),
            "--nsamples", "64",
            "--seqlen", "2048",
            "--avg_dim", "32",
            "--seed", "2"
        ]
        t0 = time.time()
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        t_cost = time.time() - t0
        if res.returncode != 0:
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Failed for eb={eb:.5f}! Return code: {res.returncode}", flush=True)
            print("STDERR:", res.stderr, flush=True)
        else:
            try:
                with open(out_file) as f:
                    data = json.load(f)
                b_live = data.get("B_live")
                rank_sha = data.get("rank_sha")[:12]
                print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] eb={eb:.5f} DONE in {t_cost:.1f}s: B_live={b_live}, SHA={rank_sha}...", flush=True)
            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Completed but error reading json: {e}", flush=True)

    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Lattice worker on GPU {args.gpu} finished all tasks!", flush=True)

if __name__ == "__main__":
    main()
