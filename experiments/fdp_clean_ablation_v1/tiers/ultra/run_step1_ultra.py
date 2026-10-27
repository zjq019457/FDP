#!/usr/bin/env python3
import subprocess
import sys
import time

print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Starting Step-1 Diagnostic for Ultra e=0.120...")
cmd = [
    "/home/bai-project/anaconda3/envs/eigen/bin/python",
    "-u",
    "/home/bai-project/jiaqizhang/EigenAttn/EigenAttn/step1_diagnose_llama.py",
    "--base", "/diskD/jiaqizhang/Llama-3",
    "--cmp", "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/pass1_uniform/final_model",
    "--tok", "/diskD/jiaqizhang/Llama-3",
    "--dtype", "bf16",
    "--seqlen", "512",
    "--num_seqs", "2",
    "--out", "/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles/step1_llama3_8b_eb120.json",
    "--gpu", "3",
    "--attn_implementation", "eager"
]

res = subprocess.run(cmd)
if res.returncode != 0:
    print(f"Error running step1 diagnostic: return code {res.returncode}")
    sys.exit(res.returncode)
print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Step-1 Diagnostic successfully finished!")
