#!/usr/bin/env python3
import os, json, math, csv
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

WORKSPACE   = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1")
FORMAL_DIR  = WORKSPACE / "formal_ablation"
PER_WIN_DIR = FORMAL_DIR / "CORE_FORMAL_PER_WINDOW"

def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

with open(FORMAL_DIR / "CORE_RESOURCE_AUTHORITY.json") as f:
    core_rows = json.load(f)

eval_cache = {}
for p in PER_WIN_DIR.glob("*/STATE_EVAL_DONE.json"):
    sha = p.parent.name
    with open(p) as f:
        eval_cache[sha] = json.load(f)

print(f"Loaded {len(eval_cache)} evaluation records.")

formal_rows = []
zs_rows = []
matching_rows = []

for r in core_rows:
    tier = r["tier"]
    method = r["method"]
    target_b_live = r["target_b_live"]
    
    # Natural
    nat_sha = r["natural_rank_sha"]
    nat_eval = eval_cache.get(nat_sha, {})
    nat_tot_rank = r["natural_tot_rank"]
    nat_b_live = r["natural_b_live"]
    nat_gap = r["natural_gap_pct"]
    nat_status = "MATCHED" if nat_gap <= 0.15 else ("NEAR_MATCHED" if nat_gap <= 1.0 else "UNMATCHED")
    
    # Matched
    mat_sha = r["matched_rank_sha"]
    mat_eval = eval_cache.get(mat_sha, {})
    mat_tot_rank = r["matched_tot_rank"]
    mat_b_live = r["matched_b_live"]
    mat_gap = r["matched_gap_pct"]
    mat_status = "MATCHED" if mat_gap <= 0.15 else ("NEAR_MATCHED" if mat_gap <= 1.0 else "UNMATCHED")
    
    formal_rows.append({
        "tier": tier,
        "method": method,
        "target_b_live": target_b_live,
        "natural_tot_rank": nat_tot_rank,
        "natural_b_live": nat_b_live,
        "natural_gap_pct": round(nat_gap, 4),
        "natural_match_status": nat_status,
        "natural_rank_sha": nat_sha,
        "natural_wiki_ppl": round(nat_eval.get("wiki_ppl", 0.0), 4) if nat_eval.get("wiki_ppl") else None,
        "natural_c4_ppl": round(nat_eval.get("c4_ppl", 0.0), 4) if nat_eval.get("c4_ppl") else None,
        "natural_zs_macro": round(nat_eval.get("zero_shot_macro", 0.0), 4) if nat_eval.get("zero_shot_macro") else None,
        "matched_tot_rank": mat_tot_rank,
        "matched_b_live": mat_b_live,
        "matched_gap_pct": round(mat_gap, 4),
        "matched_match_status": mat_status,
        "matched_rank_sha": mat_sha,
        "matched_wiki_ppl": round(mat_eval.get("wiki_ppl", 0.0), 4) if mat_eval.get("wiki_ppl") else None,
        "matched_c4_ppl": round(mat_eval.get("c4_ppl", 0.0), 4) if mat_eval.get("c4_ppl") else None,
        "matched_zs_macro": round(mat_eval.get("zero_shot_macro", 0.0), 4) if mat_eval.get("zero_shot_macro") else None,
    })
    
    for kind, sha, ev, tot_r, b_l in [
        ("natural", nat_sha, nat_eval, nat_tot_rank, nat_b_live),
        ("matched", mat_sha, mat_eval, mat_tot_rank, mat_b_live)
    ]:
        zs_tasks = ev.get("zero_shot_tasks", {})
        zs_rows.append({
            "tier": tier,
            "method": method,
            "evaluation_kind": kind,
            "rank_sha": sha,
            "tot_rank": tot_r,
            "b_live": b_l,
            "hellaswag": round(zs_tasks.get("hellaswag_localjsonl", 0.0), 4) if zs_tasks.get("hellaswag_localjsonl") is not None else None,
            "arc_challenge": round(zs_tasks.get("arc_challenge_localjsonl", 0.0), 4) if zs_tasks.get("arc_challenge_localjsonl") is not None else None,
            "arc_easy": round(zs_tasks.get("arc_easy_localjsonl", 0.0), 4) if zs_tasks.get("arc_easy_localjsonl") is not None else None,
            "winogrande": round(zs_tasks.get("winogrande_localjsonl", 0.0), 4) if zs_tasks.get("winogrande_localjsonl") is not None else None,
            "piqa": round(zs_tasks.get("piqa_localjsonl", 0.0), 4) if zs_tasks.get("piqa_localjsonl") is not None else None,
            "macro_avg": round(ev.get("zero_shot_macro", 0.0), 4) if ev.get("zero_shot_macro") is not None else None,
        })
        
        matching_rows.append({
            "tier": tier,
            "method": method,
            "evaluation_kind": kind,
            "target_b_live": target_b_live,
            "realized_b_live": b_l,
            "tot_rank": tot_r,
            "abs_gap_bytes": abs(b_l - target_b_live),
            "gap_pct": round(abs(b_l - target_b_live) / target_b_live * 100, 4),
            "match_class": "STRICT_MATCH" if abs(b_l - target_b_live) / target_b_live * 100 <= 0.15 else ("NEAR_MATCHED" if abs(b_l - target_b_live) / target_b_live * 100 <= 1.0 else "UNMATCHED"),
            "rank_sha": sha
        })

with open(FORMAL_DIR / "CORE_FORMAL_RESULTS.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(formal_rows[0].keys()))
    writer.writeheader()
    writer.writerows(formal_rows)

with open(FORMAL_DIR / "CORE_ZERO_SHOT_RESULTS.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(zs_rows[0].keys()))
    writer.writeheader()
    writer.writerows(zs_rows)

with open(FORMAL_DIR / "CORE_RESOURCE_MATCHING.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(matching_rows[0].keys()))
    writer.writeheader()
    writer.writerows(matching_rows)

print("Wrote CORE_FORMAL_RESULTS.csv, CORE_ZERO_SHOT_RESULTS.csv, CORE_RESOURCE_MATCHING.csv.")

# Bootstrap (10,000 resamples)
bootstrap_results = []
rng = np.random.default_rng(seed=42)
N_BOOT = 10000

for tier in ["light", "medium", "aggressive", "ultra"]:
    m1_sha = next(r["natural_rank_sha"] for r in core_rows if r["tier"] == tier and r["method"] == "M1_F_DEV_STAR")
    uni_sha = next(r["natural_rank_sha"] for r in core_rows if r["tier"] == tier and r["method"] == "M3_UNIFORM")
    noanc_sha = next(r["natural_rank_sha"] for r in core_rows if r["tier"] == tier and r["method"] == "M4_NO_ANCHOR")

    for benchmark in ["wiki", "c4"]:
        m1_file = PER_WIN_DIR / m1_sha / f"{benchmark}_windows.jsonl"
        uni_file = PER_WIN_DIR / uni_sha / f"{benchmark}_windows.jsonl"
        noanc_file = PER_WIN_DIR / noanc_sha / f"{benchmark}_windows.jsonl"

        if m1_file.exists() and uni_file.exists():
            m1_lines = [json.loads(l) for l in m1_file.read_text().splitlines() if l.strip()]
            uni_lines = [json.loads(l) for l in uni_file.read_text().splitlines() if l.strip()]
            
            n_win = min(len(m1_lines), len(uni_lines))
            m1_nlls = np.array([float(m1_lines[i]["total_nll"]) / int(m1_lines[i]["token_count"]) for i in range(n_win)])
            uni_nlls = np.array([float(uni_lines[i]["total_nll"]) / int(uni_lines[i]["token_count"]) for i in range(n_win)])
            
            diff_orig = m1_nlls - uni_nlls
            mean_diff = float(np.mean(diff_orig))
            
            idx = rng.integers(0, n_win, size=(N_BOOT, n_win))
            boot_means = np.mean(diff_orig[idx], axis=1)
            ci_lower = float(np.percentile(boot_means, 2.5))
            ci_upper = float(np.percentile(boot_means, 97.5))
            p_val = float(2 * min(np.mean(boot_means <= 0), np.mean(boot_means >= 0)))
            p_val = min(p_val, 1.0)
            
            bootstrap_results.append({
                "tier": tier,
                "benchmark": benchmark,
                "comparison": "M1_F_DEV_STAR vs M3_UNIFORM (Natural)",
                "n_windows": n_win,
                "mean_nll_diff": round(mean_diff, 6),
                "ci_95_lower": round(ci_lower, 6),
                "ci_95_upper": round(ci_upper, 6),
                "p_value_two_sided": round(p_val, 4),
                "m1_mean_nll": round(float(np.mean(m1_nlls)), 6),
                "comparator_mean_nll": round(float(np.mean(uni_nlls)), 6),
            })

        if m1_file.exists() and noanc_file.exists():
            noanc_lines = [json.loads(l) for l in noanc_file.read_text().splitlines() if l.strip()]
            n_win = min(len(m1_lines), len(noanc_lines))
            m1_nlls = np.array([float(m1_lines[i]["total_nll"]) / int(m1_lines[i]["token_count"]) for i in range(n_win)])
            noanc_nlls = np.array([float(noanc_lines[i]["total_nll"]) / int(noanc_lines[i]["token_count"]) for i in range(n_win)])
            
            diff_orig = m1_nlls - noanc_nlls
            mean_diff = float(np.mean(diff_orig))
            idx = rng.integers(0, n_win, size=(N_BOOT, n_win))
            boot_means = np.mean(diff_orig[idx], axis=1)
            ci_lower = float(np.percentile(boot_means, 2.5))
            ci_upper = float(np.percentile(boot_means, 97.5))
            p_val = float(2 * min(np.mean(boot_means <= 0), np.mean(boot_means >= 0)))
            p_val = min(p_val, 1.0)
            
            bootstrap_results.append({
                "tier": tier,
                "benchmark": benchmark,
                "comparison": "M1_F_DEV_STAR vs M4_NO_ANCHOR (Natural)",
                "n_windows": n_win,
                "mean_nll_diff": round(mean_diff, 6),
                "ci_95_lower": round(ci_lower, 6),
                "ci_95_upper": round(ci_upper, 6),
                "p_value_two_sided": round(p_val, 4),
                "m1_mean_nll": round(float(np.mean(m1_nlls)), 6),
                "comparator_mean_nll": round(float(np.mean(noanc_nlls)), 6),
            })

with open(FORMAL_DIR / "CORE_BOOTSTRAP_RESULTS.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(bootstrap_results[0].keys()))
    writer.writeheader()
    writer.writerows(bootstrap_results)

print("Wrote CORE_BOOTSTRAP_RESULTS.csv.")

status = {
    "status": "CORE_FORMAL_CAMPAIGN_COMPLETE",
    "completed_at": utc(),
    "unique_physical_states_evaluated": len(eval_cache),
    "core_matrix_rows": len(formal_rows),
    "bootstrap_evaluations": len(bootstrap_results),
    "authorities_locked": True,
}
with open(FORMAL_DIR / "CORE_CAMPAIGN_STATUS.json", "w") as f:
    json.dump(status, f, indent=2)

print("SUCCESS: CORE_FORMAL_CAMPAIGN_COMPLETE")
