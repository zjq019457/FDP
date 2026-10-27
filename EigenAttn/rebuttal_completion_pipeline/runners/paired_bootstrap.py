#!/usr/bin/env python3
"""Paired bootstrap (10000), McNemar, Holm from instrumented window/sample raw outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parents[1]
LEGACY = HERE.parent / "rebuttal_legacy_reproduction"
sys.path.insert(0, str(HERE))
from common import BOOTSTRAP_N, BOOTSTRAP_SEED, atomic_write_json, here, legacy_artifacts, load_jsonl, now_iso  # noqa: E402

ART = here() / "artifacts"


def _ppl_windows(method_prefix: str, dataset: str) -> List[Dict[str, Any]]:
    name = f"{method_prefix}_wikitext_ppl" if dataset == "wikitext2" else f"{method_prefix}_c4_ppl"
    for base in (ART, legacy_artifacts(), LEGACY / "artifacts"):
        p = base / name / "ppl_windows.jsonl"
        if p.exists():
            return load_jsonl(p)
    return []


def paired_bootstrap_ppl(w_a: List[Dict], w_b: List[Dict], n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    by_id_a = {r["window_id"]: r for r in w_a}
    by_id_b = {r["window_id"]: r for r in w_b}
    common = sorted(set(by_id_a) & set(by_id_b))
    if not common:
        return {"error": "no common windows"}
    rng = random.Random(seed)
    deltas = []
    for _ in range(n):
        sample = [common[rng.randint(0, len(common) - 1)] for _ in range(len(common))]
        nll_a = sum(by_id_a[w]["total_nll"] for w in sample)
        nll_b = sum(by_id_b[w]["total_nll"] for w in sample)
        tok_a = sum(by_id_a[w]["token_count"] for w in sample)
        tok_b = sum(by_id_b[w]["token_count"] for w in sample)
        ppl_a = math.exp(nll_a / tok_a)
        ppl_b = math.exp(nll_b / tok_b)
        deltas.append(ppl_b - ppl_a)
    deltas.sort()
    lo = deltas[int(0.025 * n)]
    hi = deltas[int(0.975 * n)]
    return {
        "n_bootstrap": n,
        "n_windows": len(common),
        "mean_delta_ppl": sum(deltas) / n,
        "ci95_low": lo,
        "ci95_high": hi,
        "significant_at_05": not (lo <= 0 <= hi),
    }


def _zs_sample_path(method_prefix: str, task: str) -> Optional[Path]:
    for base in (ART, legacy_artifacts(), LEGACY / "artifacts"):
        for suffix in ("zs_hs_arc", "zs_wg_piqa"):
            p = base / f"{method_prefix}_{suffix}" / "zero_shot_samples" / f"{task}.jsonl"
            if p.exists():
                return p
    return None


def _zs_samples(method_prefix: str, task: str) -> List[Dict]:
    for base in (ART, legacy_artifacts(), LEGACY / "artifacts"):
        for suffix in ("zs_hs_arc", "zs_wg_piqa"):
            p = base / f"{method_prefix}_{suffix}" / "zero_shot_samples" / f"{task}.jsonl"
            if p.exists():
                return load_jsonl(p)
    return []


def paired_bootstrap_acc(samples_a: List[Dict], samples_b: List[Dict], n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    samples_a = [r for r in samples_a if isinstance(r, dict)]
    samples_b = [r for r in samples_b if isinstance(r, dict)]
    by_id_a = {r.get("sample_id", i): r for i, r in enumerate(samples_a)}
    by_id_b = {r.get("sample_id", i): r for i, r in enumerate(samples_b)}
    common = sorted(set(by_id_a) & set(by_id_b))
    if not common:
        return {"error": "no common samples"}
    rng = random.Random(seed)
    deltas = []
    for _ in range(n):
        sample = [common[rng.randint(0, len(common) - 1)] for _ in range(len(common))]
        acc_a = sum(1 for s in sample if by_id_a[s].get("correct")) / len(sample)
        acc_b = sum(1 for s in sample if by_id_b[s].get("correct")) / len(sample)
        deltas.append(acc_b - acc_a)
    deltas.sort()
    lo, hi = deltas[int(0.025 * n)], deltas[int(0.975 * n)]
    return {"mean_delta_acc": sum(deltas) / n, "ci95_low": lo, "ci95_high": hi, "significant_at_05": not (lo <= 0 <= hi)}


def mcnemar(samples_a: List[Dict], samples_b: List[Dict]) -> Dict[str, Any]:
    samples_a = [r for r in samples_a if isinstance(r, dict)]
    samples_b = [r for r in samples_b if isinstance(r, dict)]
    by_id_a = {r.get("sample_id", i): r for i, r in enumerate(samples_a)}
    by_id_b = {r.get("sample_id", i): r for i, r in enumerate(samples_b)}
    common = sorted(set(by_id_a) & set(by_id_b))
    b01 = b10 = 0
    for s in common:
        ca, cb = by_id_a[s].get("correct"), by_id_b[s].get("correct")
        if ca and not cb:
            b10 += 1
        elif cb and not ca:
            b01 += 1
    n = b01 + b10
    if n == 0:
        return {"p_value": 1.0, "b01": 0, "b10": 0}
    from math import comb
    p = 0.0
    for k in range(b01, n + 1):
        p += comb(n, k) * (0.5 ** n)
    return {"b01": b01, "b10": b10, "p_value": min(1.0, 2 * p), "n_discordant": n}


def holm_correction(pvalues: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
    m = len(pvalues)
    sorted_p = sorted(pvalues, key=lambda x: x[1])
    results = []
    for i, (name, p) in enumerate(sorted_p):
        adj = min(1.0, p * (m - i))
        results.append({"test": name, "p_raw": p, "p_holm": adj, "reject_at_05": adj < 0.05})
    return results


def run_legacy_confirmatory(out: Path) -> None:
    comparisons = [
        ("R2_uniform", "R3_fdp", "wikitext2", "Uniform vs FDP Wiki"),
        ("R2_uniform", "R3_fdp", "c4", "Uniform vs FDP C4"),
    ]
    rows = []
    for a, b, ds, label in comparisons:
        wa = _ppl_windows(a.replace("R2_", "R2_uniform").replace("R3_", "R3_fdp") if False else a, ds)
        wb = _ppl_windows(b, ds)
        if not wa:
            wa = _ppl_windows("R1_dense" if a.startswith("R1") else a, ds)
        stat = paired_bootstrap_ppl(wa, wb)
        stat["comparison"] = label
        stat["method_a"] = a
        stat["method_b"] = b
        rows.append(stat)
    atomic_write_json(out / "PPL_PAIRED_BOOTSTRAP_legacy.json", {"generated_at": now_iso(), "rows": rows})
    with (out / "PPL_PAIRED_BOOTSTRAP.csv").open("w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
            w.writeheader()
            w.writerows(rows)

    tasks = ["hellaswag_localjsonl", "arc_challenge_localjsonl", "arc_easy_localjsonl",
             "winogrande_localjsonl", "piqa_localjsonl"]
    task_rows, mcnemar_rows, pvals = [], [], []
    for task in tasks:
        sp = _zs_sample_path("R2_uniform", task) or _zs_sample_path("R1_dense", task)
        sp_b = _zs_sample_path("R3_fdp", task)
        sa = load_jsonl(sp) if sp else []
        sb = load_jsonl(sp_b) if sp_b else []
        bs = paired_bootstrap_acc(sa, sb)
        mc = mcnemar(sa, sb)
        bs["task"] = task
        task_rows.append(bs)
        mcnemar_rows.append({"task": task, **mc})
        if "p_value" in mc:
            pvals.append((task, mc["p_value"]))
    atomic_write_json(out / "TASK_PAIRED_BOOTSTRAP.json", {"rows": task_rows})
    atomic_write_json(out / "MCNEMAR.json", {"rows": mcnemar_rows})
    holm = holm_correction(pvals) if pvals else []
    atomic_write_json(out / "HOLM_CORRECTION.json", {"rows": holm})
    with (out / "STATISTICAL_CLAIM_GUARD.md").open("w", encoding="utf-8") as f:
        f.write("# Statistical Claim Guard\n\n")
        f.write("- Bootstrap n=10000, seed=42\n")
        f.write("- Do NOT claim significance unless CI excludes zero\n\n")
        for r in rows:
            f.write(f"- {r.get('comparison')}: CI [{r.get('ci95_low')}, {r.get('ci95_high')}], sig={r.get('significant_at_05')}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="legacy_confirmatory")
    args = ap.parse_args()
    out = HERE / "artifacts" / "statistics"
    out.mkdir(parents=True, exist_ok=True)
    if args.scope == "legacy_confirmatory":
        run_legacy_confirmatory(out)
    else:
        atomic_write_json(out / f"BOOTSTRAP_{args.scope}.json", {"status": "pending", "scope": args.scope})
    print(json.dumps({"ok": True, "scope": args.scope}, ensure_ascii=False))


if __name__ == "__main__":
    main()
