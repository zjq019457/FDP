#!/usr/bin/env python3
"""Paired bootstrap + McNemar + Holm for operating points (moderate/aggressive/mild).

Aligns ZS samples by (task, sample_id, document_hash) — not file line order.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    ARTIFACT_ROOT,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    atomic_write_json,
    load_jsonl,
    load_zs_samples_dir,
    now_iso,
)

TASKS = [
    "hellaswag_localjsonl",
    "arc_challenge_localjsonl",
    "arc_easy_localjsonl",
    "winogrande_localjsonl",
    "piqa_localjsonl",
]


def _ppl_windows(point: str, method: str, dataset: str) -> List[Dict[str, Any]]:
    tag = f"OP_{point}_{method}_{dataset}_ppl"
    root = ARTIFACT_ROOT / tag
    jsonl = root / "ppl_windows.jsonl"
    if jsonl.exists():
        rows = load_jsonl(jsonl)
        if rows and isinstance(rows[0], dict) and "window_id" in rows[0]:
            return rows
    # per-window files
    wdir = root / "windows"
    rows = []
    if wdir.is_dir():
        for p in sorted(wdir.glob("*.json")):
            if p.name.endswith(".DONE.json"):
                continue
            try:
                rows.append(json.loads(p.read_text()))
            except Exception:
                pass
    return rows


def _zs_root(point: str, method: str, task: str) -> Optional[Path]:
    bund = "hs_arc" if task.startswith(("hellaswag", "arc_")) else "wg_piqa"
    for name in (
        f"OP_{point}_{method}_{bund}",
        f"OP_{point}_{method}_zs_{bund}",
    ):
        p = ARTIFACT_ROOT / name
        if p.exists():
            return p
    return None


def _key(r: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(r.get("task") or ""),
        str(r.get("sample_id") if r.get("sample_id") is not None else ""),
        str(r.get("document_hash") or r.get("prompt_hash") or ""),
    )


def paired_bootstrap_ppl(wa: List[Dict], wb: List[Dict], n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    by_a = {r["window_id"]: r for r in wa if "window_id" in r}
    by_b = {r["window_id"]: r for r in wb if "window_id" in r}
    common = sorted(set(by_a) & set(by_b))
    if not common:
        return {"error": "no_common_windows", "n_a": len(by_a), "n_b": len(by_b)}
    rng = random.Random(seed)
    deltas = []
    for _ in range(n):
        sample = [common[rng.randint(0, len(common) - 1)] for _ in range(len(common))]
        nll_a = sum(by_a[w]["total_nll"] for w in sample)
        nll_b = sum(by_b[w]["total_nll"] for w in sample)
        tok_a = sum(by_a[w]["token_count"] for w in sample)
        tok_b = sum(by_b[w]["token_count"] for w in sample)
        deltas.append(math.exp(nll_b / tok_b) - math.exp(nll_a / tok_a))
    deltas.sort()
    lo, hi = deltas[int(0.025 * n)], deltas[int(0.975 * n)]
    return {
        "n_bootstrap": n,
        "n_windows": len(common),
        "mean_delta_ppl": sum(deltas) / n,
        "ci95_low": lo,
        "ci95_high": hi,
        "significant_at_05": not (lo <= 0 <= hi),
        "align": "window_id",
    }


def paired_bootstrap_acc(sa: List[Dict], sb: List[Dict], n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    by_a = {_key(r): r for r in sa if isinstance(r, dict)}
    by_b = {_key(r): r for r in sb if isinstance(r, dict)}
    # Prefer keys with document_hash; fall back to (task, sample_id)
    if not (set(by_a) & set(by_b)):
        by_a = {(r.get("task"), str(r.get("sample_id"))): r for r in sa}
        by_b = {(r.get("task"), str(r.get("sample_id"))): r for r in sb}
    common = sorted(set(by_a) & set(by_b), key=lambda x: str(x))
    if not common:
        return {"error": "no_common_samples", "n_a": len(sa), "n_b": len(sb)}
    rng = random.Random(seed)
    deltas = []
    macros = []
    for _ in range(n):
        sample = [common[rng.randint(0, len(common) - 1)] for _ in range(len(common))]
        acc_a = sum(1 for s in sample if by_a[s].get("correct")) / len(sample)
        acc_b = sum(1 for s in sample if by_b[s].get("correct")) / len(sample)
        deltas.append(acc_b - acc_a)
    deltas.sort()
    lo, hi = deltas[int(0.025 * n)], deltas[int(0.975 * n)]
    return {
        "n_bootstrap": n,
        "n_samples": len(common),
        "mean_delta_acc": sum(deltas) / n,
        "ci95_low": lo,
        "ci95_high": hi,
        "significant_at_05": not (lo <= 0 <= hi),
        "align": "task+sample_id(+document_hash)",
    }


def mcnemar(sa: List[Dict], sb: List[Dict]) -> Dict[str, Any]:
    by_a = {_key(r): r for r in sa}
    by_b = {_key(r): r for r in sb}
    if not (set(by_a) & set(by_b)):
        by_a = {(r.get("task"), str(r.get("sample_id"))): r for r in sa}
        by_b = {(r.get("task"), str(r.get("sample_id"))): r for r in sb}
    common = sorted(set(by_a) & set(by_b), key=lambda x: str(x))
    b01 = b10 = 0
    for s in common:
        ca, cb = bool(by_a[s].get("correct")), bool(by_b[s].get("correct"))
        if ca and not cb:
            b10 += 1
        elif cb and not ca:
            b01 += 1
    n = b01 + b10
    if n == 0:
        return {"p_value": 1.0, "b01": 0, "b10": 0, "n_common": len(common)}
    from math import comb
    p = sum(comb(n, k) * (0.5 ** n) for k in range(b01, n + 1))
    return {"b01": b01, "b10": b10, "p_value": min(1.0, 2 * p), "n_discordant": n, "n_common": len(common)}


def holm(pvalues: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
    m = len(pvalues)
    out = []
    for i, (name, p) in enumerate(sorted(pvalues, key=lambda x: x[1])):
        adj = min(1.0, p * (m - i))
        out.append({"test": name, "p_raw": p, "p_holm": adj, "reject_at_05": adj < 0.05})
    return out


def run_point(point: str, out: Path) -> None:
    rows_ppl = []
    for ds, label in [("wikitext2", "Wiki"), ("c4", "C4")]:
        wa = _ppl_windows(point, "uniform", ds)
        wb = _ppl_windows(point, "fdp", ds)
        stat = paired_bootstrap_ppl(wa, wb)
        stat.update({"point": point, "dataset": ds, "comparison": f"Uniform vs FDP {label}", "method_a": "uniform", "method_b": "fdp"})
        rows_ppl.append(stat)

    task_rows, mc_rows, pvals = [], [], []
    for task in TASKS:
        ra = _zs_root(point, "uniform", task)
        rb = _zs_root(point, "fdp", task)
        sa = load_zs_samples_dir(ra, task) if ra else []
        sb = load_zs_samples_dir(rb, task) if rb else []
        bs = paired_bootstrap_acc(sa, sb)
        mc = mcnemar(sa, sb)
        bs["task"] = task
        bs["point"] = point
        task_rows.append(bs)
        mc_rows.append({"task": task, "point": point, **mc})
        if "p_value" in mc:
            pvals.append((f"{point}:{task}", mc["p_value"]))

    # macro bootstrap: average per-task deltas if all tasks ok
    macro = {"point": point, "status": "partial"}
    ok_tasks = [t for t in task_rows if "mean_delta_acc" in t]
    if len(ok_tasks) == 5:
        # recompute joint macro on concatenated samples
        all_a, all_b = [], []
        for task in TASKS:
            ra = _zs_root(point, "uniform", task)
            rb = _zs_root(point, "fdp", task)
            sa = load_zs_samples_dir(ra, task) if ra else []
            sb = load_zs_samples_dir(rb, task) if rb else []
            all_a.extend(sa)
            all_b.extend(sb)
        macro = {"point": point, **paired_bootstrap_acc(all_a, all_b), "status": "ok"}

    holm_rows = holm(pvals)
    atomic_write_json(out / f"PPL_BOOTSTRAP_{point}.json", {"generated_at": now_iso(), "rows": rows_ppl})
    atomic_write_json(out / f"TASK_BOOTSTRAP_{point}.json", {"generated_at": now_iso(), "rows": task_rows, "macro": macro})
    atomic_write_json(out / f"MCNEMAR_{point}.json", {"generated_at": now_iso(), "rows": mc_rows})
    atomic_write_json(out / f"HOLM_{point}.json", {"generated_at": now_iso(), "rows": holm_rows})

    with (out / f"PPL_BOOTSTRAP_{point}.csv").open("w", newline="", encoding="utf-8") as f:
        if rows_ppl:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows_ppl for k in r}))
            w.writeheader()
            w.writerows(rows_ppl)
    with (out / f"HOLM_{point}.csv").open("w", newline="", encoding="utf-8") as f:
        if holm_rows:
            w = csv.DictWriter(f, fieldnames=list(holm_rows[0].keys()))
            w.writeheader()
            w.writerows(holm_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", required=True, choices=["mild", "moderate", "aggressive", "all"])
    args = ap.parse_args()
    out = HERE / "artifacts" / "statistics"
    out.mkdir(parents=True, exist_ok=True)
    points = ["mild", "moderate", "aggressive"] if args.point == "all" else [args.point]
    for p in points:
        run_point(p, out)
    print(json.dumps({"ok": True, "points": points}, ensure_ascii=False))


if __name__ == "__main__":
    main()
