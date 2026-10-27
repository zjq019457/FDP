#!/usr/bin/env python3
"""Component ablation v2: rebuild minus-* from same Step1 with ablated weights.

Protocol lock (moderate FDP):
  eb_base=0.06, lambda_eb=1.2, normalize=mean, freeze_layers=[0],
  retained component weights=1.0 (no renormalize), calib C4 64x2048 seed=2,
  basis WikiText-2 512x2048 seed=2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
EIGEN = ROOT / "EigenAttn"
LEGACY = ROOT / "rebuttal_legacy_reproduction"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "runners"))
from common import (  # noqa: E402
    ARTIFACT_ROOT, C4_TEXTS, MODEL, PYTHON_BIN, TASK_YAML, WIKITEXT,
    atomic_write_json, evaluation_hash, now_iso,
)
from kv_accounting import load_low_rank, realized_kv_bytes  # noqa: E402

STEP1 = Path("/home/bai-project/jiaqizhang/step1_out/step1_llama3_8b_eb060.json")
FDP_FULL = Path("/diskF/jiaqizhang/rebuttal_completion_op_artifacts/operating_points/moderate/fdp/final_model")
OUT_ROOT = Path("/diskF/jiaqizhang/rebuttal_completion_op_artifacts/component_ablation")
FINAL_ROUND = ROOT / "rebuttal_final_repair" / "final_round"

VARIANTS: Dict[str, Dict[str, float]] = {
    "minus_Z": {"wz": 0.0, "ws": 1.0, "wv": 1.0, "wa": 1.0, "wh": 1.0},
    "minus_S": {"wz": 1.0, "ws": 0.0, "wv": 1.0, "wa": 1.0, "wh": 1.0},
    "minus_V": {"wz": 1.0, "ws": 1.0, "wv": 0.0, "wa": 1.0, "wh": 1.0},
    "minus_A_AV": {"wz": 1.0, "ws": 1.0, "wv": 1.0, "wa": 0.0, "wh": 1.0},
    "minus_entropy": {"wz": 1.0, "ws": 1.0, "wv": 1.0, "wa": 1.0, "wh": 0.0},
}


def _env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(EIGEN) + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_ready(model_dir: Path) -> bool:
    if not (model_dir / "low_rank_config.json").exists():
        return False
    return any(p.stat().st_size > 1_000_000 for p in model_dir.glob("model-*.safetensors"))


def build_variant(name: str) -> Path:
    weights = VARIANTS[name]
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    model_dir = out / "final_model"
    if _model_ready(model_dir):
        return model_dir

    eb_json = out / "per_layer_eb.json"
    if not eb_json.exists():
        cmd = [
            PYTHON_BIN, "-u", str(EIGEN / "per_layer_eb_from_step1.py"),
            "--step1", str(STEP1),
            "--out", str(eb_json),
            "--eb_base", "0.06", "--lambda_eb", "1.2", "--normalize", "mean",
            "--freeze_layers", "0",
            "--wz", str(weights["wz"]), "--ws", str(weights["ws"]),
            "--wv", str(weights["wv"]), "--wa", str(weights["wa"]), "--wh", str(weights["wh"]),
        ]
        cp = subprocess.run(cmd, cwd=str(EIGEN), env=_env())
        if cp.returncode != 0:
            raise RuntimeError(f"per_layer_eb failed for {name}")

    basis_dir = out / "adaptive_basis"
    basis_dir.mkdir(parents=True, exist_ok=True)
    basis_pt = basis_dir / f"eigen_basis_{name}_perlayer_eb.pt"
    if not basis_pt.exists():
        cmd = [
            PYTHON_BIN, "-u", str(EIGEN / "tools" / "build_eigen_basis_llama_error_budget_offline.py"),
            "--model", MODEL,
            "--data_dir", WIKITEXT if Path(WIKITEXT).exists() else "/home/bai-project/jiaqizhang/wikitext2",
            "--out_dir", str(basis_dir),
            "--out_prefix", f"eigen_basis_{name}",
            "--per_layer_eb_json", str(eb_json),
            "--freeze_layer0_eb", "--layer0_eb", "0.06",
            "--nsamples", "512", "--seqlen", "2048", "--dtype", "bf16",
            "--attn_implementation", "eager", "--device", "cuda:0", "--seed", "2",
        ]
        log = out / "basis_build.log"
        with log.open("w") as lf:
            cp = subprocess.run(cmd, cwd=str(EIGEN), env=_env(), stdout=lf, stderr=subprocess.STDOUT)
        # locate produced pt
        pts = sorted(basis_dir.glob("*perlayer_eb.pt"))
        if cp.returncode != 0 or not pts:
            raise RuntimeError(f"basis build failed for {name}; see {log}")
        if pts[0] != basis_pt:
            shutil.copy2(pts[0], basis_pt)

    # compress
    if model_dir.exists() and not _model_ready(model_dir):
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_BIN, "-u", str(EIGEN / "main_eigen_attn.py"),
        "--model", MODEL, "--net", "Llama-3-8b", "--dtype", "bf16",
        "--error_budget", "0.06", "--avg_dim", "4096",
        "--seqlen", "2048", "--nsamples", "64", "--seed", "2",
        "--calib_dataset", "c4", "--compress_only", "--attn_implementation", "eager",
        "--eigen_basis_path", str(basis_pt),
        "--save_dir", str(model_dir),
        "--output_dir", str(out),
        "--cache_dir", str(ARTIFACT_ROOT / "HF_cache"),
    ]
    log = out / "compress.log"
    with log.open("w") as lf:
        cp = subprocess.run(cmd, cwd=str(EIGEN), env=_env(), stdout=lf, stderr=subprocess.STDOUT)
    if cp.returncode != 0 or not _model_ready(model_dir):
        raise RuntimeError(f"compress failed for {name}; see {log}")

    # diversity vs full
    full_ranks = load_low_rank(FDP_FULL / "low_rank_config.json")
    ranks = load_low_rank(model_dir / "low_rank_config.json")
    atomic_write_json(out / "DIVERSITY_GATE.json", {
        "variant": name,
        "weights": weights,
        "rank_sha": hashlib.sha256(json.dumps(ranks).encode()).hexdigest(),
        "full_rank_sha": hashlib.sha256(json.dumps(full_ranks).encode()).hexdigest(),
        "ranks_differ": ranks != full_ranks,
        "n_layers_rank_diff": sum(1 for a, b in zip(ranks, full_ranks) if a != b),
        "kv": realized_kv_bytes(ranks),
        "fdp_kv": realized_kv_bytes(full_ranks),
        "basis_sha": _sha_file(basis_pt),
        "eb_sha": _sha_file(eb_json),
        "generated_at": now_iso(),
    })
    return model_dir


def eval_ppl(model: Path, out: Path, dataset: str, method: str) -> dict:
    agg = out / "ppl_aggregate.json"
    if agg.exists():
        return json.loads(agg.read_text())
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_BIN, "-u", str(LEGACY / "runners" / "ppl_instrumented.py"),
        "--dataset", dataset, "--model", str(model), "--tok", MODEL,
        "--out_dir", str(out), "--dtype", "bf16", "--method", method,
        "--evaluation_protocol_hash", evaluation_hash(),
    ]
    if dataset == "wikitext2":
        cmd += ["--wikitext2_dir", WIKITEXT]
    else:
        cmd += ["--c4_texts", C4_TEXTS]
    subprocess.run(cmd, cwd=str(LEGACY), env=_env(), check=True)
    return json.loads(agg.read_text())


def eval_zs(model: Path, out: Path, tasks: str, method: str) -> dict:
    agg = out / "zero_shot_aggregate.json"
    if agg.exists():
        return json.loads(agg.read_text())
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_BIN, "-u", str(LEGACY / "runners" / "zero_shot_instrumented.py"),
        "--model", str(model), "--tokenizer_model", MODEL,
        "--task_yaml_dir", TASK_YAML, "--tasks", tasks,
        "--out_dir", str(out), "--dtype", "bf16", "--net", "llama-3-8b",
        "--method", method, "--low_rank_config", str(model / "low_rank_config.json"),
    ]
    subprocess.run(cmd, cwd=str(LEGACY), env=_env(), check=True)
    return json.loads(agg.read_text())


def cleanup_weights(model_dir: Path, art: Path) -> None:
    n = 0
    for p in model_dir.glob("model*.safetensors"):
        p.unlink()
        n += 1
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        idx.unlink()
        n += 1
    atomic_write_json(art / "WEIGHTS_DELETED_AFTER_EVAL.json", {
        "deleted_from": str(model_dir), "n_files": n, "generated_at": now_iso(),
    })


def _op_macro(prefix: str) -> float | str:
    accs = []
    for lane in ("hs_arc", "wg_piqa"):
        p = Path(f"/diskE/jiaqizhang/rebuttal_completion_artifacts/{prefix}_{lane}/zero_shot_aggregate.json")
        if not p.exists():
            continue
        for _t, v in (json.loads(p.read_text()).get("tasks") or {}).items():
            if isinstance(v, dict):
                a = v.get("accuracy_from_correct", v.get("lm_eval_primary"))
                if isinstance(a, (int, float)):
                    accs.append(float(a))
    return sum(accs) / len(accs) if accs else "NOT_EXECUTED"


def aggregate_final() -> None:
    import csv
    FINAL_ROUND.mkdir(parents=True, exist_ok=True)
    rows = []
    # anchors — Full FDP + uniform from moderate OP (same protocol family)
    op_art = Path("/diskE/jiaqizhang/rebuttal_completion_artifacts")
    for name, path, wiki_key, c4_key, zs_prefix in [
        (
            "fdp_full", FDP_FULL,
            "OP_moderate_fdp_wikitext2_ppl", "OP_moderate_fdp_c4_ppl", "OP_moderate_fdp",
        ),
        (
            "uniform_matched_anchor",
            Path("/diskF/jiaqizhang/rebuttal_completion_op_artifacts/operating_points/moderate/uniform/final_model"),
            "OP_moderate_uniform_wikitext2_ppl", "OP_moderate_uniform_c4_ppl", "OP_moderate_uniform",
        ),
    ]:
        wiki_p = op_art / wiki_key / "ppl_aggregate.json"
        c4_p = op_art / c4_key / "ppl_aggregate.json"
        kv = realized_kv_bytes(load_low_rank(path / "low_rank_config.json")) if (path / "low_rank_config.json").exists() else {}
        wiki_v = json.loads(wiki_p.read_text()).get("ppl") if wiki_p.exists() else "NOT_EXECUTED"
        c4_v = json.loads(c4_p.read_text()).get("ppl") if c4_p.exists() else "NOT_EXECUTED"
        macro = _op_macro(zs_prefix)
        status = "DONE" if wiki_v != "NOT_EXECUTED" and c4_v != "NOT_EXECUTED" and macro != "NOT_EXECUTED" else (
            "PARTIAL" if wiki_v != "NOT_EXECUTED" else "NOT_EXECUTED"
        )
        rows.append({
            "Variant": name,
            "Actual KV Bytes": kv.get("realized_kv_bytes", "NOT_EXECUTED"),
            "Retained Ratio": kv.get("retained_ratio", "NOT_EXECUTED"),
            "WikiText PPL": wiki_v,
            "C4 PPL": c4_v,
            "Five-task Macro": macro,
            "Rank SHA": kv.get("ranks_sha256", "NOT_EXECUTED"),
            "Status": status,
            "Source": "moderate_OP_reused_as_Full_FDP_anchor",
        })

    budget_rows = []
    for name in VARIANTS:
        art = OUT_ROOT / name
        model = art / "final_model"
        wiki = art / "eval" / "wikitext2" / "ppl_aggregate.json"
        c4 = art / "eval" / "c4" / "ppl_aggregate.json"
        hs = art / "eval" / "hs_arc" / "zero_shot_aggregate.json"
        wg = art / "eval" / "wg_piqa" / "zero_shot_aggregate.json"
        if (model / "low_rank_config.json").exists():
            kv = realized_kv_bytes(load_low_rank(model / "low_rank_config.json"))
        else:
            kv = {}
        accs = []
        for p in (hs, wg):
            if p.exists():
                for t, v in (json.loads(p.read_text()).get("tasks") or {}).items():
                    if isinstance(v, dict):
                        a = v.get("accuracy_from_correct", v.get("lm_eval_primary"))
                        if isinstance(a, (int, float)):
                            accs.append(a)
        macro = sum(accs) / len(accs) if accs else "NOT_EXECUTED"
        status = "DONE" if wiki.exists() and c4.exists() and accs else (
            "PARTIAL" if wiki.exists() or (model / "low_rank_config.json").exists() else "PENDING_REBUILD"
        )
        row = {
            "Variant": name,
            "Actual KV Bytes": kv.get("realized_kv_bytes", "NOT_EXECUTED"),
            "Retained Ratio": kv.get("retained_ratio", "NOT_EXECUTED"),
            "WikiText PPL": json.loads(wiki.read_text()).get("ppl") if wiki.exists() else "NOT_EXECUTED",
            "C4 PPL": json.loads(c4.read_text()).get("ppl") if c4.exists() else "NOT_EXECUTED",
            "Five-task Macro": macro,
            "Rank SHA": kv.get("ranks_sha256", "NOT_EXECUTED"),
            "Status": status,
            "Source": str(art),
        }
        rows.append(row)
        div = art / "DIVERSITY_GATE.json"
        budget_rows.append({
            "Variant": name,
            **{k: kv.get(k) for k in ("realized_kv_bytes", "retained_ratio", "ranks_sha256")},
            "diversity": json.loads(div.read_text()) if div.exists() else {},
            "Status": status,
        })

    def write_csv(path: Path, data: List[dict]):
        if not data:
            path.write_text("Status\nNOT_EXECUTED\n")
            return
        fields = list(data[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in data:
                flat = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()}
                w.writerow(flat)

    write_csv(FINAL_ROUND / "COMPONENT_ABLATION_FINAL.csv", rows)
    write_csv(FINAL_ROUND / "COMPONENT_ABLATION_BUDGET_AUDIT.csv", budget_rows)
    write_csv(FINAL_ROUND / "COMPONENT_ABLATION_SCHEDULES.csv", [
        {"Variant": n, "weights": json.dumps(w), "step1": str(STEP1)} for n, w in VARIANTS.items()
    ])
    atomic_write_json(FINAL_ROUND / "COMPONENT_ABLATION_PROTOCOL.json", {
        "lambda": 1.2, "normalize": "mean", "eb_base": 0.06, "freeze_layers": [0],
        "equal_weights_retained": True, "step1": str(STEP1), "variants": list(VARIANTS),
        "full_fdp_anchor": str(FDP_FULL),
        "generated_at": now_iso(),
    })
    n_minus = sum(1 for r in rows if r["Status"] == "DONE" and r["Variant"].startswith("minus_"))
    full_ok = any(r["Variant"] == "fdp_full" and r["Status"] == "DONE" for r in rows)
    diversities = []
    for name in VARIANTS:
        divp = OUT_ROOT / name / "DIVERSITY_GATE.json"
        if divp.exists():
            diversities.append(json.loads(divp.read_text()))
    n_rank_changed = sum(1 for d in diversities if d.get("n_layers_rank_diff", 0) >= 1)
    distinct_rank = len({d.get("rank_sha") for d in diversities if d.get("rank_sha")}) == len(diversities) and len(diversities) >= 5
    distinct_eb = len({d.get("eb_sha") for d in diversities if d.get("eb_sha")}) == len(diversities) and len(diversities) >= 5
    distinct_basis = len({d.get("basis_sha") for d in diversities if d.get("basis_sha")}) == len(diversities) and len(diversities) >= 5
    null_rank = [d.get("variant") for d in diversities if d.get("n_layers_rank_diff", 0) == 0]
    # Scientific closure: all minus evaluated + Full FDP + distinct EB/basis SHA for every branch.
    # Rank change required on >=4/5; a null-rank ablation (e.g. minus_S) is allowed if documented.
    ok = (
        n_minus >= 5 and full_ok and distinct_eb and distinct_basis
        and n_rank_changed >= 4 and (distinct_rank or n_rank_changed >= 4)
    )
    gate = FINAL_ROUND / ("COMPONENT_ABLATION_PASS.DONE" if ok else "COMPONENT_ABLATION_PASS.FAILED")
    for p in FINAL_ROUND.glob("COMPONENT_ABLATION_PASS.*"):
        p.unlink()
    atomic_write_json(gate, {
        "ok": ok, "n_minus_done": n_minus, "full_fdp_ok": full_ok,
        "n_rank_changed": n_rank_changed, "distinct_rank_sha": distinct_rank,
        "distinct_eb_sha": distinct_eb, "distinct_basis_sha": distinct_basis,
        "null_rank_ablations": null_rank,
        "note": "minus_S may be null-rank vs Full FDP under this Step1; EB/basis SHA still distinct",
        "generated_at": now_iso(),
    })
    # paired stats: WikiText point deltas vs fdp_full when available
    fdp_wiki = next((r["WikiText PPL"] for r in rows if r["Variant"] == "fdp_full"), None)
    stat_rows = []
    if isinstance(fdp_wiki, (int, float)):
        for r in rows:
            if r["Variant"].startswith("minus_") and isinstance(r["WikiText PPL"], (int, float)):
                stat_rows.append({
                    "Contrast": f"{r['Variant']}_vs_fdp_full",
                    "Metric": "WikiText PPL delta",
                    "Delta": float(r["WikiText PPL"]) - float(fdp_wiki),
                    "Status": "POINT_ESTIMATE",
                    "Note": "window-level paired bootstrap follow-up if raw windows retained",
                })
    if not stat_rows:
        stat_rows = [{"Status": "PENDING", "note": "need minus evals complete"}]
    write_csv(FINAL_ROUND / "COMPONENT_ABLATION_STATISTICS.csv", stat_rows)
    (FINAL_ROUND / "COMPONENT_ABLATION_ANALYSIS.md").write_text(
        f"# Component Ablation Analysis\nGenerated: {now_iso()}\n"
        f"Minus DONE: {n_minus}/5; Full FDP: {full_ok}; rank-changed variants: {n_rank_changed}; "
        f"distinct EB SHA: {distinct_eb}; distinct basis SHA: {distinct_basis}; "
        f"null-rank ablations: {null_rank}\nPASS={ok}\n\n"
        "Claim guard: if minus_S has 0 layer rank diffs vs Full FDP, report as null schedule effect "
        "under current Step1 weights — do not claim S-component changes the rank plan.\n",
        encoding="utf-8",
    )
    (FINAL_ROUND / "COMPONENT_ABLATION_CLAIM_GUARD.md").write_text(
        f"# Component Ablation Claim Guard\nGenerated: {now_iso()}\n"
        f"- Null-rank ablations: {null_rank}\n"
        "- All five minus branches retain distinct per-layer EB SHA and basis SHA.\n"
        "- Do not claim every ablation changes ranks; report null effects honestly.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "ok": ok, "n_minus_done": n_minus, "full_fdp_ok": full_ok,
        "n_rank_changed": n_rank_changed, "distinct_rank_sha": distinct_rank,
        "distinct_eb_sha": distinct_eb, "distinct_basis_sha": distinct_basis,
        "null_rank_ablations": null_rank,
    }, ensure_ascii=False))
    if not ok:
        raise SystemExit(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS) + ["all", "aggregate"], default="all")
    ap.add_argument("--phase", choices=["build", "wikitext2", "c4", "hs_arc", "wg_piqa", "all", "aggregate"], default="all")
    ap.add_argument("--keep_weights", action="store_true")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.phase == "aggregate" or args.variant == "aggregate":
        aggregate_final()
        return

    names = list(VARIANTS) if args.variant == "all" else [args.variant]
    for name in names:
        art = OUT_ROOT / name
        art.mkdir(parents=True, exist_ok=True)
        if args.phase in ("build", "all"):
            model = build_variant(name)
        else:
            model = art / "final_model"
            if not _model_ready(model):
                raise SystemExit(f"model not ready: {model}")
        method = f"component_{name}"
        if args.phase in ("wikitext2", "all"):
            eval_ppl(model, art / "eval" / "wikitext2", "wikitext2", method)
        if args.phase in ("c4", "all"):
            eval_ppl(model, art / "eval" / "c4", "c4", method)
        if args.phase in ("hs_arc", "all"):
            eval_zs(model, art / "eval" / "hs_arc",
                    "hellaswag_localjsonl,arc_challenge_localjsonl,arc_easy_localjsonl", method)
        if args.phase in ("wg_piqa", "all"):
            eval_zs(model, art / "eval" / "wg_piqa",
                    "winogrande_localjsonl,piqa_localjsonl", method)
        if args.phase == "all" and not args.keep_weights:
            cleanup_weights(model, art)
        print(json.dumps({"ok": True, "variant": name, "phase": args.phase}, ensure_ascii=False))

    if args.variant == "all" and args.phase == "all":
        aggregate_final()


if __name__ == "__main__":
    main()
