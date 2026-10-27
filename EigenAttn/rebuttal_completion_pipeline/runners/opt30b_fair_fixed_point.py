#!/usr/bin/env python3
"""OPT-30B fair fixed point: Dense + Uniform(eb026) + legacy FDP(FGeb026fpv6a) WikiText PPL."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
EIGEN = ROOT / "EigenAttn"
sys.path.insert(0, str(HERE))
from common import OPT30B_MODEL, PYTHON_BIN, atomic_write_json, now_iso  # noqa: E402

MODEL = OPT30B_MODEL
FDP_MODEL = Path("/diskE/jiaqizhang/FGeb026fpv6a/final_model")
OUT = Path("/diskF/jiaqizhang/rebuttal_completion_op_artifacts/opt30b_fair")
FINAL = ROOT / "rebuttal_final_repair" / "final_round"
WIKITEXT = "/home/bai-project/jiaqizhang/wikitext2"
SANITY = EIGEN / "sanity_AB_opt30b.py"


def _env():
    e = os.environ.copy()
    e["PYTHONPATH"] = str(EIGEN) + os.pathsep + e.get("PYTHONPATH", "")
    e["HF_HUB_OFFLINE"] = "1"
    e["TRANSFORMERS_OFFLINE"] = "1"
    e["WIKITEXT2_DATA_DIR"] = WIKITEXT
    e["PYTHONUNBUFFERED"] = "1"
    # OPTEigenAttention defaults EIGEN_VSCALE_INIT=0.03; force identity scale for compress-only.
    e.setdefault("EIGEN_VSCALE_INIT", "1.0")
    e.setdefault("EIGEN_VSCALE", "1.0")
    return e


def _parse_ppl(log: Path, which: str) -> float | None:
    if not log.exists():
        return None
    txt = log.read_text(encoding="utf-8", errors="ignore")
    # look for CMP PPL or Base/WikiText PPL lines
    import re
    pats = {
        "cmp": [r"CMP PPL[:\s]+([0-9.]+)", r"\[WikiText-2\].*CMP PPL[:\s]+([0-9.]+)"],
        "base": [r"Base PPL[:\s]+([0-9.]+)", r"\[WikiText-2\].*Base PPL[:\s]+([0-9.]+)", r"Dense PPL[:\s]+([0-9.]+)"],
    }
    for pat in pats[which]:
        ms = re.findall(pat, txt)
        if ms:
            return float(ms[-1])
    # fallback: any PPL:
    ms = re.findall(r"PPL[:\s]+([0-9.]+)", txt)
    if ms:
        return float(ms[-1])
    return None


def eval_fdp(limit: int | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "fdp_wiki.log"
    cmd = [
        PYTHON_BIN, str(SANITY),
        "--cmp_only",
        "--cmp", str(FDP_MODEL),
        "--low_json", str(FDP_MODEL / "low_rank_config.json"),
        "--tok", MODEL, "--dtype", "fp16", "--cmp_device", "cuda:0", "--wikitext2",
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    with log.open("w") as lf:
        cp = subprocess.run(cmd, cwd=str(EIGEN), env=_env(), stdout=lf, stderr=subprocess.STDOUT)
    ppl = _parse_ppl(log, "cmp")
    return {"method": "FDP_FGeb026fpv6a", "ppl": ppl, "rc": cp.returncode, "log": str(log)}


def eval_dense(limit: int | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "dense_wiki.log"
    cmd = [
        PYTHON_BIN, str(SANITY),
        "--base_only", "--base", MODEL, "--tok", MODEL,
        "--dtype", "fp16", "--base_device", "auto", "--wikitext2",
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    # prefer 2 GPUs if available via CUDA_VISIBLE_DEVICES already set by orchestrator
    with log.open("w") as lf:
        cp = subprocess.run(cmd, cwd=str(EIGEN), env=_env(), stdout=lf, stderr=subprocess.STDOUT)
    ppl = _parse_ppl(log, "base")
    return {"method": "Dense", "ppl": ppl, "rc": cp.returncode, "log": str(log)}


def build_uniform() -> Path:
    """Rebuild uniform eb026 basis + compress onto diskF."""
    uni_root = OUT / "uniform_eb026"
    uni_root.mkdir(parents=True, exist_ok=True)
    model_dir = uni_root / "final_model"
    if (model_dir / "low_rank_config.json").exists() and any(model_dir.glob("pytorch_model*")):
        return model_dir
    # also accept safetensors
    if (model_dir / "low_rank_config.json").exists() and any(model_dir.glob("model*.safetensors")):
        return model_dir

    basis_dir = uni_root / "basis"
    basis_dir.mkdir(parents=True, exist_ok=True)
    pts = list(basis_dir.glob("*.pt"))
    if not pts:
        cmd = [
            PYTHON_BIN, "-u", str(EIGEN / "tools" / "build_eigen_basis_opt_error_budget_offline.py"),
            "--model", MODEL,
            "--data_dir", WIKITEXT,
            "--out_dir", str(basis_dir),
            "--out_prefix", "eigen_basis_opt30b_uniform_eb026_wiki_strict",
            "--error_budgets", "0.026",
            "--nsamples", "512", "--seqlen", "2048",
            "--device", "cuda:0", "--dtype", "fp16", "--paper_defaults", "--verbose", "1",
        ]
        log = uni_root / "basis.log"
        with log.open("w") as lf:
            cp = subprocess.run(cmd, cwd=str(EIGEN), env=_env(), stdout=lf, stderr=subprocess.STDOUT)
        pts = list(basis_dir.glob("*.pt"))
        if cp.returncode != 0 or not pts:
            raise RuntimeError(f"uniform basis failed; see {log}")

    basis_pt = sorted(pts)[0]
    model_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_BIN, "-u", str(EIGEN / "main_eigen_attn.py"),
        "--model", MODEL, "--net", "opt-30b",
        "--eigen_basis_path", str(basis_pt),
        "--repair_basis_pt", str(basis_pt),
        "--output_dir", str(uni_root),
        "--save_dir", str(model_dir),
        "--compress_only", "--error_budget", "0.026", "--dtype", "fp16",
        "--calib_dataset", "wikitext2",
        "--wikitext2_data_dir", WIKITEXT,
        "--attn_implementation", "eager",
        "--force_vscale_one", "1",
    ]
    # single GPU compress if possible; multigpu if env has multiple
    log = uni_root / "compress.log"
    with log.open("w") as lf:
        cp = subprocess.run(cmd, cwd=str(EIGEN), env=_env(), stdout=lf, stderr=subprocess.STDOUT)
    if cp.returncode != 0:
        # retry with --multigpu if single failed
        cmd2 = cmd + ["--multigpu"]
        with log.open("a") as lf:
            lf.write("\n=== RETRY MULTIGPU ===\n")
            cp = subprocess.run(cmd2, cwd=str(EIGEN), env=_env(), stdout=lf, stderr=subprocess.STDOUT)
    if cp.returncode != 0 or not (model_dir / "low_rank_config.json").exists():
        raise RuntimeError(f"uniform compress failed; see {log}")
    return model_dir


def eval_uniform(model_dir: Path, limit: int | None = None) -> dict:
    log = OUT / "uniform_wiki.log"
    cmd = [
        PYTHON_BIN, str(SANITY),
        "--cmp_only", "--cmp", str(model_dir),
        "--tok", MODEL, "--dtype", "fp16", "--cmp_device", "cuda:0", "--wikitext2",
    ]
    if (model_dir / "low_rank_config.json").exists():
        cmd += ["--low_json", str(model_dir / "low_rank_config.json")]
    if limit:
        cmd += ["--limit", str(limit)]
    with log.open("w") as lf:
        cp = subprocess.run(cmd, cwd=str(EIGEN), env=_env(), stdout=lf, stderr=subprocess.STDOUT)
    ppl = _parse_ppl(log, "cmp")
    return {"method": "Uniform_eb026", "ppl": ppl, "rc": cp.returncode, "log": str(log), "model": str(model_dir)}


METHOD_ORDER = ("Dense", "Uniform_eb026", "FDP_FGeb026fpv6a")


def _load_prior_rows() -> dict:
    path = FINAL / "OPT30B_FIXED_POINT_RESULTS.csv"
    by = {}
    if not path.exists():
        return by
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by[r.get("Method", "")] = r
    return by


def write_results(rows: list, *, smoke: bool = False) -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    by = _load_prior_rows()
    for r in rows:
        if r.get("ppl") is not None and r.get("rc") == 0:
            st = "DONE"
        elif r.get("allow_pending") and r.get("rc") == 0:
            st = "PENDING"
        else:
            st = "FAILED"
        by[r["method"]] = {
            "Model": "OPT-30B",
            "Method": r["method"],
            "WikiText PPL": r.get("ppl") if r.get("ppl") is not None else "NOT_EXECUTED",
            "C4 PPL": r.get("c4_ppl", "NOT_EXECUTED"),
            "Five-task Macro": r.get("macro", "NOT_EXECUTED"),
            "Actual KV Bytes": r.get("kv_bytes", "NOT_EXECUTED"),
            "Status": st,
            "Log": r.get("log"),
            "ModelPath": r.get("model", ""),
            "Note": r.get("note", "C4 shard empty on host unless filled; ZS optional follow-up"),
        }
    out_rows = [by[m] for m in METHOD_ORDER if m in by] + [v for k, v in by.items() if k not in METHOD_ORDER]
    if not out_rows:
        out_rows = [{"Model": "OPT-30B", "Method": "NONE", "Status": "NOT_EXECUTED", "WikiText PPL": "NOT_EXECUTED",
                     "C4 PPL": "NOT_EXECUTED", "Five-task Macro": "NOT_EXECUTED", "Actual KV Bytes": "NOT_EXECUTED",
                     "Log": "", "ModelPath": "", "Note": "empty"}]
    path = FINAL / "OPT30B_FIXED_POINT_RESULTS.csv"
    fields = list(out_rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    # KV for FDP/Uniform from low_rank if present
    kv_note = {}
    try:
        from kv_accounting import load_low_rank, realized_kv_bytes  # type: ignore
        if (FDP_MODEL / "low_rank_config.json").exists():
            kv_note["fdp"] = realized_kv_bytes(load_low_rank(FDP_MODEL / "low_rank_config.json"))
        uni_lr = OUT / "uniform_eb026" / "final_model" / "low_rank_config.json"
        if uni_lr.exists():
            kv_note["uniform"] = realized_kv_bytes(load_low_rank(uni_lr))
    except Exception as e:  # noqa: BLE001
        kv_note["error"] = repr(e)
    atomic_write_json(FINAL / "OPT30B_FIXED_POINT_PROTOCOL.json", {
        "generated_at": now_iso(),
        "dense": MODEL,
        "fdp": str(FDP_MODEL),
        "uniform": "rebuild_eb026_wikitext_calib",
        "legacy_fdp_tag": "FGeb026fpv6a / frag_eb026_v6",
        "dtype": "fp16",
        "eval": "sanity_AB_opt30b --wikitext2",
        "kv_audit": kv_note,
        "smoke_is_not_formal": True,
    })
    formal = [r for r in out_rows if r.get("Method") in METHOD_ORDER]
    n_done = sum(1 for r in formal if r.get("Status") == "DONE")
    ok = (not smoke) and n_done >= 3
    for p in FINAL.glob("OPT30B_FIXED_POINT_PASS.*"):
        p.unlink()
    gate_name = "OPT30B_FIXED_POINT_PASS.DONE" if ok else "OPT30B_FIXED_POINT_PASS.FAILED"
    if smoke:
        gate_name = "OPT30B_FIXED_POINT_PASS.FAILED"  # never PASS on smoke alone
    atomic_write_json(FINAL / gate_name, {
        "ok": ok, "n_done": n_done, "smoke": smoke, "generated_at": now_iso(),
        "methods": [r.get("Method") for r in formal],
    })
    (FINAL / "OPT30B_CLAIM_GUARD.md").write_text(
        f"# OPT Claim Guard\nGenerated: {now_iso()}\n"
        f"- Fair Dense/Uniform/FDP Wiki points done: {ok} ({n_done}/3)\n"
        "- Smoke ≠ formal. Old single CMP PPL must not stand alone as FDP evidence.\n"
        "- C4: NOT_EXECUTED if c4-en empty; mark honestly.\n"
        f"- KV audit keys: {list(kv_note)}\n",
        encoding="utf-8",
    )
    # paired uncertainty placeholder from available logs
    (FINAL / "OPT30B_STATISTICS.csv").write_text(
        "Method,Metric,Value,Status\n"
        + "\n".join(
            f"{r.get('Method')},WikiText PPL,{r.get('WikiText PPL')},{r.get('Status')}" for r in formal
        )
        + "\n",
        encoding="utf-8",
    )
    # phase_ok: this invocation succeeded (partial triples must not fail the job)
    phase_ok = True
    for r in rows:
        if r.get("rc", 0) != 0:
            phase_ok = False
        elif r.get("ppl") is None and not r.get("allow_pending"):
            phase_ok = False
    print(json.dumps({
        "ok": phase_ok, "gate_ok": ok, "n_done": n_done, "smoke": smoke, "rows": out_rows,
    }, ensure_ascii=False))
    if smoke:
        return
    if not phase_ok:
        raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["smoke", "fdp", "dense", "uniform_build", "uniform_eval", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    rows = []
    if args.phase == "smoke":
        rows.append(eval_fdp(limit=args.limit or 2))
        rows[-1]["note"] = "SMOKE_ONLY_limit; not formal FDP evidence"
        write_results(rows, smoke=True)
        return
    if args.phase in ("fdp", "all"):
        r = eval_fdp(limit=args.limit)
        r["model"] = str(FDP_MODEL)
        r["note"] = "formal FDP FGeb026fpv6a full WikiText"
        rows.append(r)
    if args.phase in ("dense", "all"):
        r = eval_dense(limit=args.limit)
        r["model"] = MODEL
        r["note"] = "formal Dense full WikiText; device_map=auto"
        rows.append(r)
    uni_model = None
    if args.phase in ("uniform_build", "uniform_eval", "all"):
        uni_model = build_uniform()
        if args.phase == "uniform_build":
            write_results([{
                "method": "Uniform_eb026", "ppl": None, "rc": 0, "allow_pending": True,
                "log": str(OUT / "uniform_eb026" / "compress.log"),
                "model": str(uni_model), "note": "uniform built; eval pending",
            }])
            print(json.dumps({"ok": True, "phase": "uniform_build", "model": str(uni_model)}))
            return
    if args.phase in ("uniform_eval", "all"):
        r = eval_uniform(uni_model or (OUT / "uniform_eb026" / "final_model"), limit=args.limit)
        r["note"] = "formal Uniform eb026 matched window WikiText"
        rows.append(r)
    write_results(rows)


if __name__ == "__main__":
    main()
