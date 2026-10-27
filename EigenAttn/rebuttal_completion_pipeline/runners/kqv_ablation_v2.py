#!/usr/bin/env python3
"""KQ/V ablation via truncated-basis compress (diversity gated)."""
from __future__ import annotations

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
LEGACY = ROOT / "rebuttal_legacy_reproduction"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "runners"))
from common import MODEL, PYTHON_BIN, WIKITEXT, ARTIFACT_ROOT, artifact_dir, atomic_write_json, evaluation_hash, now_iso  # noqa: E402
from kv_accounting import load_low_rank, realized_kv_bytes  # noqa: E402
from truncate_basis_to_ranks import truncate_basis  # noqa: E402

FDP = Path("/diskF/jiaqizhang/rebuttal_completion_op_artifacts/operating_points/moderate/fdp/final_model")
UNI = Path("/diskF/jiaqizhang/rebuttal_completion_op_artifacts/operating_points/moderate/uniform/final_model")
FDP_BASIS = Path(
    "/diskF/jiaqizhang/rebuttal_completion_op_artifacts/operating_points/moderate/fdp/"
    "adaptive_basis/eigen_basis_moderate_perlayer_eb.pt"
)
UNI_BASIS = Path("/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_eb/eigen_basis_llama3_8b_wikitext2_eb0p06.pt")
OUT_MODELS = Path("/diskF/jiaqizhang/rebuttal_completion_op_artifacts/kqv_v2_models")


def _round8(x: int) -> int:
    x = max(8, int(x))
    return (x // 8) * 8


def _sha(pairs: List[Tuple[int, int]]) -> str:
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()


def _make_branches() -> Dict[str, List[Tuple[int, int]]]:
    fdp = load_low_rank(FDP / "low_rank_config.json")
    uni = load_low_rank(UNI / "low_rank_config.json")
    target = realized_kv_bytes(fdp)["realized_kv_bytes"]

    def clip_scale(pairs):
        # clip to FDP capacity then scale toward target bytes
        clipped = [(_round8(min(a, fa)), _round8(min(b, fb))) for (a, b), (fa, fb) in zip(pairs, fdp)]
        cur = realized_kv_bytes(clipped)["realized_kv_bytes"]
        ratio = target / max(cur, 1)
        out = []
        for (a, b), (fa, fb) in zip(clipped, fdp):
            out.append((_round8(min(fa, int(a * ratio))), _round8(min(fb, int(b * ratio)))))
        return out

    return {
        "fdp_full": fdp,
        "fdp_kq_adaptive_v_uniform": clip_scale([(fk, uv) for (fk, fv), (uk, uv) in zip(fdp, uni)]),
        "fdp_kq_uniform_v_adaptive": clip_scale([(uk, fv) for (fk, fv), (uk, uv) in zip(fdp, uni)]),
        "kq_path_only": clip_scale([(fk, _round8(sum(u[1] for u in uni) // len(uni))) for (fk, fv) in fdp]),
        "v_path_only": clip_scale([(_round8(sum(u[0] for u in uni) // len(uni)), fv) for (fk, fv) in fdp]),
        "uniform": uni,
    }


def _compress(name: str, pairs: List[Tuple[int, int]], basis_src: Path, work: Path) -> Path:
    out_model = OUT_MODELS / name / "final_model"
    if (out_model / "low_rank_config.json").exists() and any(out_model.glob("model-*.safetensors")):
        return out_model
    work.mkdir(parents=True, exist_ok=True)
    basis_dst = work / f"{name}_basis.pt"
    truncate_basis(basis_src, pairs, basis_dst)
    if out_model.exists():
        shutil.rmtree(out_model)
    out_model.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "EigenAttn") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    cmd = [
        PYTHON_BIN, "-u", str(ROOT / "EigenAttn" / "main_eigen_attn.py"),
        "--model", MODEL, "--net", "Llama-3-8b", "--dtype", "bf16",
        "--error_budget", "0.06", "--avg_dim", "4096",
        "--seqlen", "2048", "--nsamples", "64", "--seed", "2",
        "--calib_dataset", "c4", "--compress_only", "--attn_implementation", "eager",
        "--eigen_basis_path", str(basis_dst),
        "--save_dir", str(out_model),
        "--output_dir", str(out_model.parent),
        "--cache_dir", str(ARTIFACT_ROOT / "HF_cache"),
    ]
    with (work / f"{name}_stdout.log").open("w") as fo, (work / f"{name}_stderr.log").open("w") as fe:
        cp = subprocess.run(cmd, cwd=str(ROOT / "EigenAttn"), env=env, stdout=fo, stderr=fe, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"compress {name} failed; see {work}/{name}_stderr.log")
    atomic_write_json(out_model / "low_rank_config.json", [[a, b] for a, b in pairs])
    return out_model


def _eval_ppl(model: Path, out: Path, method: str) -> float:
    if (out / "ppl_aggregate.json").exists():
        return float(json.loads((out / "ppl_aggregate.json").read_text())["ppl"])
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "EigenAttn") + os.pathsep + str(LEGACY) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        PYTHON_BIN, "-u", str(LEGACY / "runners" / "ppl_instrumented.py"),
        "--dataset", "wikitext2", "--model", str(model), "--tok", MODEL,
        "--out_dir", str(out), "--wikitext2_dir", WIKITEXT,
        "--dtype", "bf16", "--method", method,
        "--evaluation_protocol_hash", evaluation_hash(),
    ]
    subprocess.run(cmd, cwd=str(LEGACY), env=env, check=True)
    return float(json.loads((out / "ppl_aggregate.json").read_text())["ppl"])


def main() -> None:
    out_root = artifact_dir("kqv_ablation_v2", link_under=HERE / "artifacts" / "kqv_ablation_v2")
    branches = _make_branches()
    shas = {k: _sha(v) for k, v in branches.items()}
    diffs = {
        k: sum(1 for a, b in zip(branches["fdp_full"], pairs) if a != b)
        for k, pairs in branches.items() if k != "fdp_full"
    }
    gate = {"ok": len(set(shas.values())) >= 5 and all(v >= 1 for k, v in diffs.items() if k != "uniform"),
            "rank_plan_shas": shas, "layer_diffs_vs_fdp": diffs}
    atomic_write_json(out_root / "DIVERSITY_GATE.json", {"generated_at": now_iso(), **gate})
    if not gate["ok"]:
        raise SystemExit(f"diversity gate failed: {gate}")

    results = []
    work = out_root / "compress_work"
    for name, pairs in branches.items():
        basis = UNI_BASIS if name == "uniform" else FDP_BASIS
        # uniform branch can reuse existing UNI model
        if name == "uniform":
            model = UNI
        elif name == "fdp_full":
            model = FDP
        else:
            model = _compress(name, pairs, basis, work)
        kv = realized_kv_bytes(pairs)
        branch_out = out_root / name
        atomic_write_json(branch_out / "BRANCH_MANIFEST.json", {
            "branch": name, "model": str(model), "rank_plan_sha": _sha(pairs),
            "kv": kv, "generated_at": now_iso(),
        })
        ppl = _eval_ppl(model, branch_out / "wikitext2_ppl", f"kqv_{name}")
        results.append({
            "branch": name, "wikitext2_ppl": ppl,
            "realized_kv_bytes": kv["realized_kv_bytes"],
            "rank_plan_sha": _sha(pairs), "model": str(model),
        })
        # delete intermediate compress weights except keep configs
        if name not in ("uniform", "fdp_full"):
            for p in Path(model).glob("model*.safetensors"):
                p.unlink(missing_ok=True)

    ppls = {r["branch"]: r["wikitext2_ppl"] for r in results if r["branch"] != "uniform"}
    identical = len(set(ppls.values())) == 1
    atomic_write_json(out_root / "POST_EVAL_GATE.json", {
        "identical_ppl_failure": identical, "ppls": ppls, "pass": not identical,
    })
    import csv
    with (out_root / "KQ_V_RESULTS.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    if identical:
        raise SystemExit("POST_EVAL_GATE failed: identical PPL")
    print(json.dumps({"ok": True, "branches": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
