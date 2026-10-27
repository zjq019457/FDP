#!/usr/bin/env python3
"""Evaluate operating point using legacy instrumented runners; reuse R2/R3 for mild."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
LEGACY = ROOT / "rebuttal_legacy_reproduction"
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    PYTHON_BIN,
    C4_TEXTS,
    MODEL,
    TASK_YAML,
    WIKITEXT,
    artifact_dir,
    op_artifact_dir,
    atomic_write_json,
    evaluation_hash,
)

PPL_RUNNER = LEGACY / "runners" / "ppl_instrumented.py"
ZS_RUNNER = LEGACY / "runners" / "zero_shot_instrumented.py"
LEGACY_ART = Path("/diskF/jiaqizhang/rebuttal_legacy_reproduction_artifacts")


def _model_paths(point: str) -> dict:
    if point == "mild":
        return {
            "uniform": str(LEGACY_ART / "r2_uniform_eb045" / "final_model"),
            "fdp": str(LEGACY_ART / "r3_fgeb045" / "final_model"),
        }
    op = op_artifact_dir(f"operating_points/{point}", link_under=HERE / "artifacts" / f"OP_{point}")
    return {
        "uniform": str(op / "uniform" / "final_model"),
        "fdp": str(op / "fdp" / "final_model"),
    }


def _reuse_legacy_ppl(point: str, meth: str, dataset: str) -> bool:
    if point != "mild":
        return False
    prefix = "R2_uniform" if meth == "uniform" else "R3_fdp"
    src_name = f"{prefix}_wikitext_ppl" if dataset == "wikitext2" else f"{prefix}_c4_ppl"
    src = LEGACY_ART / src_name
    if not (src / "ppl_aggregate.json").exists():
        return False
    tag = f"OP_{point}_{meth}"
    dst = artifact_dir(f"{tag}_{dataset}_ppl", link_under=HERE / "artifacts" / f"{tag}_{dataset}_ppl")
    for fname in ("ppl_aggregate.json", "ppl_windows.jsonl"):
        sp = src / fname
        if sp.exists():
            shutil.copy2(sp, dst / fname)
    atomic_write_json(dst / "REUSED_FROM_LEGACY.json", {"source": str(src)})
    return True


def _reuse_legacy_zs(point: str, meth: str) -> bool:
    if point != "mild":
        return False
    prefix = "R2_uniform" if meth == "uniform" else "R3_fdp"
    ok = True
    for suffix in ("zs_hs_arc", "zs_wg_piqa"):
        src = LEGACY_ART / f"{prefix}_{suffix}"
        if not src.exists():
            ok = False
            continue
        tag = f"OP_{point}_{meth}"
        dst = artifact_dir(f"{tag}_{suffix}", link_under=HERE / "artifacts" / f"{tag}_{suffix}")
        agg = src / "zero_shot_aggregate.json"
        if agg.exists():
            shutil.copy2(agg, dst / "zero_shot_aggregate.json")
        samples = src / "zero_shot_samples"
        if samples.is_dir():
            ds = dst / "zero_shot_samples"
            ds.mkdir(parents=True, exist_ok=True)
            for f in samples.glob("*.jsonl"):
                shutil.copy2(f, ds / f.name)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", required=True)
    ap.add_argument("--dataset", required=True, choices=["wikitext2", "c4", "zero_shot"])
    ap.add_argument("--method", default="both", choices=["uniform", "fdp", "both"])
    ap.add_argument(
        "--bundle",
        default="all",
        choices=["all", "hs_arc", "wg_piqa"],
        help="zero_shot task bundle; use hs_arc/wg_piqa to parallelize across GPUs",
    )
    args = ap.parse_args()
    paths = _model_paths(args.point)
    methods = ["uniform", "fdp"] if args.method == "both" else [args.method]
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(ROOT / "EigenAttn")
        + os.pathsep
        + str(LEGACY)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    # Prefer local evaluate modules; avoid Hub lookup for exact_match.
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("HF_EVALUATE_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

    bundles = [
        ("hellaswag_localjsonl,arc_challenge_localjsonl,arc_easy_localjsonl", "hs_arc"),
        ("winogrande_localjsonl,piqa_localjsonl", "wg_piqa"),
    ]
    if args.bundle != "all":
        bundles = [b for b in bundles if b[1] == args.bundle]

    for meth in methods:
        if args.dataset == "zero_shot":
            if args.bundle == "all" and _reuse_legacy_zs(args.point, meth):
                continue
            model = paths[meth]
            for tasks, suffix in bundles:
                tag = f"OP_{args.point}_{meth}"
                zs_out = str(artifact_dir(f"{tag}_{suffix}", link_under=HERE / "artifacts" / f"{tag}_{suffix}"))
                if (Path(zs_out) / "zero_shot_aggregate.json").exists():
                    continue
                cmd = [
                    PYTHON_BIN, str(ZS_RUNNER),
                    "--model", model, "--tokenizer_model", MODEL,
                    "--task_yaml_dir", TASK_YAML, "--tasks", tasks,
                    "--out_dir", zs_out, "--dtype", "bf16",
                    "--net", "llama-3-8b", "--method", f"{meth}_{args.point}",
                ]
                lr = Path(model) / "low_rank_config.json"
                if lr.exists():
                    # Prefer JSON path so LMClass does not torch.load Tensor .pt into GenerationConfig.
                    cmd.extend(["--low_rank_config", str(lr)])
                subprocess.run(cmd, cwd=str(LEGACY), env=env, check=True)
            continue

        if _reuse_legacy_ppl(args.point, meth, args.dataset):
            continue

        model = paths[meth]
        tag = f"OP_{args.point}_{meth}"
        out = str(artifact_dir(f"{tag}_{args.dataset}_ppl", link_under=HERE / "artifacts" / f"{tag}_{args.dataset}_ppl"))
        if (Path(out) / "ppl_aggregate.json").exists():
            continue
        cmd = [
            PYTHON_BIN, str(PPL_RUNNER),
            "--dataset", args.dataset,
            "--model", model, "--tok", MODEL,
            "--out_dir", out, "--dtype", "bf16",
            "--method", f"{meth}_{args.point}",
            "--evaluation_protocol_hash", evaluation_hash(),
        ]
        if args.dataset == "wikitext2":
            cmd.extend(["--wikitext2_dir", WIKITEXT])
        else:
            cmd.extend(["--c4_texts", C4_TEXTS])
        # ppl_instrumented auto-loads low_rank_config.{json,pt} from --model dir.
        subprocess.run(cmd, cwd=str(LEGACY), env=env, check=True)

    print(json.dumps({
        "ok": True, "point": args.point, "dataset": args.dataset,
        "method": args.method, "bundle": args.bundle,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
