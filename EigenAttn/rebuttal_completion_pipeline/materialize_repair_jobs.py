#!/usr/bin/env python3
"""Materialize priority repair DAG — GPU packing parallel evals, disk-safe builds."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import PYTHON_BIN  # noqa: E402
from stage_materialize import make_job  # noqa: E402

CONTROLS = [
    "uniform_matched",
    "random_seed0",
    "random_seed1",
    "random_seed2",
    "spectral_tail",
    "middle_layer",
    "reconstruction_slope",
    "depth_front",
    "depth_back",
]


def build_repair_jobs():
    jobs = []
    jobs.append(make_job(
        "FIX_KV_RECOMPUTE", "FIX", "cpu",
        [PYTHON_BIN, str(HERE / "runners" / "repair_kv_recompute.py")],
        meta={"priority": 1},
    ))
    jobs.append(make_job(
        "FIX_OP_PROTOCOL_AUDIT", "FIX", "cpu",
        [PYTHON_BIN, "-c",
         "from pathlib import Path; import json; "
         "p=Path('/home/bai-project/jiaqizhang/EigenAttn/rebuttal_completion_pipeline/artifacts/kv_repair/SINGLE_SCHEDULE_LOCK_AUDIT.json'); "
         "assert p.exists(), p; print(json.dumps({'ok': True}))"],
        depends_on=["FIX_KV_RECOMPUTE"],
    ))

    # Per control: build (disk-serialized) -> 4 parallel GPU evals -> cleanup
    for ctrl in CONTROLS:
        build_id = f"MCFIX_{ctrl}_build"
        jobs.append(make_job(
            build_id, "MCFIX", "gpu",
            [PYTHON_BIN, str(HERE / "runners" / "matched_control_eval.py"),
             "--control", ctrl, "--phase", "build"],
            depends_on=["FIX_KV_RECOMPUTE"],
            meta={"reserve_mib": 22000, "control": ctrl, "role": "build"},
        ))
        eval_ids = []
        for phase, extra in [
            ("wikitext2", []),
            ("c4", []),
            ("zero_shot", ["--bundle", "hs_arc"]),
            ("zero_shot", ["--bundle", "wg_piqa"]),
        ]:
            bund = extra[1] if extra else phase
            jid = f"MCFIX_{ctrl}_{bund}"
            eval_ids.append(jid)
            jobs.append(make_job(
                jid, "MCFIX", "gpu",
                [PYTHON_BIN, str(HERE / "runners" / "matched_control_eval.py"),
                 "--control", ctrl, "--phase", phase, *extra],
                depends_on=[build_id],
                meta={"reserve_mib": 18000, "control": ctrl, "role": "eval"},
            ))
        jobs.append(make_job(
            f"MCFIX_{ctrl}_cleanup", "MCFIX", "cpu",
            [PYTHON_BIN, str(HERE / "runners" / "matched_control_eval.py"),
             "--control", ctrl, "--phase", "cleanup"],
            depends_on=eval_ids,
            meta={"control": ctrl, "role": "cleanup"},
        ))

    for pt in ["mild", "moderate", "aggressive"]:
        jobs.append(make_job(
            f"FIX_STATS_{pt}", "FIX", "cpu",
            [PYTHON_BIN, str(HERE / "runners" / "op_stats_repair.py"), "--point", pt],
        ))

    # KQV after primary MC cleanups free disk (or at least after a few builds)
    jobs.append(make_job(
        "FIX_KQV_V2", "FIX", "gpu",
        [PYTHON_BIN, str(HERE / "runners" / "kqv_ablation_v2.py")],
        depends_on=["FIX_KV_RECOMPUTE", "MCFIX_uniform_matched_cleanup"],
        meta={"reserve_mib": 22000, "role": "build"},
    ))
    return jobs


def main() -> None:
    repair = build_repair_jobs()
    jobs_path = HERE / "jobs.jsonl"
    existing = [json.loads(l) for l in jobs_path.read_text().splitlines() if l.strip()] if jobs_path.exists() else []
    existing = [j for j in existing if not str(j.get("job_id", "")).startswith(("FIX_", "MCFIX_"))]
    stop = HERE / "STOP_REQUESTED"
    if stop.exists():
        stop.unlink()
    merged = existing + repair

    art = Path("/diskE/jiaqizhang/rebuttal_completion_artifacts/matched_controls")
    model_root = Path("/diskF/jiaqizhang/rebuttal_completion_op_artifacts/matched_control_models")

    for j in merged:
        jid = j["job_id"]
        if not jid.startswith(("FIX_", "MCFIX_")):
            continue
        out = Path(j["output_dir"])
        out.mkdir(parents=True, exist_ok=True)

        # Adopt completed work
        if jid == "FIX_KV_RECOMPUTE" and (out / "DONE.json").exists():
            j["status"] = "DONE"
            continue
        if jid.startswith("FIX_STATS_") and (HERE / "artifacts" / "statistics" / f"PPL_BOOTSTRAP_{jid.split('_')[-1]}.json").exists():
            j["status"] = "DONE"
            (out / "DONE.json").write_text(json.dumps({"job_id": jid, "status": "DONE", "adopted": True}) + "\n")
            continue

        # Adopt MC phases from artifacts
        if jid.startswith("MCFIX_") and "_build" in jid:
            ctrl = j["meta"]["control"]
            if (model_root / ctrl / "final_model" / "model.safetensors.index.json").exists() or any(
                (model_root / ctrl / "final_model").glob("model-*.safetensors")
            ):
                j["status"] = "DONE"
                (out / "DONE.json").write_text(json.dumps({"job_id": jid, "status": "DONE", "adopted": True}) + "\n")
                continue
        if jid.startswith("MCFIX_"):
            ctrl = j.get("meta", {}).get("control")
            role = j.get("meta", {}).get("role")
            if ctrl and role == "eval":
                bund = jid.split(f"MCFIX_{ctrl}_")[-1]
                if bund == "wikitext2" and (art / ctrl / "wikitext2_ppl" / "ppl_aggregate.json").exists():
                    j["status"] = "DONE"
                    (out / "DONE.json").write_text(json.dumps({"job_id": jid, "status": "DONE", "adopted": True}) + "\n")
                    continue
                if bund == "c4" and (art / ctrl / "c4_ppl" / "ppl_aggregate.json").exists():
                    j["status"] = "DONE"
                    (out / "DONE.json").write_text(json.dumps({"job_id": jid, "status": "DONE", "adopted": True}) + "\n")
                    continue
                if bund == "hs_arc" and (art / ctrl / "hs_arc" / "zero_shot_aggregate.json").exists():
                    j["status"] = "DONE"
                    (out / "DONE.json").write_text(json.dumps({"job_id": jid, "status": "DONE", "adopted": True}) + "\n")
                    continue
                if bund == "wg_piqa" and (art / ctrl / "wg_piqa" / "zero_shot_aggregate.json").exists():
                    j["status"] = "DONE"
                    (out / "DONE.json").write_text(json.dumps({"job_id": jid, "status": "DONE", "adopted": True}) + "\n")
                    continue

        j["status"] = "PENDING"
        for name in ("DONE.json", "FAILED.json"):
            p = out / name
            if p.exists():
                p.unlink()
        (out / "job_state.json").write_text(json.dumps({"job_id": jid, "status": "PENDING"}) + "\n")

    jobs_path.write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in merged))
    (HERE / "repair_jobs.jsonl").write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in repair))
    print(json.dumps({"ok": True, "n_repair": len(repair), "n_total": len(merged)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
