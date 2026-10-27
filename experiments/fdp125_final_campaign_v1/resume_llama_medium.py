#!/usr/bin/env python3
import sys, shutil, traceback
sys.path.insert(0, "/diskF/jiaqizhang/fdp125_final_campaign_v1")
import llama_fdp125_tier_worker as w

GPU = 3
TIER = "medium"
scratch = w.SCRATCH_ROOT / "llama_medium_resume"
out = w.CAMPAIGN / "results/llama/medium"

def main():
    if (out / "CORE_DONE.json").is_file():
        return
    if not (out / "step1.json").is_file():
        raise FileNotFoundError(out / "step1.json")
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    commands = []
    w.atomic_json(out / "STATUS.json", {"status":"RUNNING_RESUME", "tier":TIER, "gpu":GPU, "at":w.now(), "resume_from":"step1.json"})
    try:
        step1 = out / "step1.json"
        schedules = {"parent": w.derive_eb(step1, w.BASE_E[TIER], (1,1,1,1,1)), "fdp125": w.derive_eb(step1, w.BASE_E[TIER], (1,1,0,0,1))}
        methods = {}
        for method in ("parent", "fdp125"):
            result = out / method
            result.mkdir(parents=True, exist_ok=True)
            model_dir, meta = w.basis_and_model(method, TIER, schedules[method], GPU, scratch / method, scratch / "logs", commands)
            quality = w.evaluate(method, TIER, model_dir, GPU, result, scratch / "logs", commands)
            w.atomic_json(result / "per_layer_eb.json", {"error_budget_by_layer": meta["eb_vector"]})
            w.atomic_json(result / "low_rank_config.json", meta["ranks"])
            prov = {"status":"PASS", "model":"llama3_8b", "regime":TIER, "method":method, "base_e":w.BASE_E[TIER], "lambda":1.2, "rho":0.75, "components":["d1","d2","d3","d4","d5"] if method=="parent" else ["d1","d2","d5"], **meta, "full_resource_bytes":w.FULL_BYTES, "retained_ratio":meta["B_live"]/w.FULL_BYTES, "target_B_live":w.TARGET_B[TIER], "resource_gap_pct":(meta["B_live"]-w.TARGET_B[TIER])/w.TARGET_B[TIER]*100, "quality":quality, "commands":commands, "resumed_from":str(step1), "completed_at":w.now()}
            w.atomic_json(result / "provenance.json", prov)
            methods[method] = {"B_live":meta["B_live"], "wiki_ppl":quality["wiki"]["ppl"], "c4_ppl":quality["c4"]["ppl"], "rank_sha":meta["rank_sha"], "basis_sha":meta["basis_sha"]}
            shutil.rmtree(model_dir)
        w.atomic_json(out / "CORE_DONE.json", {"status":"PASS", "model":"llama3_8b", "tier":TIER, "methods":methods, "resumed_from":"step1.json", "completed_at":w.now()})
        w.atomic_json(out / "STATUS.json", {"status":"DONE", "at":w.now(), "resumed":True})
        shutil.rmtree(scratch, ignore_errors=True)
    except Exception as exc:
        w.atomic_json(out / "RESUME_FAILED_ENGINEERING.json", {"status":"FAILED_ENGINEERING", "error":repr(exc), "traceback":traceback.format_exc(), "commands":commands, "at":w.now()})
        if (scratch / "logs").exists():
            dest = out / "resume_failure_logs"
            if dest.exists(): shutil.rmtree(dest)
            shutil.copytree(scratch / "logs", dest)
        shutil.rmtree(scratch, ignore_errors=True)
        raise

if __name__ == "__main__":
    main()
