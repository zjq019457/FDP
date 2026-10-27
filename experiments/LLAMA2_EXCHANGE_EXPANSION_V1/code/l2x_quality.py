#!/usr/bin/env python3
"""Evaluate one rank schedule on Llama-2-7B EigenAttention: frozen 166 Wiki / 256 C4 windows (legacy scorer) and the frozen
5-task zero-shot sample manifest. Model built in memory by the frozen canonical materializer; nothing written to disk but results.
usage: l2x_quality.py LABEL SCHEDULE_JSON PHYSICAL_GPU [--ppl-only]"""
import json, math, os, pathlib, sys, traceback
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import l2x_lib as X


def main(label, sched_path, gpu, ppl_only):
    import torch
    out = X.ROOT / "quality" / label
    if (out / "RESULT.json").exists(): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = X.rjson(sched_path)["rank_pairs"]
    st = {"label": label, "schedule": str(sched_path), "schedule_sha256": X.sha_file(sched_path), "rank_sha": X.sha_json(pairs),
          "B_live": X.B_of(pairs), "ratio": X.B_of(pairs) / X.FULL, "physical_gpu": gpu, "started": X.now(), "script_sha256": X.sha_file(__file__)}
    X.wjson(out / "RUN_STATUS.json", st)
    try:
        torch.cuda.set_per_process_memory_fraction(0.45, device=0)
        v5, mod, arms, basis_meta = X.materializer(gpu)
        v7 = X.load_mod(X.V7_PATH, "l2x_v7")
        runner = X.load_mod(v5.RUNNER, "l2x_ppl_runner")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(v5.MODEL), use_fast=False, local_files_only=True)
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
        _, _, _, wiki, c4, _ = v7.validate_inputs(v5, runner, tok)
        model = X.build(mod, arms, basis_meta, label, pairs)
        parity = [v5.factorized_operator_parity(model, li) for li in (0, 16, 31)]
        if any(p["relative_l2"] > 1e-3 or p["cosine"] < 0.9999 for p in parity): raise RuntimeError("factorized operator parity failed")
        st["operator_parity"] = parity
        mh = X.sha_json([label, st["rank_sha"]])
        for ds, wins in (("wikitext2", wiki), ("c4", c4)):
            ppl, agg = runner.eval_windows_legacy_llama(model, wins, device=torch.device("cuda:0"), seqlen=2048, dataset=ds,
                                                        method=f"l2x_{label}_{ds}", model_hash=mh, basis_hash=basis_meta["sha256"], rank_plan_hash=st["rank_sha"],
                                                        evaluation_protocol_hash=X.sha_file(v5.EVAL_PROTOCOL), out_dir=out / "ppl" / ds, limit=-1,
                                                        skip_nonfinite=(ds == "wikitext2"))
            st.setdefault("ppl", {})[ds] = {"ppl": ppl, "used": agg.get("used_windows"), "skipped": agg.get("skipped_windows")}
            X.wjson(out / "RUN_STATUS.json", st)
        st["J"] = 0.5 * (math.log(st["ppl"]["wikitext2"]["ppl"]) + math.log(st["ppl"]["c4"]["ppl"]))
        del model; torch.cuda.empty_cache()
        if not ppl_only:
            base = X.load_mod(X.ZS_BASE, "l2x_zs_base")
            base.OUT = out / "zs"
            census = arms
            def load_common():
                c = mod
                c.TOKENIZER = v5.MODEL
                c.check_basis = getattr(c, "check_basis", lambda: basis_meta)
                auth = {"cell_launch_authorized": True}
                return c, auth, census, {"rank_pairs": pairs}, {}
            base.load_common = load_common
            base.gate_snapshot = lambda: {"gpu_free_mib": {gpu: 10 ** 9}, "e_free_bytes": 10 ** 15}
            os.environ["FDP125_PHYSICAL_GPU"] = str(gpu)
            base.run_one_arm(label)
            agg = X.rjson(out / "zs" / label / "zero_shot_aggregate.json")
            st["zs"] = {"macro": agg["zero_shot_macro_accuracy"], "tasks": {t: v["accuracy_from_correct"] for t, v in agg["tasks"].items()},
                        "samples_dir": str(out / "zs" / label / "zero_shot_samples")}
        st["status"] = "COMPLETE"; st["finished"] = X.now()
        X.wjson(out / "RESULT.json", st); X.wjson(out / "RUN_STATUS.json", st)
        print(json.dumps({k: st.get(k) for k in ("label", "B_live", "ppl", "J", "zs")}, default=str))
    except Exception as e:
        st.update(status="FAILED", error=repr(e), traceback=traceback.format_exc(), failed=X.now())
        X.wjson(out / "RUN_STATUS.json", st); raise


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), "--ppl-only" in sys.argv)
