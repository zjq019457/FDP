#!/usr/bin/env python3
"""Re-run the official PaLu zero-shot runner (PALU_OFFICIAL_PROTOCOL_REPRO_V1/code/run_official_zero_shot.py) unchanged except:
log_samples=True and output to a new directory with per-item correctness, so Fisher and PaLu-FDP initial can be paired.
Gate: the six task accuracies must equal the existing official_zero_shot/ratio_0p5/<method>/zeroshot.json."""
import json, pathlib, sys
SRC = pathlib.Path("/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1/code/run_official_zero_shot.py")
OUT = pathlib.Path("/diskF/jiaqizhang/NAACL_PAPER_SUPPLEMENT_V1/palu_zs_logged")
method = sys.argv[1]
src = SRC.read_text()
src = src.replace("outdir = ROOT / 'official_zero_shot' / ('ratio_' + tag) / args.method", "outdir = OUT_DIR / ('ratio_' + tag) / args.method")
src = src.replace("log_samples=False,", "log_samples=True,")
assert "OUT_DIR / ('ratio_' + tag)" in src and "log_samples=True," in src
g = {"__name__": "palu_zs_logged", "__file__": str(SRC), "OUT_DIR": OUT}
exec(compile(src, str(SRC), "exec"), g)
import lm_eval
orig = lm_eval.simple_evaluate
def wrapped(*a, **k):
    r = orig(*a, **k)
    d = OUT / "ratio_0p5" / method / "samples"; d.mkdir(parents=True, exist_ok=True)
    for task, rows in (r.get("samples") or {}).items():
        with open(d / f"{task}.jsonl", "w") as f:
            for s in rows:
                f.write(json.dumps({"doc_id": s.get("doc_id"), "acc": s.get("acc"), "acc_norm": s.get("acc_norm"), "doc_hash": s.get("doc_hash"), "target": str(s.get("target"))}) + "\n")
    return r
lm_eval.simple_evaluate = wrapped
sys.modules["lm_eval"].simple_evaluate = wrapped
sys.argv = [str(SRC), "--ratio", "0.5", "--method", method]
g["main"]()
