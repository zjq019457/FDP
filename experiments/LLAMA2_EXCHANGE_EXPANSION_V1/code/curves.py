import json, pathlib
R = pathlib.Path("/diskF/jiaqizhang/LLAMA2_EXCHANGE_EXPANSION_V1")
H = pathlib.Path("/diskE/jiaqizhang/ICLR_FDP125_PAPER_EVIDENCE_V1/13_MODEL_MATRIX_4X3/LLAMA2_E068_NATIVE_ADD_BENEFIT_V1_ATTEMPT1_20261001/LLAMA2_E068_NATIVE_EXCHANGE_ITERATION_ATTEMPT4_ATTEMPTS.jsonl")
out = {}
def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]
hist = [x for x in load(H) if "iteration" in x]
out["e068_hist"] = [{"it": x["iteration"], "d": x["actual_delta_loss"]["full"], "acc": x["accepted"]} for x in hist]
for n in ("func_e060_std", "func_e068_dd", "func_e090_std", "func_f060_std", "func_f068_std", "func_f090_std", "func_e068_std_repro"):
    it = [x for x in load(R / "schedules" / n / "ATTEMPTS.jsonl") if "iteration" in x]
    out[n] = [{"it": x["iteration"], "d": x["actual_delta"]["full"], "acc": x["accepted"]} for x in it]
for k, v in out.items(): print(k, [(a["it"], round(a["d"], 5), a["acc"]) for a in v])
json.dump(out, open(R / "results" / "EXCHANGE_TUNING_CURVES.json", "w"), indent=1)
