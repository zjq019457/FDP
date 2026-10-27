import json, pathlib, sys
O = pathlib.Path("/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1/PALU_OFFICIAL_LONGBENCH_V1/runs/ratio_0p5")
P = pathlib.Path("/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1/longbench_supplemental_4096/PALU_V3_PAIRED_POSTHOC_V1/ratio_0.5")
TASKS = ["triviaqa", "qasper", "trec", "samsum", "lcc", "repobench-p", "qmsum", "multi_news"]
def jl(p): return [json.loads(l) for l in open(p) if l.strip()]
for t in TASKS:
    f = jl(O / "fisher" / t / "predictions.jsonl"); n = jl(O / "fdp_native" / t / "predictions.jsonl")
    v1 = jl(P / "PALU_V1_NATIVE_MATCHED" / f"{t}.raw.jsonl"); v3 = jl(P / "PALU_V3_SELECTED" / f"{t}.raw.jsonl")
    key = lambda r: r.get("sample_id") or r.get("_id") or r.get("row_index")
    pred = lambda r: r.get("prediction", r.get("pred"))
    mn = {key(r): pred(r) for r in n}; m1 = {key(r): pred(r) for r in v1}
    common = set(mn) & set(m1)
    same = sum(mn[k] == m1[k] for k in common)
    print(t, "n", len(f), len(n), len(v1), len(v3), "| keys v1:", list(v1[0].keys())[:8], "| common", len(common), "identical native-vs-V1", same)
