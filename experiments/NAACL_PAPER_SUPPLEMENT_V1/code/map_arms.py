import json, glob, math, os
roots = ["/diskF/jiaqizhang/fdp125_final_campaign_v1/results/llama", "/diskF/jiaqizhang/TACL_LLAMA3_MEDIUM_NONFUNCTIONAL_CONTROLS_COMMON_BASIS_V1",
         "/diskF/jiaqizhang/TACL_LLAMA3_SPECTRAL_CROSS_TIER_V1/materialized", "/diskE/jiaqizhang/FDP125_STRICT_FORMAL_CLOSURE_V1/materialized_formal_checkpoints",
         "/diskE/jiaqizhang/FDP125_STRICT_FORMAL_CLOSURE_V1/quality_scratch", "/diskE/jiaqizhang/FDP125_STRICT_FORMAL_CLOSURE_V1/official_control"]
rows = {}
for r in roots:
    for p in glob.glob(r + "/**/ppl_aggregate.json", recursive=True):
        try: a = json.load(open(p))
        except Exception: continue
        if a.get("used_windows") not in (141, 256): continue
        d = os.path.dirname(p); key = os.path.dirname(d)
        ds = a.get("dataset")
        rows.setdefault(key, {})[ds] = (round(a["ppl"], 6), a.get("method"), os.path.exists(d + "/ppl_windows.jsonl"))
for k, v in sorted(rows.items()):
    w = v.get("wikitext2"); c = v.get("c4")
    print(k.replace("/diskF/jiaqizhang/", "F:").replace("/diskE/jiaqizhang/", "E:"), "W", w and w[0], "C", c and c[0], "win", (w and w[2], c and c[2]), (w or c)[1])
