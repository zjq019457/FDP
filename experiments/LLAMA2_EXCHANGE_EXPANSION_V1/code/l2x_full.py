#!/usr/bin/env python3
"""Per-arm absolute metrics for the Llama-2 exchange table (read-only)."""
import glob, json, math, pathlib
Q = pathlib.Path("/diskF/jiaqizhang/LLAMA2_EXCHANGE_EXPANSION_V1/quality")
TASKS = ["arc_challenge_localjsonl", "arc_easy_localjsonl", "hellaswag_localjsonl", "piqa_localjsonl", "winogrande_localjsonl"]
ARMS = ["uniform_e060", "func_e060_std", "gate_uniform068", "func_e068_std", "func_e068_dd", "uniform_e090", "func_e090_std",
        "uniform_f060", "func_f060_std", "uniform_f068", "func_f068_std", "uniform_f090", "func_f090_std"]
out = {}
for a in ARMS:
    r = {}
    try:
        R = json.load(open(Q / a / "RESULT.json")); r["B_live"] = R.get("B_live"); r["ratio"] = R.get("ratio")
    except Exception as e:
        r["err"] = str(e)
    for ds in ("wikitext2", "c4"):
        n = t = c = 0
        for f in glob.glob(str(Q / a / "ppl" / ds / "windows" / "*.json")):
            d = json.load(open(f))
            if d.get("skipped"): continue
            n += d["total_nll"]; t += d["token_count"]; c += 1
        r[ds] = {"ppl": math.exp(n / t) if t else None, "lnppl": n / t if t else None, "windows": c}
    if r["wikitext2"]["lnppl"] is not None and r["c4"]["lnppl"] is not None:
        r["J"] = (r["wikitext2"]["lnppl"] + r["c4"]["lnppl"]) / 2
    acc = {}
    for tk in TASKS:
        p = Q / a / "zs" / a / "zero_shot_samples" / f"{tk}.jsonl"
        if p.exists():
            rows = [json.loads(x) for x in open(p) if x.strip()]
            acc[tk.replace("_localjsonl", "")] = 100 * sum(bool(x["correct"]) for x in rows) / len(rows)
    r["acc"] = acc
    if len(acc) == 5: r["avg"] = sum(acc.values()) / 5
    out[a] = r
json.dump(out, open("/diskF/jiaqizhang/LLAMA2_EXCHANGE_EXPANSION_V1/results/ABSOLUTE_METRICS_FOR_TABLE.json", "w"), indent=1)
for a, r in out.items():
    acc = r["acc"]
    print(a, r.get("ratio"), r.get("B_live"), "W %.3f C %.3f J %.4f" % (r["wikitext2"]["ppl"] or 0, r["c4"]["ppl"] or 0, r.get("J", 0)),
          r["wikitext2"]["windows"], r["c4"]["windows"], " ".join("%s=%.2f" % (k[:6], v) for k, v in acc.items()), "avg %.2f" % r.get("avg", 0))
