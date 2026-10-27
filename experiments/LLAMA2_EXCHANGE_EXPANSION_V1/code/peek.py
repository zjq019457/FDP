import json, glob, os, math
R = "/diskF/jiaqizhang/LLAMA2_EXCHANGE_EXPANSION_V1"
s = json.load(open(R + "/STATUS.json")); print("STATUS", s["at"], s["done"], "/", s["total"], "running", s["running"], "failed", s["failed"])
for n in sorted(glob.glob(R + "/schedules/func_*/SCHEDULE.json")):
    d = json.load(open(n)); print("FUNC", os.path.basename(os.path.dirname(n)), "K", d.get("K"), d.get("stop_reason"), d.get("rank_sha", "")[:10])
for n in sorted(glob.glob(R + "/schedules/func_*/ATTEMPTS.jsonl")):
    L = [json.loads(l) for l in open(n)]; it = [x for x in L if "iteration" in x]
    print("ATT", os.path.basename(os.path.dirname(n)), "probes", sum("probe" in x for x in L), "iters", [(x["donor"], x["receiver"], x["accepted"]) for x in it])
for p in sorted(glob.glob(R + "/profiles/*/run/*PROFILE_V1.json")): print("PROFILE done", p)
for p in sorted(glob.glob(R + "/logs/profile_*.log")):
    t = open(p).read().strip().splitlines(); print("PROFILE log", os.path.basename(p), t[-1][:120] if t else "")
rows = []
for p in sorted(glob.glob(R + "/quality/*/RESULT.json")):
    d = json.load(open(p)); z = d.get("zs", {}).get("macro")
    rows.append((d["label"], d["B_live"], round(d["ppl"]["wikitext2"]["ppl"], 3), round(d["ppl"]["c4"]["ppl"], 3), round(d["J"], 4), round(100 * z, 2) if z else None))
for r in rows: print("Q", *r)
