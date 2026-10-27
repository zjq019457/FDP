import json, glob, pathlib
R = pathlib.Path("/diskF/jiaqizhang/LLAMA2_EXCHANGE_EXPANSION_V1")
S = lambda n: json.load(open(R / "schedules" / n / "SCHEDULE.json"))
for a, b in (("e060", "f060"), ("e068", "f068"), ("e090", "f090")):
    pa, pb = S(f"uniform_{a}")["rank_pairs"], S(f"uniform_{b}")["rank_pairs"]
    ch = [(i, pa[i], pb[i]) for i in range(32) if pa[i] != pb[i]]
    qk_lo = [i for i in range(32) if pa[i][0] < 1843]
    print(a, "B", sum((x+y)*4096 for x, y in pa), "->", b, sum((x+y)*4096 for x, y in pb), "| no-floor layers with QK<45%:", qk_lo, "| min QK", min(p[0] for p in pa), "| floor-raised:", [(i, f"{x[0]}/{x[1]}->{y[0]}/{y[1]}") for i, x, y in ch][:10])
for n in ("func_e060_std", "func_e068_std", "func_e090_std", "func_f060_std", "func_f068_std", "func_f090_std", "fdp_e068", "fdp_f068"):
    d = S(n)
    if "accepted" in d: mv = [(m["donor"], m["receiver"]) for m in d["accepted"]]
    elif "moves" in d: mv = [(m["donor"], m["receiver"]) for m in d["moves"]]
    else:
        u = S("uniform_" + d["op"])["rank_pairs"]; p = d["rank_pairs"]
        mv = [(i, u[i], p[i]) for i in range(32) if u[i] != p[i]]
    print(n, "K", d.get("K"), mv)
for p in sorted(glob.glob(str(R / "quality" / "*" / "RESULT.json"))):
    d = json.load(open(p))
    if d["label"].startswith("rand_"): continue
    t = d.get("zs", {}).get("tasks", {})
    print("Q", d["label"], round(d["ratio"], 4), round(d["ppl"]["wikitext2"]["ppl"], 3), round(d["ppl"]["c4"]["ppl"], 3), round(d["J"], 4), round(100 * d["zs"]["macro"], 2))
for op in ("e060", "e068", "e090", "f060", "f068", "f090"):
    rs = [json.load(open(p)) for p in glob.glob(str(R / "quality" / f"rand_{op}_*" / "RESULT.json"))]
    if rs:
        import statistics as st
        print("RAND", op, len(rs), "J mean", round(st.mean(r["J"] for r in rs), 4), "acc mean", round(100 * st.mean(r["zs"]["macro"] for r in rs), 2))
s = json.load(open(R / "STATUS_FLOOR.json")); print("FLOOR", s["done"], s["total"], s["running"], s["failed"])
