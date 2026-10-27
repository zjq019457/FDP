#!/usr/bin/env python3
"""Functional (measured-NLL) exact-budget exchange at a Llama-2 Uniform operating point.
Algorithm identical to the frozen e068 run (LLAMA2_E068_NATIVE_EXCHANGE_ITERATIVE.py + add-benefit runner + stability addendum):
probe add/remove one 32-rank quantum on all 64 paths on the tuning windows; donors: remove-delta <= 0 on full and both halves;
receivers: add-delta < 0 on full and both halves; pairs from different layers with positive predicted gain on full and both halves;
take the best pair, accept only if the joint full NLL decreases and both halves are non-worsening, else stop; re-probe the two moved
paths after each acceptance; at most 12 accepted exchanges. Evaluation data never used.
usage: l2x_exchange.py OP WINDOWS(std|dd) PHYSICAL_GPU OUT_NAME"""
import copy, json, pathlib, sys, traceback
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import l2x_lib as X

MAX_ACCEPTED = 12


def main(op, wkind, gpu, name):
    import torch
    out = X.ROOT / "schedules" / name
    if (out / "SCHEDULE.json").exists(): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "ATTEMPTS.jsonl", "a")
    torch.cuda.set_per_process_memory_fraction(0.45, device=0)
    dev = torch.device("cuda:0")
    v5, mod, arms, basis_meta = X.materializer(gpu)
    add, windows = X.tuning_windows(wkind)
    pairs = X.uniform_pairs(op); B0 = X.B_of(pairs)
    model = X.build(mod, arms, basis_meta, f"uniform_{op}", pairs)
    from transformers import AutoModelForCausalLM
    from models.decompose_modules import LlamaEigenAttnDecoderLayer
    from types import SimpleNamespace
    base_model = AutoModelForCausalLM.from_pretrained(str(v5.MODEL), local_files_only=True, torch_dtype=torch.bfloat16,
                                                      attn_implementation="eager", low_cpu_mem_usage=True).eval()
    raw = torch.load(mod.BASIS_FILE, map_location="cpu", weights_only=False, mmap=True)
    hd = 128

    def build_layer(li, kr, vr):
        bl = copy.deepcopy(base_model.model.layers[li]).to(dev)
        pre = f"base_model.model.model.layers.{li}.self_attn."
        bk = raw[pre + "k_proj_basis"][:, :kr // 32].to(device=dev, dtype=torch.bfloat16)
        vp = raw[pre + "v_proj_basis"]
        vs = torch.stack([vp[h * hd:(h + 1) * hd, h * hd:(h + 1) * hd] for h in range(32)], 0)[:, :, :vr // 32].to(device=dev, dtype=torch.bfloat16)
        return LlamaEigenAttnDecoderLayer(bl, SimpleNamespace(), bk, kr, vs, vr, model.config, li).to(device=dev, dtype=torch.bfloat16)

    # same-rank layer rebuild canary (as the frozen run)
    ids = torch.tensor(windows[0]["input_ids"], dtype=torch.long, device=dev).unsqueeze(0)
    with torch.no_grad(): ref_logits = model(input_ids=ids, use_cache=False).logits.float()
    old = model.model.layers[4]; model.model.layers[4] = build_layer(4, *pairs[4])
    with torch.no_grad(): d = float((model(input_ids=ids, use_cache=False).logits.float() - ref_logits).abs().max())
    model.model.layers[4] = old
    if d > 1e-3: raise RuntimeError(f"layer rebuild canary failed {d}")
    del ref_logits, ids; torch.cuda.empty_cache()

    reference = add.score_windows(model, windows, dev, torch)

    def uid_lp(u): return int(u[1:3]), u.split("_")[1]
    def rank_of(p, u): li, pa = uid_lp(u); return p[li][0 if pa == "QK" else 1]
    def upd(p, u, dlt): li, pa = uid_lp(u); p[li][0 if pa == "QK" else 1] += dlt

    def probe(p, ref, u, direction):
        nr = rank_of(p, u) + direction * X.QUANTUM
        if not (X.rmin(op) <= nr <= X.RMAX): return {"delta": None, "A": None, "B": None}
        c = [x[:] for x in p]; upd(c, u, direction * X.QUANTUM); li, _ = uid_lp(u)
        o = model.model.layers[li]; model.model.layers[li] = build_layer(li, *c[li])
        try: m = add.score_windows(model, windows, dev, torch)
        finally: model.model.layers[li] = o; torch.cuda.empty_cache()
        a, b = X.loss(ref), X.loss(m)
        return {"delta": b["full"] - a["full"], "A": b["A"] - a["A"], "B": b["B"] - a["B"]}

    units = [f"L{li:02d}_{pa}" for li in range(32) for pa in ("QK", "V")]
    cache = {}
    for u in units:
        cache[u] = {"remove": probe(pairs, reference, u, -1), "add": probe(pairs, reference, u, +1)}
        log.write(json.dumps({"probe": u, **cache[u]}) + "\n"); log.flush()

    def choose(p):
        don = [u for u in units if rank_of(p, u) > X.rmin(op) and cache[u]["remove"]["delta"] is not None and cache[u]["remove"]["delta"] <= 0
               and cache[u]["remove"]["A"] <= 0 and cache[u]["remove"]["B"] <= 0]
        rec = [u for u in units if rank_of(p, u) < X.RMAX and cache[u]["add"]["delta"] is not None and cache[u]["add"]["delta"] < 0
               and cache[u]["add"]["A"] < 0 and cache[u]["add"]["B"] < 0]
        ch = []
        for d_ in don:
            for r_ in rec:
                if uid_lp(d_)[0] == uid_lp(r_)[0]: continue
                dm, ra = cache[d_]["remove"], cache[r_]["add"]
                g = (-(dm["delta"] + ra["delta"]), -(dm["A"] + ra["A"]), -(dm["B"] + ra["B"]))
                if min(g) > 0: ch.append((g[0], d_, r_))
        ch.sort(key=lambda x: (-x[0], x[1], x[2])); return ch

    accepted, stop = [], None
    for it in range(1, MAX_ACCEPTED + 1):
        ch = choose(pairs)
        if not ch: stop = "NO_POSITIVE_SPLIT_STABLE_PREDICTED_EXCHANGE"; break
        g, d_, r_ = ch[0]
        c = [x[:] for x in pairs]; upd(c, d_, -X.QUANTUM); upd(c, r_, X.QUANTUM); assert X.B_of(c) == B0
        dl, rl = uid_lp(d_)[0], uid_lp(r_)[0]
        olds = {dl: model.model.layers[dl], rl: model.model.layers[rl]}
        model.model.layers[dl] = build_layer(dl, *c[dl]); model.model.layers[rl] = build_layer(rl, *c[rl])
        m = add.score_windows(model, windows, dev, torch)
        a, b = X.loss(reference), X.loss(m); dlt = {h: b[h] - a[h] for h in ("full", "A", "B")}
        ok = dlt["full"] < 0 and dlt["A"] <= 0 and dlt["B"] <= 0
        rec = {"iteration": it, "donor": d_, "receiver": r_, "predicted_gain": g, "actual_delta": dlt, "accepted": ok, "at": X.now()}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if not ok:
            model.model.layers[dl], model.model.layers[rl] = olds[dl], olds[rl]; stop = "JOINT_EXCHANGE_FAILED_FULL_OR_SPLIT_ACCEPTANCE"; break
        accepted.append(rec); pairs = c; reference = m
        for u in (d_, r_): cache[u] = {"remove": probe(pairs, reference, u, -1), "add": probe(pairs, reference, u, +1)}
    else:
        stop = "MAX_ACCEPTED"
    X.wjson(out / "SCHEDULE.json", {"method": "functional measured-NLL exchange (frozen e068 algorithm)", "op": op, "tuning_windows": wkind,
                                    "rank_pairs": pairs, "B_live": X.B_of(pairs), "uniform_B_live": B0, "accepted": accepted, "K": len(accepted),
                                    "stop_reason": stop, "rank_min": X.rmin(op), "uniform_rank_sha": X.sha_json(X.uniform_pairs(op)), "rank_sha": X.sha_json(pairs),
                                    "final_tuning_loss": X.loss(reference), "script_sha256": X.sha_file(__file__), "evaluation_data_used": False, "at": X.now()})
    print(json.dumps({"op": op, "K": len(accepted), "stop": stop}))


if __name__ == "__main__":
    try: main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
    except Exception:
        out = X.ROOT / "schedules" / sys.argv[4]; out.mkdir(parents=True, exist_ok=True)
        (out / "FAILED.txt").write_text(traceback.format_exc()); raise
