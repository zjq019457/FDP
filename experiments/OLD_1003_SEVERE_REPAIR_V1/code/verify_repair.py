#!/usr/bin/env python3
"""OLD_1003_SEVERE_REPAIR_V1 post-evaluation verification, same checks as verify_o3s.py (usage: verify_repair.py <ARM>; CPU only).

1 ranks: low_rank_config.{json,pt} and the stored factor shapes of all 32 layers equal the frozen PLAN; B_live from tensors.
2 manifest: size + SHA256 of every checkpoint file.
3 basis: exact-rank basis == 10/3 master slice (fill_arm2 column rule), master SHA unchanged.
4 operator parity, existing Severe V2 spec OPERATOR_PARITY_V3_DTYPE_AWARE_FROZEN (unchanged thresholds), all 32 layers:
  3A q_proj / k_proj_up bit-exact; 3B k_low / v_low / o_up max bf16 ULP distance <= 1; 3C K and V->O probes relL2 <= 2^-8.
5 PPL: 141 / 256 windows, 0 skipped; window ids and input_ids SHA equal to the 10/3 Full evaluation.
6 zero-shot: per task, sample ids and document hashes equal to the 10/3 Full evaluation.
7 dual-process CPU canary: two fresh processes load the checkpoint and score the same fixed WikiText-2 window;
  weights digest, logits SHA and NLL must match bit-exactly.
"""
import hashlib, json, os, subprocess, sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from safetensors import safe_open

torch.set_num_threads(16)
REP = Path("/diskF/jiaqizhang/OLD_1003_SEVERE_REPAIR_V1")
REC = Path("/diskF/jiaqizhang/FDP_OFFICIAL_PROTOCOL_V1/records")
FULLQ = Path("/diskF/jiaqizhang/OLD_1003_RANKFLOOR_EXTENSION_V1/quality/FULL")
BASE = Path("/diskD/jiaqizhang/Llama-3")
WIKI = "/home/bai-project/jiaqizhang/wikitext2/test-00000-of-00001.parquet"
U = 2.0 ** -8
CANARY_TOKENS = 512


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()


def jsha(o): return hashlib.sha256(json.dumps(o, separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------- canary child
if len(sys.argv) > 1 and sys.argv[1] == "--canary":
    fm, out = sys.argv[2], sys.argv[3]
    sys.path.insert(0, "/home/bai-project/jiaqizhang/EigenAttn/EigenAttn")
    from transformers import AutoConfig, AutoTokenizer
    from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn
    import pyarrow.parquet as pq
    ranks = json.loads((Path(fm) / "low_rank_config.json").read_text())
    cfg = AutoConfig.from_pretrained(str(BASE), attn_implementation="eager", local_files_only=True)
    m = LlamaForCausalLM_EigenAttn.from_pretrained(fm, config=cfg, low_rank_config=ranks, torch_dtype=torch.bfloat16, local_files_only=True).eval()
    wd = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        wd.update(k.encode()); wd.update(v.contiguous().view(torch.uint8).numpy().tobytes())
    tok = AutoTokenizer.from_pretrained(str(BASE), use_fast=False, local_files_only=True)
    ids = tok("\n\n".join(pq.read_table(WIKI).column("text").to_pylist()), return_tensors="pt").input_ids[:, :CANARY_TOKENS]
    with torch.no_grad():
        o = m(input_ids=ids, labels=ids)
    Path(out).write_text(json.dumps({"pid": os.getpid(), "weights_digest": wd.hexdigest(), "input_ids_sha": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                                     "logits_sha": hashlib.sha256(o.logits.float().contiguous().numpy().tobytes()).hexdigest(), "mean_nll": float(o.loss)}))
    sys.exit(0)

res = {"schema": "OLD_1003_SPECTRAL_TAIL_VERIFY_V1", "device": "cpu"}
ARM = sys.argv[1]; plan = json.loads((REP / "REPAIR_PLAN.json").read_text())["arms"][ARM]; O3S = REP / ARM; O3S.mkdir(exist_ok=True)
fill = json.loads((REC / f"rep_fill_{ARM}" / "result.json").read_text())
fm = Path(fill["final_model"]); work = fm.parent
ranks = plan["ranks"]

# 1 ranks + 2 manifest
lrc = json.loads((fm / "low_rank_config.json").read_text())
lrc_pt = torch.load(fm / "low_rank_config.pt", map_location="cpu", weights_only=False)
lrc_pt = [[int(a), int(b)] for a, b in lrc_pt]
idx = json.loads((fm / "model.safetensors.index.json").read_text())["weight_map"]
H = {}
def T(root, index, key, cache):
    sh = index[key]
    if (root, sh) not in cache: cache[(root, sh)] = safe_open(str(root / sh), framework="pt", device="cpu")
    return cache[(root, sh)].get_tensor(key)
shape_ranks = []
for li in range(32):
    p = f"model.layers.{li}.self_attn."
    shape_ranks.append([T(fm, idx, p + "k_proj_low.weight", H).shape[0], T(fm, idx, p + "v_proj_low.weight", H).shape[0]])
res["ranks"] = {"low_rank_config_json_eq_plan": lrc == ranks, "low_rank_config_pt_eq_plan": lrc_pt == ranks, "tensor_shapes_eq_plan": shape_ranks == ranks,
                "rank_sha": jsha(lrc), "plan_rank_sha": plan["rank_sha256"], "B_live_from_tensors": sum((a + b) * 4096 for a, b in shape_ranks),
                "plan_B": plan["B"], "per_layer": shape_ranks}
res["ranks"]["PASS"] = res["ranks"]["low_rank_config_json_eq_plan"] and res["ranks"]["low_rank_config_pt_eq_plan"] and res["ranks"]["tensor_shapes_eq_plan"] \
    and res["ranks"]["rank_sha"] == plan["rank_sha256"] and res["ranks"]["B_live_from_tensors"] == plan["B"]
res["manifest"] = {"checkpoint": str(fm), "files": [{"name": f.name, "bytes": f.stat().st_size, "sha256": fsha(f)} for f in sorted(fm.iterdir())]}
print("ranks", res["ranks"]["PASS"], flush=True)

# 3 basis
master = json.loads((REC / "old_master" / "result.json").read_text())
mb = torch.load(master["basis_pt"], map_location="cpu", weights_only=False)
bp = work / "basis" / f"eigen_basis_OLD1003_REPAIR_{ARM}_exact_ranks.pt"
eb = torch.load(bp, map_location="cpu", weights_only=False)
bad = []
for li, (kq, v) in enumerate(ranks):
    p = f"base_model.model.model.layers.{li}.self_attn"
    kc = kq // 8; V = mb[f"{p}.v_proj_basis"]; kc_m = V.shape[1] // 8
    want = {"q": mb[f"{p}.q_proj_basis"][:, :kc], "k": mb[f"{p}.k_proj_basis"][:, :kc], "v": V[:, [h * kc_m + c for h in range(8) for c in range(v // 8)]]}
    got = {"q": eb[f"{p}.q_proj_basis"], "k": eb[f"{p}.k_proj_basis"], "v": eb[f"{p}.v_proj_basis"]}
    if not all(torch.equal(want[x], got[x]) for x in want): bad.append(li)
res["basis"] = {"exact_basis": str(bp), "exact_basis_sha256": fsha(bp), "master": master["basis_pt"], "master_sha256_now": fsha(master["basis_pt"]),
                "master_sha256_recorded": master["sha256"], "layers_not_equal_master_slice": bad}
res["basis"]["PASS"] = not bad and res["basis"]["master_sha256_now"] == master["sha256"]
del mb
print("basis", res["basis"]["PASS"], flush=True)


# 4 operator parity (V3 spec)
def ordered(t):
    b = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(b < 0, -(b & 0x7FFF), b)


def ulp(a, b): return int((ordered(a) - ordered(b)).abs().max())
def rel(a, b): return float(torch.linalg.norm(a - b) / torch.linalg.norm(b).clamp_min(1e-30))


bidx = json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"]
BH = {}
per = []; ok3a = True; worst_ulp = 0; worst_k = 0.0; worst_vo = 0.0
for li, (kq, v) in enumerate(ranks):
    p = f"model.layers.{li}.self_attn."; bpfx = f"base_model.model.model.layers.{li}.self_attn"
    rk, rv = kq // 8, v // 8
    obs = {n: T(fm, idx, p + n + ".weight", H) for n in ("q_proj", "k_proj_low", "k_proj_up", "v_proj_low", "o_proj_up")}
    Wq, Wk, Wv, Wo = (T(BASE, bidx, p + n + ".weight", BH) for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
    Uq = eb[f"{bpfx}.q_proj_basis"].float(); Uv = eb[f"{bpfx}.v_proj_basis"].float()
    dt = obs["k_proj_up.weight" if False else "k_proj_up"].dtype
    a3 = {"q_proj": obs["q_proj"].shape == Wq.shape and obs["q_proj"].dtype == Wq.dtype and torch.equal(obs["q_proj"], Wq),
          "k_proj_up": tuple(obs["k_proj_up"].shape) == tuple(Uq.shape) and torch.equal(obs["k_proj_up"], Uq.to(dt))}
    ok3a &= all(a3.values())
    ref_kl = torch.cat([(Uq.T @ Wk.float()[h * 128:(h + 1) * 128, :]) for h in range(8)]).to(obs["k_proj_low"].dtype)
    ref_vl = torch.cat([(Uv[h * 128:(h + 1) * 128, h * rv:(h + 1) * rv].T @ Wv.float()[h * 128:(h + 1) * 128, :]) for h in range(8)]).to(obs["v_proj_low"].dtype)
    ref_ou = torch.cat([(Wo.float()[:, qh * 128:(qh + 1) * 128] @ Uv[(qh // 4) * 128:(qh // 4 + 1) * 128, (qh // 4) * rv:(qh // 4 + 1) * rv]) for qh in range(32)], dim=1).to(obs["o_proj_up"].dtype)
    u3 = {"k_proj_low": ulp(obs["k_proj_low"], ref_kl), "v_proj_low": ulp(obs["v_proj_low"], ref_vl), "o_proj_up": ulp(obs["o_proj_up"], ref_ou)}
    g = torch.Generator().manual_seed(20261005 + li); x = torch.randn((4, 4096), generator=g, dtype=torch.float32)
    kr, vr = 0.0, 0.0
    for h in range(8):
        s = slice(h * rk, (h + 1) * rk)
        kr = max(kr, rel((x @ obs["k_proj_low"][s].float().T) @ obs["k_proj_up"].float().T, (x @ ref_kl[s].float().T) @ Uq.to(dt).float().T))
        vs = slice(h * rv, (h + 1) * rv)
        for gq in range(4):
            qh = h * 4 + gq; os_ = slice(qh * rv, (qh + 1) * rv)
            vr = max(vr, rel((x @ obs["v_proj_low"][vs].float().T) @ obs["o_proj_up"][:, os_].float().T, (x @ ref_vl[vs].float().T) @ ref_ou[:, os_].float().T))
    worst_ulp = max(worst_ulp, *u3.values()); worst_k = max(worst_k, kr); worst_vo = max(worst_vo, vr)
    per.append({"layer": li, "gate3A": a3, "gate3B_max_ulp": u3, "gate3C_relL2_K": kr, "gate3C_relL2_VO": vr})
res["operator_parity_v3"] = {"spec": "SEVERE_V2 STRUCTURAL_CLOSURE_V1/OPERATOR_PARITY_V3_DTYPE_AWARE_FROZEN.json (thresholds unchanged)",
                             "gate3A_all_exact": ok3a, "gate3B_worst_max_ulp": worst_ulp, "gate3C_worst_relL2_K": worst_k, "gate3C_worst_relL2_VO": worst_vo,
                             "per_layer": per}
res["operator_parity_v3"]["PASS"] = ok3a and worst_ulp <= 1 and worst_k <= U and worst_vo <= U
del eb
print("parity", res["operator_parity_v3"]["PASS"], worst_ulp, worst_k, worst_vo, flush=True)


# 5 PPL integrity
def win(d): return {r["window_id"]: r["input_ids_sha256"] for r in map(json.loads, filter(str.strip, open(d / "ppl_windows.jsonl")))}
pp = {}
for ds, n in (("wiki", 141), ("c4", 256)):
    a = json.loads((O3S / "quality" / ds / "ppl_aggregate.json").read_text())
    w, wf = win(O3S / "quality" / ds), win(FULLQ / ds)
    pp[ds] = {"used": a["used_windows"], "skipped": a["skipped_windows"], "ppl": a.get("ppl"), "windows_and_input_sha_eq_full": w == wf,
              "PASS": (a["used_windows"], a["skipped_windows"]) == (n, 0) and w == wf}
res["ppl"] = pp

# 6 zero-shot sample ids
zs = {}
for t in (("piqa", "winogrande", "arc_easy", "arc_challenge", "hellaswag") if plan["acc"] else ()):
    f = lambda d: {r["sample_id"]: r["document_hash"] for r in map(json.loads, filter(str.strip, open(d / "zs_bf16/output/zero_shot_samples" / f"{t}_localjsonl.jsonl")))}
    a, b = f(O3S / "quality"), f(FULLQ)
    zs[t] = {"n": len(a), "ids_and_doc_hash_eq_full": a == b}
res["zero_shot"] = {"tasks": zs, "PASS": all(v["ids_and_doc_hash_eq_full"] for v in zs.values())} if plan["acc"] else {"not_run": "PPL-only arm", "PASS": True}
print("ppl/zs", {k: v["PASS"] for k, v in pp.items()}, res["zero_shot"]["PASS"], flush=True)

# 7 dual-process canary
outs = []
for i in (1, 2):
    o = O3S / f"canary_run{i}.json"
    subprocess.run([sys.executable, "-u", __file__, "--canary", str(fm), str(o)], check=True)
    outs.append(json.loads(o.read_text()))
keys = ("weights_digest", "input_ids_sha", "logits_sha", "mean_nll")
res["dual_process_canary"] = {"tokens": CANARY_TOKENS, "runs": outs, "match": {k: outs[0][k] == outs[1][k] for k in keys},
                              "distinct_pids": outs[0]["pid"] != outs[1]["pid"]}
res["dual_process_canary"]["PASS"] = all(res["dual_process_canary"]["match"].values()) and res["dual_process_canary"]["distinct_pids"]
gates = {"ranks": res["ranks"]["PASS"], "basis": res["basis"]["PASS"], "operator_parity_v3": res["operator_parity_v3"]["PASS"],
         "ppl_wiki": pp["wiki"]["PASS"], "ppl_c4": pp["c4"]["PASS"], "zero_shot_ids": res["zero_shot"]["PASS"], "dual_process_canary": res["dual_process_canary"]["PASS"]}
res["gates"] = gates; res["ALL_PASS"] = all(gates.values())
(O3S / f"VERIFY_{ARM}.json").write_text(json.dumps(res, indent=1))
print("GATES", gates, "ALL_PASS", res["ALL_PASS"], flush=True)
sys.exit(0 if res["ALL_PASS"] else 3)
