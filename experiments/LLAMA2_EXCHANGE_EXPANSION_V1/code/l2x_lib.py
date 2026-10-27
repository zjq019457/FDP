"""LLAMA2_EXCHANGE_EXPANSION_V1 shared helpers.

Reuses the frozen Llama-2 EigenAttention infrastructure unchanged (canonical in-memory materializer via the V5 patch,
legacy PPL scorer + frozen Wiki/C4 windows via V7, tuning-window scorer from the add-benefit runner, zero-shot base runner).
Only the rank schedule given to the materializer varies.
"""
import builtins, copy, csv, datetime, hashlib, importlib.util, json, os, pathlib, subprocess

ROOT = pathlib.Path("/diskF/jiaqizhang/LLAMA2_EXCHANGE_EXPANSION_V1")
M4 = pathlib.Path("/diskE/jiaqizhang/ICLR_FDP125_PAPER_EVIDENCE_V1/13_MODEL_MATRIX_4X3")
OP_DIR = M4 / "LLAMA2_OPERATING_POINT_E068_FDP070_CAPPED_V1"
V5_PATH = OP_DIR / "LLAMA2_E068_FDP070_CAPPED_CANARY_RUNNER_V5.py"
V7_PATH = OP_DIR / "LLAMA2_E068_FDP070_CAPPED_FORMAL_PPL_RUNNER_V7.py"
ADD_DIR = M4 / "LLAMA2_E068_NATIVE_ADD_BENEFIT_V1_ATTEMPT1_20261001"
ADD_RUNNER = ADD_DIR / "LLAMA2_E068_NATIVE_ADD_BENEFIT_RUNNER_V1.py"
ADD_LOCK = ADD_DIR / "LLAMA2_E068_NATIVE_ADD_BENEFIT_PROTOCOL_LOCK.json"
ZS_BASE = M4 / "cells/llama2_7b/medium/formal_accuracy_after_ppl_v3_20260925T103500Z/l2_medium_formal_zeroshot_v1.py"
REP = pathlib.Path("/diskF/jiaqizhang/fdp125_tacl_llama2_exact_3comp_pass1_v1_20260930")
UNI068_CSV = OP_DIR / "LLAMA2_E068_UNIFORM_FDP070_CAPPED_PER_LAYER.csv"
FUNC068 = ADD_DIR / "LLAMA2_E068_NATIVE_EXCHANGE_ITERATION_ATTEMPT4_FROZEN_SCHEDULE.json"
PROFILE068 = M4 / "LLAMA2_CLEAN_REDESIGN_UNIFORM068_PROFILE_V1_20261001T1516_ATTEMPT7/LLAMA2_CLEAN_REDESIGN_D1_D2_D5_PROFILE_V1.json"
QUANTUM, BPR, RMIN, RMAX = 32, 4096, 64, 4096
FULL = 32 * 2 * 4096 * BPR


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def sha_json(v): return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
def rjson(p): return json.loads(pathlib.Path(p).read_text())
def wjson(p, o):
    p = pathlib.Path(p); p.parent.mkdir(parents=True, exist_ok=True); t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(o, indent=1)); os.replace(t, p)
def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def B_of(pairs): return sum((k + v) * BPR for k, v in pairs)


def rmin(op):
    """Smallest legal rank: 64 (Algorithm 1); with the 45% rank floor ('f' operating points) a path must stay above 0.45*4096."""
    return 1856 if op.startswith("f") else RMIN


def uniform_pairs(op):
    """Uniform operating points: rank schedules of historical Llama-2 builder runs (same calibration, shared decomposition);
    'fXXX' = the same budget with the 45% rank floor (exact replay, l2x_floor_uniform.py)."""
    if op.startswith("f"):
        return rjson(ROOT / "schedules" / f"uniform_{op}" / "SCHEDULE.json")["rank_pairs"]
    if op == "e068":
        rows = list(csv.DictReader(open(UNI068_CSV)))
        return [[int(r["uniform068_QK_rank"]), int(r["uniform068_V_rank"])] for r in rows]
    f, i = {"e060": ("llama2_7b_uniform_pass1_e060_report.json", 0),
            "e090": ("eb070_eb090_shared_decomp/llama2_7b_uniform_pass1_eb070_eb090_report.json", 1)}[op]
    r = rjson(REP / f)
    return [[int(l["runs"][i]["rank_kq_final"]), int(l["runs"][i]["rank_v_final"])] for l in r["layers"]]


def legal(pairs, lo=RMIN): return all(lo <= x <= RMAX and x % QUANTUM == 0 for p in pairs for x in p)


# ------------------------------------------------------------------ in-memory model (frozen canonical materializer)
def materializer(physical_gpu):
    import torch
    from transformers.cache_utils import Cache
    builtins.Cache = Cache
    v5 = load_mod(V5_PATH, "l2x_v5")
    manifest, rows, checks = v5.load_locked_geometry()
    mod, arms, basis_meta = v5.patch_canonical_materializer(manifest, rows, checks)
    mod.GPU = 0
    mod.GPU_GATE_MIB = 20000  # two workers may share one 80-GB GPU; each caps its allocator fraction
    mod.gpu_free_mib = lambda: int(torch.cuda.mem_get_info(0)[0] // (1024 * 1024))
    return v5, mod, arms, basis_meta


def build(mod, arms, basis_meta, label, pairs):
    pairs = [[int(k), int(v)] for k, v in pairs]
    assert legal(pairs), "illegal rank schedule"
    s = sha_json(pairs)
    arms[label] = {"per_layer": [{"layer": i, "rank_kq": k, "rank_v": v} for i, (k, v) in enumerate(pairs)], "rank_pairs": pairs,
                   "B_live_actual": B_of(pairs), "rank_implied_B_live_bytes": B_of(pairs), "rank_sha256": s, "allocation_sha256": s, "realization_sha256": s}
    model, factor_calc = mod.build_model(label, basis_meta)
    if [[int(x["rank_kq"]), int(x["rank_v"])] for x in factor_calc] != pairs: raise RuntimeError("runtime rank schedule differs")
    return model


def tuning_windows(kind):
    """'std' = frozen historical 61 windows (32 WikiText-2 train + 29 C4 continuations of eval documents);
    'dd' = 32 identical WikiText-2 train windows + 29 C4 windows from validation documents not in the evaluation set."""
    add = load_mod(ADD_RUNNER, "l2x_add")
    profile, resource, tm, rows, pairs, windows, halves, hashes = add.load_sources()
    lock = rjson(ADD_LOCK)
    id_to_half = {wid: h for h, ids in lock["tuning_split"]["half_ids"].items() for wid in ids}
    for w in windows: w["_half"] = id_to_half[w["window_id"]]
    if kind == "std": return add, windows
    dd = [json.loads(l) for l in open(ROOT / "data/C4_DOC_DISJOINT_TUNING_WINDOWS.jsonl") if l.strip()]
    wiki = [w for w in windows if w["dataset"] == "wikitext2"]
    for i, w in enumerate(dd): w["_half"] = "A" if i % 2 == 0 else "B"
    return add, wiki + dd


def loss(res):
    t = res["A"]["token_count"] + res["B"]["token_count"]
    return {"full": (res["A"]["nll_sum"] + res["B"]["nll_sum"]) / t, "A": res["A"]["mean_nll"], "B": res["B"]["mean_nll"]}


def gpu_snapshot():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    return {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines())}
