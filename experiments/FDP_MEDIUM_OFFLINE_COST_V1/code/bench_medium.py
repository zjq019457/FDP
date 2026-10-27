#!/usr/bin/env python3
"""FDP_MEDIUM_OFFLINE_COST_V1 - one cold, real end-to-end Medium build with stage instrumentation.

Frozen pipeline (fdp125_final_campaign_v1/llama_fdp125_tier_worker.py, Medium, base_e 0.060) executed unchanged:
  FDP chain (GPU1):  B parent Uniform basis+rank search (eb 0.060)  -> C parent materialize -> D step1 profiling
                     -> E derive_eb (worker's own function) -> F FDP Pass-2 basis+rank search -> G FDP materialize
  Uniform comparator (GPU2, concurrently): frozen matched Uniform eb 0.0567 basis+rank search -> materialize
No PPL / zero-shot. Every artifact is checked against the frozen Medium SHAs afterwards.
"""
import hashlib, importlib.util, json, os, shutil, sys, threading, time, traceback
from pathlib import Path

ROOT = Path("/diskF/jiaqizhang/FDP_MEDIUM_OFFLINE_COST_V1")
sys.path.insert(0, str(ROOT / "code"))
from timed_stage import run_stage  # noqa: E402

PY = "/home/bai-project/anaconda3/envs/eigen/bin/python"
REPO = Path("/home/bai-project/jiaqizhang/EigenAttn"); EIGEN = REPO / "EigenAttn"
MODEL = Path("/diskD/jiaqizhang/Llama-3"); WIKI = Path("/home/bai-project/jiaqizhang/wikitext2")
BUILDER = EIGEN / "tools/build_eigen_basis_llama_error_budget_offline.py"; MAIN = EIGEN / "main_eigen_attn.py"; STEP1 = EIGEN / "step1_diagnose_llama.py"
WORKER = Path("/diskF/jiaqizhang/fdp125_final_campaign_v1/llama_fdp125_tier_worker.py")
LAUNCH = ROOT / "code/cost_launch.py"
W = ROOT / "work"; S = ROOT / "stages"
BUILDER_MARKERS = (r"\[per_layer_eb\] loaded", r"\[layer 00\]\[eb=", r"\[layer 31\] done", r"\[saved\]")
STEP1_MARKERS = (r"\[Phase 1\]", r"\[Phase 2\]", r"\[Phase 3\]", r"\[Phase 4\]")
MAT_MARKERS = (r"Loading checkpoint shards", r"(?i)saving|save_pretrained|saved")
RESULTS = {}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def jsha(o): return hashlib.sha256(json.dumps(o, separators=(",", ":")).encode()).hexdigest()
def wj(p, o): Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(json.dumps(o, indent=1))


def env(gpu, offline=True):
    e = dict(os.environ); e.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": f"{EIGEN}:{REPO}"})
    if offline: e.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    return e


def L(name, script, args):
    return [PY, "-u", str(LAUNCH), str(S / f"{name}.torchpeak.json"), str(script)] + [str(a) for a in args]


def builder(name, gpu, work, prefix, eb_vec, layer0):
    bd = work / "basis"; bd.mkdir(parents=True, exist_ok=True)
    ebj = work / "per_layer_eb.json"; wj(ebj, {"meta": {"benchmark": name}, "error_budget_by_layer": eb_vec})
    args = ["--model", MODEL, "--data_dir", WIKI, "--out_dir", bd, "--out_prefix", prefix, "--per_layer_eb_json", ebj, "--freeze_layer0_eb",
            "--layer0_eb", layer0, "--nsamples", 512, "--seqlen", 2048, "--dtype", "bf16", "--attn_implementation", "eager", "--device", "cuda:0", "--seed", 2]
    r = run_stage(name, L(name, BUILDER, args), env(gpu), EIGEN, S, BUILDER_MARKERS, [bd], S / f"{name}.torchpeak.json")
    RESULTS[name] = r
    return bd / f"{prefix}_perlayer_eb.pt"


def materialize(name, gpu, work, basis, eb, save_dir):
    args = ["--model", MODEL, "--net", "Llama-3-8b", "--dtype", "bf16", "--error_budget", eb, "--avg_dim", 4096, "--seqlen", 2048, "--nsamples", 64,
            "--seed", 2, "--calib_dataset", "c4", "--compress_only", "--attn_implementation", "eager", "--eigen_basis_path", basis,
            "--save_dir", save_dir, "--output_dir", work, "--cache_dir", EIGEN / "HF_cache"]
    RESULTS[name] = run_stage(name, L(name, MAIN, args), env(gpu), EIGEN, S, MAT_MARKERS, [save_dir, work / "final_model"], S / f"{name}.torchpeak.json")


def fdp_chain(gpu):
    pw = W / "fdp_chain/uniform_parent"; fw = W / "fdp_chain/fdp125"
    pb = builder("B_parent_uniform_basis_rank_search", gpu, pw, "fdp125_llama_medium_uniform", [0.06] * 32, 0.06)
    materialize("C_parent_uniform_materialize", gpu, pw, pb, 0.06, pw / "final_model")
    step1 = W / "fdp_chain/step1.json"
    args = ["--base", MODEL, "--cmp", pw / "final_model", "--tok", MODEL, "--dtype", "bf16", "--seqlen", 512, "--num_seqs", 2, "--out", step1,
            "--gpu", 0, "--attn_implementation", "eager"]
    RESULTS["D_fdp_step1_profiling"] = run_stage("D_fdp_step1_profiling", L("D_fdp_step1_profiling", STEP1, args), env(gpu, offline=False), EIGEN, S,
                                                 STEP1_MARKERS, [step1], S / "D_fdp_step1_profiling.torchpeak.json")
    t0 = time.perf_counter()
    spec = importlib.util.spec_from_file_location("frozen_worker", WORKER); wk = importlib.util.module_from_spec(spec); spec.loader.exec_module(wk)
    eb = wk.derive_eb(step1, wk.BASE_E["medium"], (1, 1, 0, 0, 1))
    RESULTS["E_fdp_score_aggregation_eb"] = {"stage": "E_fdp_score_aggregation_eb", "wall_s": time.perf_counter() - t0, "device": "CPU",
                                            "eb_vector": eb, "eb_sha": jsha([round(x, 8) for x in eb]), "worker_sha256": sha(WORKER)}
    wj(S / "E_fdp_score_aggregation_eb.timing.json", RESULTS["E_fdp_score_aggregation_eb"])
    fb = builder("F_fdp_pass2_basis_rank_search", gpu, fw, "fdp125_llama_medium_fdp125", eb, 0.06)
    materialize("G_fdp_materialize", gpu, fw, fb, 0.06, fw / "final_model")


def uniform_chain(gpu):
    uw = W / "uniform_comparator"
    ub = builder("U_uniform_comparator_basis_rank_search", gpu, uw, "official_uniform_medium_eb_0p0567_retry_gpu2", [0.0567] * 32, 0.0567)
    materialize("U_uniform_comparator_materialize", gpu, uw, ub, 0.0567, uw / "model")


def guard(fn, *a):
    try: fn(*a)
    except Exception: RESULTS.setdefault("errors", []).append(traceback.format_exc())


T0 = time.time()
threads = [threading.Thread(target=guard, args=(fdp_chain, 1)), threading.Thread(target=guard, args=(uniform_chain, 2))]
[t.start() for t in threads]; [t.join() for t in threads]
RESULTS["benchmark_wall_s"] = time.time() - T0
wj(ROOT / "MEDIUM_STAGE_TIMINGS_RAW.json", RESULTS)
print("BENCH_DONE", "errors" in RESULTS, flush=True)
