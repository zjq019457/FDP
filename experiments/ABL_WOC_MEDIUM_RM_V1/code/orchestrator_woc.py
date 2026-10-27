#!/usr/bin/env python3
"""ABL_WOC_MEDIUM_RM_V1 GPU-safe dynamic scheduler (derived from FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1/code/orchestrator.py).

Medium w/o Causal ablation rebuilt at the cache size of the resource-matched Full profile (frozen T06 pipeline: avg_dim 32,
frozen offline builder without floor, same lambda / normalisation / layer-0 rule). LongBench omitted.

Never kills or signals any process it did not start. Picks a GPU only when its free memory covers the
job's need and none of our jobs started on it in the last COOLDOWN seconds. Retries only recoverable
failures (OOM / CUDA alloc / disk full). Scientific gates (rank identity, replay agreement, window
counts) are never retried: a failure blocks the dependents and is written to FAILED_<job>.json.
"""
import datetime, hashlib, json, os, re, shutil, subprocess, sys, time, traceback
from pathlib import Path

ROOT = Path("/diskF/jiaqizhang/ABL_WOC_MEDIUM_RM_V1")
AVG_DIM = 32
sys.path.insert(0, str(ROOT / "code"))
import c45lib as L

PY = "/home/bai-project/anaconda3/envs/eigen/bin/python"
REPO = Path("/home/bai-project/jiaqizhang/EigenAttn")
EIGEN = REPO / "EigenAttn"
MODEL = Path("/diskD/jiaqizhang/Llama-3")
WIKI = Path("/home/bai-project/jiaqizhang/wikitext2")
C4 = Path("/diskD/jiaqizhang/llama3_8b_eb045_eval_minimal-20260502T170035Z-3-001/llama3_8b_eb045_eval_minimal/data/c4_ppl_selected_texts.jsonl")
BUILDER = Path("/diskF/jiaqizhang/FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1/code/build_eigen_basis_llama_error_budget_offline_c45.py")
BUILDER_SHA = "601bf0fdea31d85fe136c9b2ec044b7e4158f34704b9049bde638e74806dc210"
MAIN = EIGEN / "main_eigen_attn.py"
STEP1 = EIGEN / "step1_diagnose_llama.py"
PPL = REPO / "rebuttal_legacy_reproduction/runners/ppl_instrumented.py"
ZS = REPO / "rebuttal_legacy_reproduction/runners/zero_shot_instrumented.py"
LB = REPO / "rebuttal_completion_pipeline/runners/wave2_longbench_eigen.py"
LOCK = REPO / "rebuttal_legacy_reproduction/LEGACY_EVALUATION_PROTOCOL_LOCK.json"
ZS_DATA = Path("/diskF/jiaqizhang/a1804298693/offline_datasets")
LB_DATA5 = Path("/home/bai-project/jiaqizhang/EigenAttn/data/LongBench")
LB_DATA_NQA = Path("/diskE/jiaqizhang/FDP125_STRICT_FORMAL_CLOSURE_V1/longbench_qa6_v1/data")
LB_COUNTS = {"narrativeqa": 200, "qasper": 200, "multifieldqa_en": 150, "hotpotqa": 200, "2wikimqa": 200, "musique": 200}
SCR = {"E": Path("/diskE/jiaqizhang/ABL_WOC_MEDIUM_RM_V1_SCRATCH"), "F": Path("/diskF/jiaqizhang/ABL_WOC_MEDIUM_RM_V1_SCRATCH")}
OFFICIAL_EVAL = Path("/diskF/jiaqizhang/OFFICIAL_EIGENATTN_REPRO_V1/eval")
OFFICIAL_TAGS = ["p_eb0p090_ns512_ad4_save", "p_eb0p045_ns512_ad4_save", "p_eb0p060_ns512_ad4_save"]
COOLDOWN = 180
CKPT_GB = 16
MIN_FREE_GB = 4
RECOVERABLE = re.compile(r"out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDA error: out|No space left on device|OSError: \[Errno 28\]", re.I)

def now(): return datetime.datetime.now().astimezone().isoformat()
def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def wjson(p, o):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".tmp"); t.write_text(json.dumps(o, indent=1)); os.replace(t, p)
def rjson(p): return json.loads(Path(p).read_text())
def log(msg):
    with open(ROOT / "logs/orchestrator.log", "a") as f: f.write(f"{now()} {msg}\n")


def env_for(gpu, offline=True):
    e = dict(os.environ)
    e.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": f"{EIGEN}:{REPO}:{e.get('PYTHONPATH', '')}", "PYTHONHASHSEED": "2"})
    if offline: e.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    return e


def gpu_free():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines())}


def disk_free_gb(k): return shutil.disk_usage(SCR[k]).free / 1024 ** 3


# ----------------------------------------------------------------------------- job definitions
class Job:
    def __init__(self, name, deps, gpu_need=0, ckpt=False, cpu_fn=None, build=None, post=None, ready_fn=None, skip_fn=None):
        self.name, self.deps, self.gpu_need, self.ckpt, self.skip_fn = name, deps, gpu_need, ckpt, skip_fn
        self.cpu_fn, self.build, self.post, self.ready_fn = cpu_fn, build, post, ready_fn


JOBS = {}
def add(j): JOBS[j.name] = j
REC = ROOT / "records"


def builder_cmd(out_dir, prefix, eb_json, layer0, mt):
    return [PY, "-u", str(BUILDER), "--model", str(MODEL), "--data_dir", str(WIKI), "--out_dir", str(out_dir), "--out_prefix", prefix,
            "--per_layer_eb_json", str(eb_json), "--freeze_layer0_eb", "--layer0_eb", str(layer0), "--nsamples", "512", "--seqlen", "2048",
            "--dtype", "bf16", "--attn_implementation", "eager", "--device", "cuda:0", "--seed", "2", "--official_min_thresh", str(mt), "--avg_dim", str(AVG_DIM)]


def builder_ranks(out_dir, prefix):
    rep = rjson(Path(out_dir) / f"{prefix}_perlayer_eb_report.json")
    runs = [l["runs"][0] for l in rep["layers"]]
    return [[int(r["rank_kq_final"]), int(r["rank_v_final"])] for r in runs], [r.get("stop_reason") for r in runs]


def mk_builder_job(name, deps, prefix, eb_fn, layer0_fn, mt, check_fn, skip=None):
    """eb_fn() -> per-layer EB list (read at launch time); check_fn(ranks, stops) raises on gate failure."""
    work = REC / "basis" / name
    def build(gpu, scr):
        if fsha(BUILDER) != BUILDER_SHA: raise RuntimeError("C45 builder sha changed")
        bd = SCR[scr] / "basis" / name
        if bd.exists(): shutil.rmtree(bd)
        bd.mkdir(parents=True)
        ebj = work / "per_layer_eb.json"
        wjson(ebj, {"error_budget_by_layer": eb_fn(), "meta": {"job": name, "official_min_thresh": mt}})
        wjson(work / "basis_dir.json", {"basis_dir": str(bd)})
        return builder_cmd(bd, prefix, ebj, layer0_fn(), mt), env_for(gpu), EIGEN
    def post():
        bd = Path(rjson(work / "basis_dir.json")["basis_dir"])
        ranks, stops = builder_ranks(bd, prefix)
        shutil.copy2(bd / f"{prefix}_perlayer_eb_report.json", work / "builder_report.json")
        res = {"ranks": ranks, "stops": stops, "stop_dist": L.stopdist(stops), "B_live": L.B_of(ranks), "rank_sha_compact": L.json_sha(ranks),
               "basis_pt": str(bd / f"{prefix}_perlayer_eb.pt"), "basis_sha256": fsha(bd / f"{prefix}_perlayer_eb.pt"), "official_min_thresh": mt}
        wjson(work / "result.json", res)
        check_fn(ranks, stops)
    add(Job(name, deps, gpu_need=50000, build=build, post=post, skip_fn=skip))


def mk_mat_job(name, deps, basis_job, eb_label_fn, skip=None):
    work = REC / "models" / name
    def build(gpu, scr):
        b = rjson(REC / "basis" / basis_job / "result.json")
        md = SCR[scr] / "models" / name
        if md.exists(): shutil.rmtree(md)
        md.mkdir(parents=True)
        wjson(work / "model_dir.json", {"work": str(md), "final_model": str(md / "final_model"), "scratch": scr})
        cmd = [PY, "-u", str(MAIN), "--model", str(MODEL), "--net", "Llama-3-8b", "--dtype", "bf16", "--error_budget", str(eb_label_fn()), "--avg_dim", "4096",
               "--seqlen", "2048", "--nsamples", "64", "--seed", "2", "--calib_dataset", "c4", "--compress_only", "--attn_implementation", "eager",
               "--eigen_basis_path", b["basis_pt"], "--save_dir", str(md / "final_model"), "--output_dir", str(md), "--cache_dir", str(EIGEN / "HF_cache")]
        return cmd, env_for(gpu), EIGEN
    def post():
        b = rjson(REC / "basis" / basis_job / "result.json")
        fm = Path(rjson(work / "model_dir.json")["final_model"])
        ranks = rjson(fm / "low_rank_config.json")
        if ranks != b["ranks"]: raise GateFail(f"{name}: materialized ranks != builder ranks")
        shutil.copy2(fm / "low_rank_config.json", work / "low_rank_config.json")
        wjson(work / "result.json", {"final_model": str(fm), "ranks": ranks, "B_live": L.B_of(ranks), "low_rank_config_sha256": fsha(fm / "low_rank_config.json"),
                                     "rank_sha_compact": L.json_sha(ranks), "materialized_at": now()})
        bd = Path(b["basis_pt"]).parent
        if bd.exists(): shutil.rmtree(bd)            # rebuildable; report + sha kept in records/basis
    add(Job(name, deps, gpu_need=36000, ckpt=True, build=build, post=post, skip_fn=skip))


class GateFail(Exception):
    pass


def model_dir(mat_job): return rjson(REC / "models" / mat_job / "result.json")["final_model"]


def mk_eval_jobs(tier, method, mat_job, skip=None):
    out = ROOT / "tiers" / tier / method
    names = []
    for ds in ["wikitext2", "c4"]:
        n = f"ppl_{ds}_{tier}_{method}"
        def build(gpu, scr, ds=ds, n=n):
            od = out / f"ppl_{ds}_bf16"
            if od.exists(): shutil.rmtree(od)
            cmd = [PY, "-u", str(PPL), "--dataset", ds, "--model", model_dir(mat_job), "--tok", str(MODEL), "--out_dir", str(od), "--dtype", "bf16",
                   "--method", f"WOC_{tier}_{method}", "--evaluation_protocol_hash", fsha(LOCK)]
            cmd += ["--wikitext2_dir", str(WIKI)] if ds == "wikitext2" else ["--c4_texts", str(C4)]
            return cmd, env_for(gpu), REPO
        def post(ds=ds):
            a = rjson(out / f"ppl_{ds}_bf16/ppl_aggregate.json")
            want = (141, 0) if ds == "wikitext2" else (256, 0)
            if (a["used_windows"], a["skipped_windows"]) != want: raise GateFail(f"window integrity {ds}")
        add(Job(n, [mat_job], gpu_need=30000, build=build, post=post, skip_fn=skip)); names.append(n)
    n = f"zs_{tier}_{method}"
    def build_zs(gpu, scr):
        z = out / "zs_bf16"; c = SCR[scr] / "zs_cache" / f"{tier}_{method}"
        if z.exists(): shutil.rmtree(z)
        (c / "datasets").mkdir(parents=True, exist_ok=True); (c / "tmp").mkdir(parents=True, exist_ok=True)
        e = env_for(gpu, offline=False)
        e.update({"HF_HOME": str(c), "TRANSFORMERS_CACHE": str(c), "HF_DATASETS_CACHE": str(c / "datasets"), "TMPDIR": str(c / "tmp"), "TOKENIZERS_PARALLELISM": "false"})
        cmd = [PY, "-u", str(ZS), "--model", model_dir(mat_job), "--tokenizer_model", str(MODEL), "--task_yaml_dir", str(ZS_DATA),
               "--tasks", "piqa_localjsonl,winogrande_localjsonl,arc_easy_localjsonl,arc_challenge_localjsonl,hellaswag_localjsonl",
               "--out_dir", str(z / "output"), "--net", "Llama-3-8b", "--dtype", "bf16", "--method", f"WOC_{tier}_{method}", "--cache_dir", str(c),
               "--attn_implementation", "eager", "--batch_size", "8"]
        return cmd, e, REPO
    def post_zs():
        z = out / "zs_bf16/output"
        if not z.exists() or not any(z.iterdir()): raise GateFail("zs output missing")
        for k in SCR.values():
            shutil.rmtree(k / "zs_cache" / f"{tier}_{method}", ignore_errors=True)
    add(Job(n, [mat_job], gpu_need=30000, build=build_zs, post=post_zs, skip_fn=skip)); names.append(n)
    if False:  # AD512: LongBench omitted
        for part, tasks, data in [("lb5", "qasper,multifieldqa_en,hotpotqa,2wikimqa,musique", LB_DATA5), ("lbnqa", "narrativeqa", LB_DATA_NQA)]:
            n = f"{part}_{tier}_{method}"
            def build_lb(gpu, scr, tasks=tasks, data=data, part=part):
                od = out / "longbench_qa6" / part
                od.mkdir(parents=True, exist_ok=True)
                cmd = [PY, "-u", str(LB), "--method", f"c45_{method}", "--model", model_dir(mat_job), "--tok", str(MODEL), "--data_dir", str(data),
                       "--max_len", "8192", "--limit_per_task", "100000", "--tasks", tasks, "--out_dir", str(od)]
                return cmd, env_for(gpu), REPO
            def post_lb(tasks=tasks, part=part):
                for t in tasks.split(","):
                    p = out / "longbench_qa6" / part / f"{t}.jsonl"
                    rows = [x for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
                    if len(rows) != LB_COUNTS[t]: raise GateFail(f"longbench {t}: {len(rows)} != {LB_COUNTS[t]}")
            add(Job(n, [mat_job], gpu_need=45000, build=build_lb, post=post_lb)); names.append(n)
    cn = f"cleanup_{tier}_{method}"
    def clean():
        fm = Path(model_dir(mat_job))
        man = {"checkpoint": str(fm), "deleted_at": now(), "reason": "all dependent evaluations persisted and validated", "files": []}
        for f in sorted(fm.iterdir()):
            man["files"].append({"name": f.name, "bytes": f.stat().st_size, "sha256": fsha(f)})
        wjson(REC / "models" / mat_job / "CHECKPOINT_DELETION_MANIFEST.json", man)
        for keep in ["config.json", "low_rank_config.pt", "low_rank_config.json"]:
            if (fm / keep).exists(): shutil.copy2(fm / keep, REC / "models" / mat_job / keep)
        shutil.rmtree(fm.parent)
    add(Job(cn, names, cpu_fn=clean, skip_fn=skip))
    return names + [cn]


# ----------------------------------------------------------------------------- pipeline graph (Medium w/o Causal, resource-matched)
STEP1_MED = Path("/diskF/jiaqizhang/fdp125_final_campaign_v1/results/llama/medium/step1.json")
TARGET_B = 196640768          # resource-matched Full profile at Medium (TACL_LLAMA3_3COMP_5COMP_AUDIT/FULL_PROFILE_RESOURCE_ROBUSTNESS_V1)
FULL_RM_BASE = 0.06178259509322545


def plan_woc():
    """Quality-blind. w/o Causal = T06 profile without d2, weights (1,0,0,0,1); same lambda, normalisation and layer-0 rule;
    only the base tolerance is changed so that B matches the resource-matched Full profile (exact CPU replay, no floor)."""
    best = None
    for i in range(0, 6001):
        b = 0.06 + i * 2e-6
        eb = L.derive_eb(STEP1_MED, b, (1, 0, 0, 0, 1)); v = list(eb); v[0] = b
        r, s = L.select(v, False); B = L.B_of(r)
        key = (abs(B - TARGET_B), abs(b - FULL_RM_BASE))
        if best is None or key < best[0]: best = (key, b, eb, r, B)
    _, b, eb, r, B = best
    gap = 100 * (B - TARGET_B) / TARGET_B
    # gates: the same replay reproduces the resource-matched Full schedule and the fixed-budget w/o Causal schedule
    ef = L.derive_eb(STEP1_MED, FULL_RM_BASE, (1, 1, 0, 0, 1)); vf = list(ef); vf[0] = FULL_RM_BASE
    e6 = L.derive_eb(STEP1_MED, 0.06, (1, 0, 0, 0, 1)); v6 = list(e6); v6[0] = 0.06
    gates = {"full_rm_replay_B": L.B_of(L.select(vf, False)[0]), "woc_fixed_replay_B": L.B_of(L.select(v6, False)[0])}
    if gates != {"full_rm_replay_B": 196640768, "woc_fixed_replay_B": 199753728}: raise GateFail(f"replay gate {gates}")
    p = {"arm": "w/o Causal, resource-matched (Medium)", "weights": [1, 0, 0, 0, 1], "base": b, "eb_vector": eb, "ranks": r, "B": B, "target_B": TARGET_B,
         "gap_pct": gap, "class": L.match_class(gap), "gates": gates, "step1": str(STEP1_MED), "quality_scores_read": False, "at": now()}
    wjson(ROOT / "PLAN.json", p); os.chmod(ROOT / "PLAN.json", 0o444)
    if L.match_class(gap) != "MATCHED": raise GateFail(f"gap {gap}")
add(Job("plan_woc", [], cpu_fn=plan_woc))


def chk(ranks, stops):
    if ranks != rjson(ROOT / "PLAN.json")["ranks"]: raise GateFail("builder ranks != replay")
mk_builder_job("woc_basis_medium", ["plan_woc"], "woc_rm_medium", lambda: rjson(ROOT / "PLAN.json")["eb_vector"], lambda: rjson(ROOT / "PLAN.json")["base"], 0.0, chk)
mk_mat_job("woc_mat_medium", ["woc_basis_medium"], "woc_basis_medium", lambda: rjson(ROOT / "PLAN.json")["base"])
mk_eval_jobs("medium", "woc_rm", "woc_mat_medium")


# ----------------------------------------------------------------------------- scheduler
STATE_P = ROOT / "state/jobs.json"
state = rjson(STATE_P) if STATE_P.exists() else {}
for n in JOBS: state.setdefault(n, {"status": "PENDING", "attempts": 0})
for n, s in state.items():
    if s["status"] == "RUNNING":              # orchestrator restarted: the child is gone with it
        s["status"] = "PENDING"; s["note"] = "requeued after orchestrator restart"
for n in [x.strip() for x in os.environ.get("C45_REQUEUE", "").split(",") if x.strip()]:
    s = state[n]                              # explicit operator requeue of an engineering failure
    s.setdefault("history", []).append({"status": s["status"], "error": s.get("error"), "requeued_at": now()})
    s.update({"status": "PENDING", "attempts": 0}); s.pop("error", None)
for n, s in state.items():
    if s["status"] == "BLOCKED": s["status"] = "PENDING"
_changed = True
while _changed:                               # re-block anything still depending on an un-requeued failure
    _changed = False
    for m, j in JOBS.items():
        if state[m]["status"] == "PENDING" and any(state[d]["status"] in ("FAILED", "BLOCKED") for d in j.deps):
            state[m]["status"] = "BLOCKED"; _changed = True
running = {}
last_launch = {}
reserved_ckpt = {"E": 0, "F": 0}


def save_state():
    wjson(STATE_P, state)
    summ = {}
    for n, s in state.items(): summ.setdefault(s["status"], []).append(n)
    wjson(ROOT / "STATUS.json", {"at": now(), "counts": {k: len(v) for k, v in summ.items()}, "running": {n: state[n].get("gpu") for n in running},
                                  "failed": summ.get("FAILED", []) + summ.get("BLOCKED", []), "done": len(summ.get("DONE", []))})


def finish(n, ok, err=None):
    s = state[n]
    if ok:
        s["status"] = "DONE"; s["done_at"] = now(); log(f"DONE {n}")
    else:
        s["status"] = "FAILED"; s["error"] = err; log(f"FAILED {n}: {err}")
        wjson(ROOT / "logs" / f"FAILED_{n}.json", {"job": n, "error": err, "at": now()})
        changed = True
        while changed:
            changed = False
            for m, j in JOBS.items():
                if state[m]["status"] == "PENDING" and any(state[d]["status"] in ("FAILED", "BLOCKED") for d in j.deps):
                    state[m]["status"] = "BLOCKED"; changed = True


log("orchestrator start")
for k in SCR.values(): k.mkdir(parents=True, exist_ok=True)
(ROOT / "logs/jobs").mkdir(parents=True, exist_ok=True)
while True:
    # poll running
    for n in list(running):
        p, logp, scr = running[n]
        rc = p.poll()
        if rc is None: continue
        del running[n]
        j = JOBS[n]
        if j.ckpt: reserved_ckpt[scr] -= CKPT_GB
        if rc == 0:
            try:
                j.post(); finish(n, True)
            except GateFail as g:
                finish(n, False, f"GATE: {g}")
            except Exception:
                finish(n, False, "POST_ERROR: " + traceback.format_exc()[-1500:])
        else:
            tail = Path(logp).read_text(errors="ignore")[-20000:]
            st = state[n]
            if RECOVERABLE.search(tail) and st["attempts"] < 8:
                st["status"] = "PENDING"; st["retry_after"] = time.time() + 300
                shutil.copy2(logp, str(logp) + f".attempt{st['attempts']}_recoverable")
                log(f"RETRY {n} rc={rc} attempt={st['attempts']} (recoverable)")
            else:
                finish(n, False, f"rc={rc}; log={logp}")
        save_state()
    # launch ready
    free = None
    for n, j in JOBS.items():
        s = state[n]
        if s["status"] != "PENDING" or n in running: continue
        if time.time() < s.get("retry_after", 0): continue
        if not all(state[d]["status"] == "DONE" for d in j.deps): continue
        if j.ready_fn and not j.ready_fn(): continue
        if j.skip_fn:
            try:
                if j.skip_fn():
                    s["status"] = "DONE"; s["done_at"] = now(); s["note"] = "SKIPPED_ALIAS (identical schedule realized by an earlier job)"
                    log(f"SKIPPED {n} (alias)"); save_state(); continue
            except Exception:
                finish(n, False, "SKIP_ERROR: " + traceback.format_exc()[-1500:]); save_state(); continue
        if j.cpu_fn:
            try:
                j.cpu_fn(); finish(n, True)
            except GateFail as g:
                finish(n, False, f"GATE: {g}")
            except Exception:
                finish(n, False, "CPU_ERROR: " + traceback.format_exc()[-1500:])
            save_state(); continue
        free = free or gpu_free()
        gpu = None
        for g, f in sorted(free.items(), key=lambda kv: -kv[1]):
            if f >= j.gpu_need and time.time() - last_launch.get(g, 0) >= COOLDOWN: gpu = g; break
        if gpu is None: continue
        scr = "E"
        if j.ckpt:
            cands = [k for k in ("E", "F") if disk_free_gb(k) - reserved_ckpt[k] >= CKPT_GB + MIN_FREE_GB]
            if not cands: continue
            scr = cands[0]
        try:
            cmd, env, cwd = j.build(gpu, scr)
        except Exception:
            finish(n, False, "BUILD_ERROR: " + traceback.format_exc()[-1500:]); save_state(); continue
        s["attempts"] += 1
        logp = ROOT / "logs/jobs" / f"{n}.log"
        with open(logp, "w") as lf:
            lf.write(f"# {now()} gpu={gpu} scratch={scr}\n# {' '.join(map(str, cmd))}\n"); lf.flush()
            p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        running[n] = (p, logp, scr)
        if j.ckpt: reserved_ckpt[scr] += CKPT_GB
        last_launch[gpu] = time.time(); free[gpu] -= j.gpu_need
        s.update({"status": "RUNNING", "gpu": gpu, "pid": p.pid, "started_at": now(), "scratch": scr})
        log(f"START {n} gpu={gpu} pid={p.pid} scratch={scr} attempt={s['attempts']}")
        save_state()
    if not running and all(s["status"] in ("DONE", "FAILED", "BLOCKED") for s in state.values()):
        log("orchestrator finished"); save_state(); break
    time.sleep(30)
