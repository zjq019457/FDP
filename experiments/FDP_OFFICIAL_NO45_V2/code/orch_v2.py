#!/usr/bin/env python3
"""FDP_OFFICIAL_NO45_V2 pipeline orchestrator (see PREREGISTRATION.json, read-only).

Stage 0  wait_trace: no45 full candidate trace + upstream eigenvalues; gates (prefix = V1 curves, official replay = official Uniform ranks).
Stage 1  Light/Medium: plan (quality-blind) -> build FDP-EB / Spectral-tail (official pipeline, no45) -> PPL + zero-shot -> cleanup
         gate_stage1: STOP if FDP-EB is worse than Uniform or Spectral-tail on J or five-task average at Light or Medium.
Stage 2  (only after PASS) Aggressive FDP/Spectral; Severe Uniform(no45) -> step1 -> FDP/Spectral; ablations at all four settings.
Files: state/jobs.json, logs/jobs/<job>.log, records/<job>/..., STATUS.json; STOP file halts launches; PAUSE file pauses launches."""
import datetime, hashlib, json, math, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plan_lib as PL  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

N = PL.N; REC = N / "records"; LOGS = N / "logs" / "jobs"; STATE_P = N / "state" / "jobs.json"
SCR = Path("/diskF/jiaqizhang/FDP_OFFICIAL_NO45_V2_SCRATCH")
PY = "/home/bai-project/anaconda3/envs/eigen/bin/python"
REPO = Path("/home/bai-project/jiaqizhang/EigenAttn"); EIGEN = REPO / "EigenAttn"
MODEL = "/diskD/jiaqizhang/Llama-3"
SHIM = N / "code/run_shim_v2.py"; UP_NO45 = N / "code/upstream_no45"
PPL = REPO / "rebuttal_legacy_reproduction/runners/ppl_instrumented.py"
ZS = REPO / "rebuttal_legacy_reproduction/runners/zero_shot_instrumented.py"
STEP1 = EIGEN / "step1_diagnose_llama.py"
C4 = "/diskD/jiaqizhang/llama3_8b_eb045_eval_minimal-20260502T170035Z-3-001/llama3_8b_eb045_eval_minimal/data/c4_ppl_selected_texts.jsonl"
ZS_DATA = "/diskF/jiaqizhang/a1804298693/offline_datasets"
OFF = Path("/diskF/jiaqizhang/OFFICIAL_EIGENATTN_REPRO_V1/eval")
OFF_TAG = {"light": "p_eb0p045_ns512_ad4_save", "medium": "p_eb0p060_ns512_ad4_save", "aggressive": "p_eb0p090_ns512_ad4_save"}
WINDOWS_SHA = "1b6ad8b1a345b6fb61101f34df02115721e2b40618eac069d783b428ceb58390"
SEVERE_TARGET = 151289856
TASKS = ["arc_challenge", "arc_easy", "hellaswag", "piqa", "winogrande"]
ABL = {"abl_noZ": (0, 0, 0, 0, 1), "abl_noH": (1, 0, 0, 0, 0), "abl_plusd2": (1, 1, 0, 0, 1)}
MAX_CKPTS = 3


class GateFail(Exception): pass
class NotReady(Exception): pass


def now(): return datetime.datetime.now().astimezone().isoformat()
def rjson(p): return json.loads(Path(p).read_text())
def wjson(p, o):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); t = p.with_suffix(p.suffix + ".tmp"); t.write_text(json.dumps(o, indent=1)); os.replace(t, p)
def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
def log(msg):
    with open(N / "logs" / "orchestrator.log", "a") as f: f.write(f"{now()} {msg}\n")
def readonly(p): os.chmod(p, 0o444)


def env_for(gpu, offline=True):
    e = dict(os.environ); e.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONHASHSEED": "2", "TOKENIZERS_PARALLELISM": "false"})
    if offline: e.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    e["PYTHONPATH"] = f"{EIGEN}:{REPO}:{e.get('PYTHONPATH', '')}"
    return e


class Job:
    def __init__(self, name, deps, gpu_need=0, heavy=False, ckpt=False, build=None, post=None, cpu_fn=None):
        self.name, self.deps, self.gpu_need, self.heavy, self.ckpt, self.build, self.post, self.cpu_fn = name, deps, gpu_need, heavy, ckpt, build, post, cpu_fn


JOBS = {}
def add(j): JOBS[j.name] = j


# ------------------------------------------------------------------ stage 0
def curves():
    return rjson(REC / "wait_trace" / "curves.json")


TRACE_PARTS = [("fulltrace_no45", 0, 23), ("trace_part_24_27", 24, 27), ("trace_part_28_31", 28, 31)]
# user decision 10:22: the trace may use 3 GPUs (model builds are never split). fulltrace_no45 is stopped with SIGINT
# after layer 23 (non-zero rc expected; its atexit trace holds layers 0-23); completeness is gated per layer below.


def wait_trace():
    """Merge the no45 full-search layers of three parallel runs (fulltrace_no45 is stopped with SIGINT after layer 15;
    its atexit trace holds layers 0-15 complete). Each run fast-forwards the other layers with one candidate, which must
    equal the full-search run's first candidate for that layer (cross-run determinism gate)."""
    d = (N / "logs" / "driver.log").read_text()
    if "END evals_official rc=" in d and "END evals_official rc=0" not in d: raise GateFail("eigenvalue run failed")
    for name, lo, hi in TRACE_PARTS[1:]:
        if f"END {name} rc=" in d and f"END {name} rc=0" not in d: raise GateFail(f"{name} failed")
    if not (N / "runs" / "fulltrace_no45" / "official_trace.json").exists() and "END fulltrace_no45 rc=" in d: raise GateFail("fulltrace_no45 wrote no trace")
    if "END evals_official rc=0" not in d or any(f"END {n} rc=" not in d for n, _, _ in TRACE_PARTS): raise NotReady()
    by_run = {}
    for name, lo, hi in TRACE_PARTS:
        tr = rjson(N / "runs" / name / "official_trace.json")
        g = {}
        for c in tr["candidate_trace"]: g.setdefault(c["layer"], []).append(c)
        by_run[name] = (tr, g)
    cur = []
    for li in range(32):
        src = next(n for n, lo, hi in TRACE_PARTS if lo <= li <= hi)
        cs = by_run[src][1][li]
        if any("err" not in c for c in cs[:-1]): raise GateFail(f"layer {li}: candidate without error")
        cur.append({"loop": [[c["rank_kq"], c["rank_v"], c["err"], c["err_dtype"], c["threshold_after"]] for c in cs[:-1]],
                    "final": [cs[-1]["rank_kq"], cs[-1]["rank_v"]], "source": src})
        for other in by_run:
            oc = by_run[other][1].get(li)
            if other != src and oc and "err" in oc[0]:
                if [oc[0]["rank_kq"], oc[0]["rank_v"], oc[0]["err"]] != cur[li]["loop"][0][:3]:
                    raise GateFail(f"layer {li}: first candidate differs between {src} and {other}")
    tr = by_run["fulltrace_no45"][0]
    for name, _, _ in TRACE_PARTS:
        t2 = by_run[name][0]
        if (t2["upstream"], t2["upstream_sha256"], t2["shim_sha256"]) != (tr["upstream"], tr["upstream_sha256"], tr["shim_sha256"]): raise GateFail("code differs between trace parts")
    union = PL.union_v1_curves()
    gate = {"upstream": tr["upstream"], "upstream_sha256": tr["upstream_sha256"], "shim_sha256": tr["shim_sha256"],
            "sources": TRACE_PARTS, "cross_run_first_candidate_equal": True}
    for li in range(32):
        a, b = union[li]["loop"], cur[li]["loop"]
        if len(b) < len(a) or any(a[k][:3] != b[k][:3] for k in range(len(a))): raise GateFail(f"layer {li}: full trace prefix != V1 official curves")
        if not (b[-1][4] > 0.3 and b[-1][4] - 0.02 <= 0.3 + 1e-9): raise GateFail(f"layer {li}: trace did not run to the threshold floor ({b[-1][4]})")
        if list(cur[li]["final"]) != list(b[-2][:2]): raise GateFail(f"layer {li}: floor step-back semantics not reproduced")
    gate["prefix_equal_v1"] = True; gate["floor_semantics_ok"] = True
    gate["official_replay"] = {}
    for t, eb in PL.OFFICIAL_EB.items():
        par = rjson(PL.V1 / "records" / f"parent_{t}" / "result.json")
        r = PL.replay(cur, [eb] * 32, "official", complete=True)
        if r != par["ranks"]: raise GateFail(f"{t}: official replay on the full trace != official Uniform ranks")
        gate["official_replay"][t] = {"B": PL.B_of(r), "rank_sha": PL.jsha(r)}
    e1 = torch.load(N / "runs/evals_official/evals.pt")
    if sorted(e1) != list(range(32)): raise GateFail("eigenvalue dump incomplete")
    for name, _, _ in TRACE_PARTS:
        p = N / "runs" / name / "evals.pt"
        if p.exists():
            e2 = torch.load(p)
            for i in e2:
                if not (torch.equal(e1[i]["eval_kq"], e2[i]["eval_kq"]) and torch.equal(e1[i]["eval_v"], e2[i]["eval_v"])):
                    raise GateFail(f"eigenvalues differ between the official run and {name} at layer {i}")
    gate["evals_equal_official_no45"] = True
    gate["n_candidates"] = [len(c["loop"]) for c in cur]
    wjson(REC / "wait_trace" / "curves.json", cur); wjson(REC / "wait_trace" / "GATE.json", gate)
    shutil.copy2(N / "runs/evals_official/evals.pt", REC / "wait_trace" / "evals.pt")


add(Job("wait_trace", [], cpu_fn=wait_trace))


# ------------------------------------------------------------------ planning (quality-blind)
def step1_path(s):
    return PL.V1 / "tiers" / s / "step1.json" if s != "severe" else REC / "step1_severe" / "step1.json"


def uniform_of(s):
    if s == "severe":
        r = rjson(REC / "build_uni_severe" / "result.json"); return r["ranks"], r["B_live"], rjson(REC / "plan_uni_severe" / "PLAN.json")["eb"]
    par = rjson(PL.V1 / "records" / f"parent_{s}" / "result.json"); return par["ranks"], par["B_live"], PL.OFFICIAL_EB[s]


def plan_setting(s):
    cur = curves(); U, BU, eb = uniform_of(s)
    out = {"setting": s, "quality_scores_read": False, "uniform_ranks_sha": PL.jsha(U), "target_B": BU, "prefer_base": eb, "arms": {}}
    qs = {"fdp": PL.fdp_q(step1_path(s)), "spec": PL.spectral_q(REC / "wait_trace" / "evals.pt", U)[0]}
    for a, w in ABL.items(): qs[a] = PL.fdp_q(step1_path(s), w)
    for a, q in qs.items():
        m = PL.match_exact(cur, q, BU, eb, lo=0.3 * eb, hi=3.0 * eb)
        if m["class"] == "NO_MATCH": raise GateFail(f"{s}/{a}: no matched schedule")
        out["arms"][a] = dict(m, q=q, q_sha=PL.jsha(q))
    p = REC / f"plan_{s}" / "PLAN.json"
    if p.exists(): raise GateFail(f"refuse to overwrite frozen plan {p}")
    wjson(p, out); readonly(p)


def plan_uni_severe():
    cur = curves()
    errs = sorted({c[2] for L in cur for c in L["loop"]})
    pts = {}
    for e in [errs[0] / 2] + [(x + y) / 2 for x, y in zip(errs, errs[1:])]:
        r = PL.replay(cur, [e] * 32, "no45", complete=True)
        pts.setdefault(PL.B_of(r), (e, r))
    B, (e, r) = min(pts.items(), key=lambda kv: (abs(kv[0] - SEVERE_TARGET), kv[0]))
    p = REC / "plan_uni_severe" / "PLAN.json"
    if p.exists(): raise GateFail("refuse to overwrite")
    wjson(p, {"quality_scores_read": False, "rule": "single eb (no45) with B nearest 151,289,856", "eb": e, "B": B, "ratio": B / 268435456,
              "gap_pct": 100 * (B - SEVERE_TARGET) / SEVERE_TARGET, "ranks": r, "rank_sha": PL.jsha(r)}); readonly(p)


# ------------------------------------------------------------------ build (official pipeline via shim v2)
def mk_build(name, deps, plan_fn, upstream_no45=True):
    """plan_fn() -> (cli_eb, per_layer_eb or None, expected_ranks)."""
    rec = REC / name
    def build(gpu):
        run = SCR / "runs" / name; md = SCR / "models" / name / "final_model"
        for p in (run, md.parent):
            if p.exists(): shutil.rmtree(p)
        (run / "cache").mkdir(parents=True); md.mkdir(parents=True)
        cli, pl, exp = plan_fn()
        e = env_for(gpu); e["EIGENATTN_TRACE_OUT"] = str(run / "official_trace.json")
        if upstream_no45: e["EIGENATTN_UPSTREAM"] = str(UP_NO45)
        if pl is not None:
            wjson(rec / "per_layer_eb.json", {"error_budget_by_layer": pl}); e["EIGENATTN_PER_LAYER_EB_JSON"] = str(rec / "per_layer_eb.json")
        wjson(rec / "paths.json", {"run": str(run), "final_model": str(md), "expected_rank_sha": PL.jsha(exp), "cli_eb": cli})
        return [PY, "-u", str(SHIM), "--model", MODEL, "--calib_dataset", "wikitext2", "--nsamples", "512", "--seed", "2", "--avg_dim", "4",
                "--error_budget", repr(float(cli)), "--attn_implementation", "eager", "--output_dir", str(run), "--cache_dir", str(run / "cache"),
                "--save_dir", str(md)], e, str(N)
    def post():
        p = rjson(rec / "paths.json"); run, md = Path(p["run"]), Path(p["final_model"])
        tr, cur = PL.load_trace_curves(run / "official_trace.json")
        ranks = [list(c["final"]) for c in cur]
        lrc = [[int(a), int(b)] for a, b in torch.load(md / "low_rank_config.pt")]
        if lrc != ranks: raise GateFail(f"{name}: saved ranks != traced final ranks")
        if PL.jsha(ranks) != p["expected_rank_sha"]: raise GateFail(f"{name}: realized ranks != planned ranks (replay gate)")
        d = torch.load(run / "cache/dataloader_Llama_wikitext2_512.cache"); h = hashlib.sha256()
        for inp, _ in d: h.update(inp.numpy().astype("int64").tobytes())
        if (h.hexdigest(), len(d)) != (WINDOWS_SHA, 512): raise GateFail(f"{name}: calibration windows differ from the audited paper protocol")
        if tr["upstream"] != (str(UP_NO45) if upstream_no45 else "/diskF/jiaqizhang/OFFICIAL_EIGENATTN_REPRO_V1/upstream"): raise GateFail("wrong upstream")
        wjson(rec / "result.json", {"final_model": str(md), "ranks": ranks, "B_live": PL.B_of(ranks), "ratio": PL.B_of(ranks) / 268435456,
                                    "rank_sha": PL.jsha(ranks), "upstream": tr["upstream"], "upstream_sha256": tr["upstream_sha256"],
                                    "shim_sha256": tr["shim_sha256"], "windows_sha256": WINDOWS_SHA, "at": now()})
        wjson(rec / "low_rank_config.json", ranks)
        shutil.copy2(run / "official_trace.json", rec / "official_trace.json")
        shutil.rmtree(run, ignore_errors=True)
    add(Job(name, deps, gpu_need=45000, heavy=True, ckpt=True, build=build, post=post))


def arm_plan(s, a):
    def f():
        pk = rjson(REC / f"plan_{s}" / "PLAN.json")["arms"][a]["pick"]; return pk["base"], pk["eps"], pk["ranks"]
    return f


def model_dir(job): return rjson(REC / job / "result.json")["final_model"]


# ------------------------------------------------------------------ evaluation
def out_dir(job): return N / "quality" / job


def mk_evals(job):
    names = []
    for ds in ("wikitext2", "c4"):
        n = f"ppl_{ds}_{job}"
        def build(gpu, ds=ds):
            od = out_dir(job) / f"ppl_{ds}_bf16"
            if od.exists(): shutil.rmtree(od)
            cmd = [PY, "-u", str(PPL), "--dataset", ds, "--model", model_dir(job), "--tok", MODEL, "--out_dir", str(od), "--dtype", "bf16", "--method", f"NO45V2_{job}_bf16"]
            if ds == "c4": cmd += ["--c4_texts", C4]
            return cmd, env_for(gpu), str(REPO)
        def post(ds=ds):
            a = rjson(out_dir(job) / f"ppl_{ds}_bf16" / "ppl_aggregate.json")
            if (a["used_windows"], a["skipped_windows"]) != ((141, 0) if ds == "wikitext2" else (256, 0)): raise GateFail("window integrity")
        add(Job(n, [job], gpu_need=26000, build=build, post=post)); names.append(n)
    n = f"zs_{job}"
    def build_zs(gpu):
        z = out_dir(job) / "zs_bf16"
        if z.exists(): shutil.rmtree(z)
        (z / "cache/datasets").mkdir(parents=True); (z / "tmp").mkdir()
        e = env_for(gpu, offline=False)
        e.update({"HF_HOME": str(z / "cache"), "TRANSFORMERS_CACHE": str(z / "cache"), "HF_DATASETS_CACHE": str(z / "cache/datasets"), "TMPDIR": str(z / "tmp")})
        return [PY, "-u", str(ZS), "--model", model_dir(job), "--tokenizer_model", MODEL, "--task_yaml_dir", ZS_DATA,
                "--tasks", "piqa_localjsonl,winogrande_localjsonl,arc_easy_localjsonl,arc_challenge_localjsonl,hellaswag_localjsonl",
                "--out_dir", str(z / "output"), "--net", "Llama-3-8b", "--dtype", "bf16", "--method", f"NO45V2_{job}_bf16",
                "--cache_dir", str(z / "cache"), "--attn_implementation", "eager", "--batch_size", "8"], e, str(REPO)
    def post_zs():
        z = out_dir(job) / "zs_bf16"
        if not (z / "output/zero_shot_aggregate.json").exists(): raise GateFail("zs aggregate missing")
        shutil.rmtree(z / "cache", ignore_errors=True); shutil.rmtree(z / "tmp", ignore_errors=True)
    add(Job(n, [job], gpu_need=26000, build=build_zs, post=post_zs)); names.append(n)
    return names


def mk_cleanup(job, deps):
    def clean():
        fm = Path(model_dir(job))
        if not fm.exists(): return
        man = {"checkpoint": str(fm), "deleted_at": now(), "reason": "all dependent evaluations persisted and validated",
               "files": [{"name": f.name, "bytes": f.stat().st_size, "sha256": fsha(f)} for f in sorted(fm.iterdir())]}
        wjson(REC / job / "CHECKPOINT_DELETION_MANIFEST.json", man)
        shutil.rmtree(fm.parent)
    add(Job(f"cleanup_{job}", deps, cpu_fn=clean))


# ------------------------------------------------------------------ paired statistics
def eval_dirs(arm, s):
    if arm == "uniform" and s != "severe":
        b = OFF / OFF_TAG[s]; return b / "ppl_wikitext2_bf16", b / "ppl_c4_bf16", b / "zs_bf16/output/zero_shot_samples"
    job = {"uniform": "build_uni_severe"}.get(arm, f"build_{arm}_{s}")
    b = out_dir(job); return b / "ppl_wikitext2_bf16", b / "ppl_c4_bf16", b / "zs_bf16/output/zero_shot_samples"


def windows(d): return {r["window_id"]: (r["total_nll"], r["token_count"], r["input_ids_sha256"]) for r in map(json.loads, filter(str.strip, open(Path(d) / "ppl_windows.jsonl")))}
def items(d, t):
    p = list(Path(d).glob(f"{t}_localjsonl*.jsonl")); assert len(p) == 1, (d, t)
    return {r["sample_id"]: (r["acc"] if "acc" in r else float(r["correct"]), r["document_hash"]) for r in map(json.loads, filter(str.strip, open(p[0])))}


def summary(arm, s):
    w, c, z = eval_dirs(arm, s)
    W, C = windows(w), windows(c); Z = {t: items(z, t) for t in TASKS}
    pw = math.exp(sum(v[0] for v in W.values()) / sum(v[1] for v in W.values())); pc = math.exp(sum(v[0] for v in C.values()) / sum(v[1] for v in C.values()))
    acc = {t: 100 * float(np.mean([v[0] for v in Z[t].values()])) for t in TASKS}
    return {"W": W, "C": C, "Z": Z, "wiki": pw, "c4": pc, "J": 0.5 * (math.log(pw) + math.log(pc)), "acc": acc, "avg": float(np.mean(list(acc.values())))}


def paired(A, B):
    out = {}; reps = {}
    for key, ds, seed in (("W", "wikitext2", 20261012), ("C", "c4", 20261013)):
        a, b = A[key], B[key]; ids = sorted(a)
        if ids != sorted(b) or any(a[i][2] != b[i][2] for i in ids): raise GateFail("window identity mismatch")
        na = np.array([a[i][0] for i in ids]); nb = np.array([b[i][0] for i in ids]); tk = np.array([a[i][1] for i in ids], float)
        rng = np.random.Generator(np.random.PCG64(seed)); idx = rng.integers(0, len(ids), size=(10000, len(ids)))
        bs = (na[idx].sum(1) - nb[idx].sum(1)) / tk[idx].sum(1); reps[ds] = bs
        out[ds] = {"delta": float((na.sum() - nb.sum()) / tk.sum()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    j = (reps["wikitext2"] + reps["c4"]) / 2
    out["J"] = {"delta": (out["wikitext2"]["delta"] + out["c4"]["delta"]) / 2, "ci95": [float(np.percentile(j, 2.5)), float(np.percentile(j, 97.5))]}
    rng = np.random.Generator(np.random.PCG64(20261014)); boots = []; per = {}
    for t in TASKS:
        a, b = A["Z"][t], B["Z"][t]; ids = sorted(a)
        if ids != sorted(b) or any(a[i][1] != b[i][1] for i in ids): raise GateFail(f"item identity mismatch {t}")
        xa = np.array([a[i][0] for i in ids], float); xb = np.array([b[i][0] for i in ids], float)
        idx = rng.integers(0, len(ids), size=(10000, len(ids))); bs = 100 * (xa[idx].mean(1) - xb[idx].mean(1)); boots.append(bs)
        per[t] = {"delta": float(100 * (xa.mean() - xb.mean())), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    m = np.mean(boots, axis=0)
    out["acc_tasks"] = per; out["avg"] = {"delta": float(np.mean([v["delta"] for v in per.values()])), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}
    return out


def setting_report(s, arms):
    S = {a: summary(a, s) for a in arms}
    rep = {"setting": s, "arms": {a: {k: S[a][k] for k in ("wiki", "c4", "J", "acc", "avg")} for a in arms}, "paired": {}}
    for a, b in (("fdp", "uniform"), ("fdp", "spec"), ("spec", "uniform")):
        if a in S and b in S: rep["paired"][f"{a}_minus_{b}"] = paired(S[a], S[b])
    for a in arms:
        if a.startswith("abl_"): rep["paired"][f"{a}_minus_fdp"] = paired(S[a], S["fdp"])
    return rep


def gate_stage1():
    res = {"rule": rjson(N / "PREREGISTRATION.json")["stop_condition"]["rule"], "settings": {}, "triggers": []}
    for s in ("light", "medium"):
        r = setting_report(s, ["uniform", "spec", "fdp"]); res["settings"][s] = r
        A = r["arms"]
        for other in ("uniform", "spec"):
            if A["fdp"]["J"] > A[other]["J"]: res["triggers"].append(f"{s}: FDP-EB J {A['fdp']['J']:.4f} > {other} {A[other]['J']:.4f}")
            if A["fdp"]["avg"] < A[other]["avg"]: res["triggers"].append(f"{s}: FDP-EB avg {A['fdp']['avg']:.2f} < {other} {A[other]['avg']:.2f}")
    res["decision"] = "STOP" if res["triggers"] else "PASS"; res["at"] = now()
    wjson(N / "STAGE1_GATE.json", res); readonly(N / "STAGE1_GATE.json")
    if res["triggers"]:
        (N / "STOP").write_text("stage-1 stop condition triggered:\n" + "\n".join(res["triggers"]) + "\n")
        raise GateFail("STOP: " + "; ".join(res["triggers"]))


def final_report():
    out = {"at": now(), "settings": {}}
    for s in ("light", "medium", "aggressive", "severe"):
        arms = ["uniform", "spec", "fdp"] + list(ABL)
        out["settings"][s] = setting_report(s, arms)
    wjson(N / "FINAL_RESULTS.json", out)


# ------------------------------------------------------------------ job graph
add(Job("plan_light", ["wait_trace"], cpu_fn=lambda: plan_setting("light")))
add(Job("plan_medium", ["wait_trace"], cpu_fn=lambda: plan_setting("medium")))
stage1_evals = []
for s in ("light", "medium"):
    for a in ("fdp", "spec"):
        j = f"build_{a}_{s}"; mk_build(j, [f"plan_{s}"], arm_plan(s, a)); ev = mk_evals(j); mk_cleanup(j, ev); stage1_evals += ev
add(Job("gate_stage1", stage1_evals, cpu_fn=gate_stage1))

add(Job("plan_aggressive", ["wait_trace"], cpu_fn=lambda: plan_setting("aggressive")))   # AMENDMENT_1: stage 2 no longer waits for gate_stage1
add(Job("plan_uni_severe", ["wait_trace"], cpu_fn=plan_uni_severe))
mk_build("build_uni_severe", ["plan_uni_severe"], lambda: (rjson(REC / "plan_uni_severe" / "PLAN.json")["eb"], None, rjson(REC / "plan_uni_severe" / "PLAN.json")["ranks"]))
uni_sev_ev = mk_evals("build_uni_severe")


def build_step1(gpu):
    o = REC / "step1_severe" / "step1.json"; o.parent.mkdir(parents=True, exist_ok=True)
    lj = REC / "step1_severe" / "low_rank_config.json"; wjson(lj, rjson(REC / "build_uni_severe" / "result.json")["ranks"])
    return [PY, "-u", str(STEP1), "--base", MODEL, "--cmp", model_dir("build_uni_severe"), "--tok", MODEL, "--low_json", str(lj), "--dtype", "bf16",
            "--seqlen", "512", "--num_seqs", "2", "--out", str(o), "--gpu", "0", "--attn_implementation", "eager"], env_for(gpu, offline=False), str(EIGEN)


def post_step1():
    if not (REC / "step1_severe" / "step1.json").exists(): raise GateFail("step1 missing")


add(Job("step1_severe", ["build_uni_severe"], gpu_need=30000, build=build_step1, post=post_step1))
mk_cleanup("build_uni_severe", uni_sev_ev + ["step1_severe"])
add(Job("plan_severe", ["step1_severe"], cpu_fn=lambda: plan_setting("severe")))

final_deps = list(uni_sev_ev)
# AMENDMENT_1 (user, 12:14): stage 2 runs in parallel with stage 1; main arms of Aggressive/Severe are queued before the ablations.
order = [(s, a) for s in ("aggressive", "severe") for a in ("fdp", "spec")] + [(s, a) for s in ("light", "medium", "aggressive", "severe") for a in ABL]
for s, a in order:
    j = f"build_{a}_{s}"
    mk_build(j, [f"plan_{s}"], arm_plan(s, a)); ev = mk_evals(j); mk_cleanup(j, ev); final_deps += ev
add(Job("final_report", final_deps + stage1_evals, cpu_fn=final_report))


# ------------------------------------------------------------------ scheduler
def gpu_free():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines())}


def alive(pid):
    try: os.kill(int(pid), 0); return True
    except Exception: return False


def main():
    LOGS.mkdir(parents=True, exist_ok=True); SCR.mkdir(parents=True, exist_ok=True)
    state = rjson(STATE_P) if STATE_P.exists() else {}
    for n in JOBS: state.setdefault(n, {"status": "PENDING"})
    for n, s in state.items():
        if s["status"] == "RUNNING" and not alive(s.get("pid", -1)) and not (LOGS / f"{n}.rc").exists():
            s["status"] = "PENDING"; s["note"] = "requeued after orchestrator restart"
    log("orchestrator start")
    last_launch = 0.0
    while True:
        # finished processes
        for n, s in state.items():
            if s["status"] != "RUNNING": continue
            rc_f = LOGS / f"{n}.rc"
            if not rc_f.exists():
                if alive(s["pid"]): continue
                time.sleep(2)
                if not rc_f.exists(): s.update(status="FAILED", error="process vanished without rc"); log(f"FAILED {n}: vanished"); continue
            rc = int(rc_f.read_text().strip() or 1)
            if rc != 0: s.update(status="FAILED", error=f"rc={rc}"); log(f"FAILED {n} rc={rc}"); continue
            try: JOBS[n].post(); s["status"] = "DONE"; s["done_at"] = now(); log(f"DONE {n}")
            except Exception as e: s.update(status="FAILED", error="POST: " + "".join(traceback.format_exception_only(type(e), e)).strip()); log(f"FAILED {n} post: {e}")
        stop = (N / "STOP").exists(); pause = (N / "PAUSE").exists()
        # launch
        if not stop and not pause:
            for n, j in JOBS.items():
                s = state[n]
                if s["status"] != "PENDING" or any(state[d]["status"] != "DONE" for d in j.deps): continue
                if j.cpu_fn:
                    try:
                        j.cpu_fn(); s["status"] = "DONE"; s["done_at"] = now(); log(f"DONE {n} (cpu)")
                    except NotReady:
                        pass
                    except Exception as e:
                        s.update(status="FAILED", error="".join(traceback.format_exception_only(type(e), e)).strip()); log(f"FAILED {n}: {e}")
                        if (N / "STOP").exists(): break
                    continue
                if time.time() - last_launch < 20: continue
                free = gpu_free(); ours = {}
                for m, t in state.items():
                    if t["status"] == "RUNNING" and t.get("gpu") is not None: ours.setdefault(t["gpu"], []).append(m)
                if j.ckpt:
                    held = sum(1 for m, t in state.items() if JOBS[m].ckpt and t["status"] in ("RUNNING", "DONE")
                               and state.get(f"cleanup_{m}", {}).get("status") != "DONE")
                    if held >= MAX_CKPTS: continue
                cands = []
                for g, f in free.items():
                    heavy_here = any(JOBS[m].heavy for m in ours.get(g, []))
                    if j.heavy and heavy_here: continue
                    if len(ours.get(g, [])) >= 2: continue
                    if f >= j.gpu_need + 3000: cands.append((len(ours.get(g, [])), -f, g))
                if not cands: continue
                g = sorted(cands)[0][2]
                try:
                    cmd, env, cwd = j.build(g)
                except Exception as e:
                    s.update(status="FAILED", error="BUILD: " + str(e)); log(f"FAILED {n} build: {e}"); continue
                rc_f = LOGS / f"{n}.rc"
                if rc_f.exists(): rc_f.unlink()
                lg = open(LOGS / f"{n}.log", "a")
                lg.write(f"\n### {now()} gpu={g} cmd={' '.join(map(str, cmd))}\n"); lg.flush()
                p = subprocess.Popen(["bash", "-c", f'"$@"; echo $? > {rc_f}', "_"] + list(map(str, cmd)), env=env, cwd=cwd,
                                     stdout=lg, stderr=subprocess.STDOUT, start_new_session=True)
                s.update(status="RUNNING", pid=p.pid, gpu=g, started=now()); last_launch = time.time()
                log(f"START {n} gpu={g} pid={p.pid}")
        wjson(STATE_P, state)
        cnt = {}
        for s in state.values(): cnt[s["status"]] = cnt.get(s["status"], 0) + 1
        wjson(N / "STATUS.json", {"at": now(), "counts": cnt, "running": {n: s.get("gpu") for n, s in state.items() if s["status"] == "RUNNING"},
                                  "failed": {n: s.get("error") for n, s in state.items() if s["status"] == "FAILED"}, "stop": stop, "pause": pause})
        running = any(s["status"] == "RUNNING" for s in state.values())
        launchable = any(s["status"] == "PENDING" and all(state[d]["status"] == "DONE" for d in JOBS[n].deps) for n, s in state.items())
        waiting = state["wait_trace"]["status"] == "PENDING"
        if not running and (stop or not (launchable or waiting)):
            log("orchestrator exit: nothing runnable" + (" (STOP)" if stop else "")); break
        time.sleep(30)


if __name__ == "__main__":
    main()
