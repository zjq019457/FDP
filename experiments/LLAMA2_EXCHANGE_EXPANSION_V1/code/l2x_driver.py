#!/usr/bin/env python3
"""Job driver for LLAMA2_EXCHANGE_EXPANSION_V1 (PREREGISTRATION.json). Launches a job when its inputs exist, on the GPU with the
most free memory (>= need, at most 2 of our jobs per GPU); never touches processes it did not start; a failed job is reported
in STATUS.json and blocks only its dependents. Restart-safe: a job whose output exists is DONE."""
import json, os, pathlib, subprocess, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import l2x_lib as X

R = X.ROOT; C = R / "code"; PY = "/home/bai-project/anaconda3/envs/eigen/bin/python"
ENV = dict(os.environ, PYTHONPATH="/home/bai-project/jiaqizhang/EigenAttn/EigenAttn:/home/bai-project/jiaqizhang/EigenAttn", HF_HUB_OFFLINE="1",
           TRANSFORMERS_OFFLINE="1", PYTHONHASHSEED="2", TOKENIZERS_PARALLELISM="false")
SEEDS = {"e068": [20261101 + i for i in range(20)], "e060": [20261201 + i for i in range(10)], "e090": [20261301 + i for i in range(10)]}
S = lambda n: R / "schedules" / n / "SCHEDULE.json"
Q = lambda n: R / "quality" / n / "RESULT.json"
JOBS = []


def job(name, kind, cmd, deps, out, need):
    JOBS.append({"name": name, "kind": kind, "cmd": cmd, "deps": deps, "out": out, "need": need})


def K(op):
    return 9 if op == "e068" else X.rjson(S(f"func_{op}_std"))["K"]


# frozen historical e068 functional schedule as an evaluable schedule file
if not S("func_e068_std").exists():
    fz = X.rjson(X.FUNC068); p = [[int(a), int(b)] for a, b in fz["final"]["rank_schedule"]]
    X.wjson(S("func_e068_std"), {"method": "functional measured-NLL exchange (frozen historical e068 run)", "op": "e068", "rank_pairs": p, "B_live": X.B_of(p),
                                 "K": fz["accepted_exchange_count"], "rank_sha": X.sha_json(p), "source": str(X.FUNC068), "source_sha256": X.sha_file(X.FUNC068)})

for op, w in (("e060", "std"), ("e090", "std"), ("e068", "dd")):
    n = f"func_{op}_{w}"
    job(n, "exchange", [PY, "-u", str(C / "l2x_exchange.py"), op, w, "{gpu}", n], [], S(n), 22000)
for op in ("e060", "e068", "e090"):
    prof = [X.PROFILE068] if op == "e068" else [R / "profiles" / op / "run" / "LLAMA2_CLEAN_REDESIGN_D1_D2_D5_PROFILE_V1.json"]
    kdep = [] if op == "e068" else [S(f"func_{op}_std")]
    job(f"plan_fdp_{op}", "cpu", lambda op=op: [PY, str(C / "l2x_plan.py"), "fdp", op, str(K(op)), f"fdp_{op}"], prof + kdep, S(f"fdp_{op}"), 0)
    for s in SEEDS[op]:
        job(f"plan_rand_{op}_{s}", "cpu", lambda op=op, s=s: [PY, str(C / "l2x_plan.py"), "rand", op, str(K(op)), str(s), f"rand_{op}_{s}"], kdep, S(f"rand_{op}_{s}"), 0)
EVAL = [f"uniform_{op}" for op in ("e060", "e090")] + ["func_e068_std", "func_e060_std", "func_e090_std", "func_e068_dd"] + \
       [f"fdp_{op}" for op in ("e068", "e060", "e090")] + [f"rand_{op}_{s}" for op in ("e068", "e060", "e090") for s in SEEDS[op]]
for n in EVAL:
    job(f"q_{n}", "gpu", [PY, "-u", str(C / "l2x_quality.py"), n, str(S(n)), "{gpu}"], [S(n)], Q(n), 30000)


def busy(n):
    lab = n[2:] if n.startswith("q_") else n
    pat = f"run_profiles.sh {lab[8:]}$" if lab.startswith("profile_") else f"l2x_(quality|exchange|plan).py .*{lab}( |$)"
    return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0


RESOURCE = ("free-VRAM safety gate", "OutOfMemoryError", "CUDA out of memory")


def archive(j, attempt):
    import shutil
    n = j["name"]; lab = n[2:] if n.startswith("q_") else n
    for p in ([R / "quality" / lab] if n.startswith("q_") else [R / "schedules" / lab] if j["kind"] == "exchange" else
              [R / "profiles" / lab[8:] / "run"] if n.startswith("profile_") else []):
        if p.exists(): shutil.move(str(p), str(p) + f"_attempt{attempt}_resource")
    lg = R / "logs" / f"{n}.log"
    if lg.exists(): shutil.move(str(lg), str(lg) + f".attempt{attempt}_resource")


def free():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines())}


def main():
    run, failed, last, tries, retry_at = {}, {}, {}, {}, {}
    while True:
        for n, (p, g, j) in list(run.items()):
            rc = p.poll()
            if rc is None: continue
            del run[n]
            if rc != 0 or not j["out"].exists():
                txt = (R / "logs" / f"{n}.log").read_text(errors="ignore")[-20000:] if (R / "logs" / f"{n}.log").exists() else ""
                tries[n] = tries.get(n, 0) + 1
                if any(k in txt for k in RESOURCE) and tries[n] <= 3:
                    archive(j, tries[n]); retry_at[n] = time.time() + 600
                    with open(R / "logs" / "ENGINEERING_EVENTS.log", "a") as f: f.write(f"{X.now()} {n}: resource failure, retry {tries[n]} in 10 min\n")
                else:
                    failed[n] = f"rc={rc}"
        done = {j["name"] for j in JOBS if j["out"].exists()}
        for j in JOBS:
            n = j["name"]
            if n in done or n in run or n in failed or not all(pathlib.Path(d).exists() for d in j["deps"]): continue
            if time.time() < retry_at.get(n, 0): continue
            if j["kind"] != "cpu" and busy(n): continue  # already running (started by an earlier driver instance)
            if j["kind"] == "cpu":
                rc = subprocess.run(j["cmd"](), env=ENV, capture_output=True, text=True)
                (R / "logs" / f"{n}.log").write_text(rc.stdout + rc.stderr)
                if rc.returncode != 0: failed[n] = f"rc={rc.returncode}"
                continue
            fr = free(); mine = {}
            for _, g, _ in run.values(): mine[g] = mine.get(g, 0) + 1
            cand = [(-f, g) for g, f in fr.items() if f >= j["need"] + 6000 and mine.get(g, 0) < 2 and time.time() - last.get(g, 0) > 60]
            if not cand: continue
            g = sorted(cand)[0][1]
            cmd = [str(g) if x == "{gpu}" else x for x in j["cmd"]]
            lg = open(R / "logs" / f"{n}.log", "a")
            p = subprocess.Popen(cmd, env=dict(ENV, CUDA_VISIBLE_DEVICES=str(g)), stdout=lg, stderr=subprocess.STDOUT, start_new_session=True)
            run[n] = (p, g, j); last[g] = time.time()
        X.wjson(R / "STATUS.json", {"at": X.now(), "done": len(done), "total": len(JOBS), "running": {n: g for n, (_, g, _) in run.items()}, "failed": failed,
                                     "waiting": [j["name"] for j in JOBS if j["name"] not in done and j["name"] not in run and j["name"] not in failed]})
        if not run and all(j["name"] in done or j["name"] in failed for j in JOBS): break
        time.sleep(30)


if __name__ == "__main__":
    main()
