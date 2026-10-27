#!/usr/bin/env python3
import json, os, re, subprocess, time
from pathlib import Path

ROOT=Path("/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1")
PYTHON="/home/bai-project/anaconda3/envs/eigen/bin/python"
RATIOS=[0.7,0.5,0.3]
SIGNALS=["fdp_prior","fdp_native"]
WORKER_PEAK_MIB=28672
RESERVED_MARGIN_MIB=12288
NAMES={0.5:["fdp_prior","fdp_native","random_0","random_1","random_2"],
       0.7:["fdp_prior","fdp_native"],0.3:["fdp_prior","fdp_native"]}

def read(p,default=None):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception: return default
def write(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(p):
    import hashlib
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def ps():
    try:
        return subprocess.check_output(["bash","-lc","ps -eo pid,cmd"],text=True)
    except Exception: return ""
def active(fragment):
    return fragment in ps()

def safe_gpu():
    try:
        raw=subprocess.check_output(
            ["nvidia-smi","--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],text=True)
    except Exception: return None
    candidates=[]
    required_free=WORKER_PEAK_MIB+RESERVED_MARGIN_MIB
    for line in raw.splitlines():
        try: idx,free,util=[int(x.strip()) for x in line.split(",")]
        except Exception: continue
        if idx==1: continue
        if free>=required_free: candidates.append((free,idx))
    return max(candidates)[1] if candidates else None

def phi_lock(ratio):
    tag=("%.1f"%ratio).replace(".","p")
    d=ROOT/"official_native_marginal"/("ratio_"+tag)
    status=read(d/"status.json")
    lock=d/"PALU_OFFICIAL_NATIVE_PHI_LOCK.json"
    units=d/"units.csv"
    if not status or status.get("status")!="MARGINAL_PROFILE_COMPLETE" or not units.exists():
        return False
    if not lock.exists():
        action=read(ROOT/"PALU_OFFICIAL_NATIVE_ACTION_SPACE.json")
        qmap=read(ROOT/"PALU_OFFICIAL_Q_GROUP_MAPPING.json")
        write(lock,{"status":"OFFICIAL_NATIVE_PHI_LOCKED","ratio":ratio,
                    "definition":"one legal official 32-rank decrement per K/V group; delta_J divided by relative native resource reduction",
                    "units_csv":str(units),"units_sha256":sha(units),
                    "profile_status_sha256":sha(d/"status.json"),
                    "action_space_sha256":action["action_space_sha256"],
                    "q_mapping_sha256":qmap["mapping_sha256"],
                    "profiling_manifest":status.get("manifest_sha256"),
                    "quality_used_to_select":False,
                    "old_phi_status":"VALID_FOR_OLD_ANCHOR_SCHEMA_ONLY"})
    return True

def imported_fisher(ratio):
    tag=("%.1f"%ratio).replace(".","p")
    out=ROOT/"official_formal_quality"/("ratio_"+tag)/"fisher"
    if (out/"status.json").exists(): return
    p=ROOT/"results"/("fisher_"+tag+"_ratio_"+("%.1f"%ratio)+"_fisher_uniform.json")
    d=read(p)
    if not d or d.get("status")!="OFFICIAL_PATH_QUALITY_COMPLETE": return
    rank=int(d["total_rank"]); resource={"total_native_rank":rank,
             "factor_bytes_fp16":rank*(4096+512)*2,
             "prefill_cache_bytes":int(d["prefill_cache_bytes"]),
             "decode_cache_bytes":int(d["decode_cache_bytes"])}
    write(out/"result.json",{"status":"OFFICIAL_FISHER_FORMAL_IMPORTED","ratio":ratio,
        "method":"fisher","official_baseline_method":"PaLu Automatic (Fisher)",
        "wiki":d["wiki"],"c4":d["c4"],"J":0.5*(d["wiki"]["mean_token_nll"]+d["c4"]["mean_token_nll"]),
        "resource":resource,"resource_match":"REFERENCE_BASELINE",
        "source":str(p),"source_sha256":sha(p),"protocol":d.get("protocol"),
        "formal_quality_used_to_select":False})
    write(out/"status.json",{"status":"OFFICIAL_FISHER_FORMAL_IMPORTED","ratio":ratio,"source":str(p)})

def status(ratio,signal):
    tag=("%.1f"%ratio).replace(".","p")
    return read(ROOT/"official_exchange_calibration"/("ratio_"+tag)/signal/"status.json")

def formal_status(ratio,method):
    tag=("%.1f"%ratio).replace(".","p")
    return read(ROOT/"official_formal_quality"/("ratio_"+tag)/method/"status.json")

def launch(script,args,gpu,log):
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu); env["PYTHONUNBUFFERED"]="1"
    stream=open(log,"ab",buffering=0)
    p=subprocess.Popen([PYTHON,str(ROOT/"code"/script)]+args,
                       stdout=stream,stderr=subprocess.STDOUT,
                       env=env,start_new_session=True)
    return p.pid

def update_state(tiers,reason=None):
    p=ROOT/"OFFICIAL_TIER_DISPATCH_STATE.json"
    write(p,{"status":"ACTIVE","foreign_jobs_interrupted":0,"gpu1_used":False,
             "gpu_policy":{"mode":"THROUGHPUT_FIRST_MEMORY_SAFE","worker_peak_mib":WORKER_PEAK_MIB,"reserved_margin_mib":RESERVED_MARGIN_MIB,"utilization_hard_block":False},
             "tiers":tiers,"last_reason":reason,"updated_epoch":time.time()})
    state=read(ROOT/"PALU_PIPELINE_STATE.json",{}) or {}
    state["official_tier_dispatcher"]="ACTIVE"
    state["official_tier_status"]=tiers
    state["foreign_jobs_interrupted"]=0
    state["gpu1_used"]=False
    state["official_native_fdp"]="TIER_READY_AUTO_DISPATCH"
    write(ROOT/"PALU_PIPELINE_STATE.json",state)

def main():
    while True:
        tiers={}; reason="waiting for tier dependency or safe GPU"
        launched=False
        for ratio in RATIOS:
            key="%.1f"%ratio; tiers[key]={"phi":"PENDING","calibration":{},"formal":{}}
            if phi_lock(ratio):
                tiers[key]["phi"]="FROZEN"
                imported_fisher(ratio); tiers[key]["formal"]["fisher"]="IMPORTED_COMPLETE"
                for signal in NAMES[ratio]:
                    s=status(ratio,signal)
                    tiers[key]["calibration"][signal]=s.get("status","PENDING") if s else "PENDING"
                if all(tiers[key]["calibration"].get(s)=="EXCHANGE_CALIBRATION_COMPLETE" for s in NAMES[ratio]):
                    for method in NAMES[ratio]:
                        fs=formal_status(ratio,method)
                        tiers[key]["formal"][method]=fs.get("status","PENDING") if fs else "PENDING"
                else:
                    for method in NAMES[ratio]: tiers[key]["formal"][method]="WAIT_CALIBRATION"
            else:
                for signal in NAMES[ratio]: tiers[key]["calibration"][signal]="WAIT_PHI"
                for method in NAMES[ratio]: tiers[key]["formal"][method]="WAIT_PHI"
        for ratio in RATIOS:
            key="%.1f"%ratio
            if tiers[key]["phi"]!="FROZEN": continue
            for signal in NAMES[ratio]:
                if status(ratio,signal): continue
                frag="run_official_exchange_calibration.py --ratio %.1f --signal %s"%(ratio,signal)
                if active(frag): launched=True; break
                gpu=safe_gpu()
                if gpu is not None:
                    tag=("%.1f"%ratio).replace(".","p")
                    log=ROOT/"logs"/("official_exchange_calibration_"+tag+"_"+signal+"_gpu"+str(gpu)+".log")
                    pid=launch("run_official_exchange_calibration.py",
                               ["--ratio","%.1f"%ratio,"--signal",signal,"--device","cuda"],gpu,log)
                    tiers[key]["calibration"][signal]="RUNNING_PID_"+str(pid)
                    launched=True
                break
            if launched: break
            if tiers[key]["calibration"] and all(tiers[key]["calibration"].get(s)=="EXCHANGE_CALIBRATION_COMPLETE" for s in NAMES[ratio]):
                for method in NAMES[ratio]:
                    if formal_status(ratio,method): continue
                    frag="run_official_formal_quality.py --ratio %.1f --method %s"%(ratio,method)
                    if active(frag): launched=True; break
                    gpu=safe_gpu()
                    if gpu is not None:
                        tag=("%.1f"%ratio).replace(".","p")
                        log=ROOT/"logs"/("official_formal_quality_"+tag+"_"+method+"_gpu"+str(gpu)+".log")
                        pid=launch("run_official_formal_quality.py",
                                   ["--ratio","%.1f"%ratio,"--method",method,"--device","cuda"],gpu,log)
                        tiers[key]["formal"][method]="RUNNING_PID_"+str(pid)
                        launched=True
                    break
                if launched: break
        update_state(tiers,reason)
        time.sleep(60)

if __name__=="__main__": main()
