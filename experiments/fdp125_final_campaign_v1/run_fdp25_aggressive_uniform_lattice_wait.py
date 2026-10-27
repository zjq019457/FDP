#!/usr/bin/env python3
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path('/diskF/jiaqizhang/fdp125_final_campaign_v1')
REPORT = ROOT / 'reports/FDP25_AGGRESSIVE_UNIFORM_RESOURCE_LATTICE.csv'
SCRATCH = Path('/diskE/jiaqizhang/fdp125_final_campaign_v1_scratch/fdp25_aggressive_uniform_lattice')
LOGS = ROOT / 'reports/fdp25_aggressive_uniform_lattice_logs'
REPO = Path('/home/bai-project/jiaqizhang/EigenAttn')
PY = '/home/bai-project/anaconda3/envs/eigen/bin/python'
MODEL = Path('/diskD/jiaqizhang/Llama-3')
WIKI = Path('/home/bai-project/jiaqizhang/wikitext2')
BUILDER = REPO / 'EigenAttn/tools/build_eigen_basis_llama_error_budget_offline.py'
MATERIALIZER = REPO / 'EigenAttn/main_eigen_attn.py'
LOCK = ROOT / 'materialize.lock'
TARGET = 159088640
FULL = 268435456
SEED = 2
MIN_E_FREE = 32 * 1024**3
MIN_GPU_FREE_MIB = 40 * 1024
CANDIDATES = [0.0865, 0.0870, 0.0875, 0.0880, 0.0885, 0.0890]
FIELDS = ['tier','target_B_live','base_e','state','status','measured_B_live','resource_gap_pct','quality_run','selection_eligible','notes','gpu','updated_at']

def now(): return dt.datetime.now().astimezone().isoformat()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
def e_free(): return shutil.disk_usage('/diskE').free
def gpu():
    try:
        out=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True).stdout
        a=[(int(x.split(',')[1]),int(x.split(',')[0])) for x in out.splitlines() if len(x.split(','))>=2 and int(x.split(',')[0]) in (2,3)]
        a=[x for x in a if x[0]>=MIN_GPU_FREE_MIB]
        return max(a)[1] if a else None
    except Exception: return None
def stable_gpu():
    g=gpu()
    if g is None: return None
    time.sleep(10)
    return g if gpu()==g else None
def _ancestor_pids():
    pids=set()
    pid=os.getpid()
    while pid and pid > 1:
        pids.add(pid)
        try:
            fields=Path(f'/proc/{pid}/stat').read_text().split()
            pid=int(fields[3])
        except Exception:
            break
    return pids
def campaign_busy():
    own=_ancestor_pids()
    out=subprocess.run(['ps','-eo','pid=,args='],text=True,capture_output=True).stdout
    for line in out.splitlines():
        fields=line.strip().split(None,1)
        if not fields: continue
        try: pid=int(fields[0])
        except Exception: continue
        if pid in own: continue
        low=fields[1].lower() if len(fields)>1 else ''
        if 'run_fdp25_aggressive_uniform_lattice_wait.py' in low or 'official_uniform_llama_gapfill' in low:
            return True
    return False
def get_rows():
    if not REPORT.exists(): return []
    with REPORT.open(newline='') as f: return list(csv.DictReader(f))
def update(e, state, status, b='', gap='', notes='', g=''):
    old=[r for r in get_rows() if abs(float(r.get('base_e') or -1)-e)>1e-9]
    row={'tier':'aggressive','target_B_live':str(TARGET),'base_e':f'{e:.4f}','state':state,'status':status,'measured_B_live':str(b),'resource_gap_pct':str(gap),'quality_run':'NO','selection_eligible':'PENDING','notes':notes,'gpu':str(g),'updated_at':now()}
    old.append(row); old.sort(key=lambda r: float(r.get('base_e') or 0))
    tmp=REPORT.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows({k:r.get(k,'') for k in FIELDS} for r in old)
    os.replace(tmp,REPORT)
def run_one(e, g):
    tag=f'eb_{e:.4f}'.replace('.','p'); work=SCRATCH/tag; basis=work/'basis'; model=work/'final_model'; LOGS.mkdir(parents=True,exist_ok=True); log=LOGS/(tag+'.log')
    work.mkdir(parents=True,exist_ok=True)
    eb=work/'per_layer_eb.json'; eb.write_text(json.dumps({'error_budget_by_layer':[e]*32,'meta':{'official_uniform':True,'base_e':e,'seed':SEED}})+'\n')
    env=dict(os.environ); env.update({'CUDA_VISIBLE_DEVICES':str(g),'PYTHONPATH':f'{REPO}:{REPO/"EigenAttn"}','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'})
    bcmd=[PY,'-u',str(BUILDER),'--model',str(MODEL),'--data_dir',str(WIKI),'--out_dir',str(basis),'--out_prefix',f'official_uniform_aggressive_{tag}','--per_layer_eb_json',str(eb),'--freeze_layer0_eb','--layer0_eb',str(e),'--nsamples','512','--seqlen','2048','--dtype','bf16','--attn_implementation','eager','--device','cuda:0','--seed',str(SEED)]
    with log.open('a') as f:
        if subprocess.run(bcmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT).returncode: raise RuntimeError('basis_failed')
    pts=list(basis.glob('*.pt'))
    if not pts: raise RuntimeError('basis_missing')
    mcmd=[PY,'-u',str(MATERIALIZER),'--model',str(MODEL),'--net','Llama-3-8b','--dtype','bf16','--error_budget',str(e),'--avg_dim','4096','--seqlen','2048','--nsamples','64','--seed',str(SEED),'--calib_dataset','c4','--compress_only','--attn_implementation','eager','--eigen_basis_path',str(pts[0]),'--save_dir',str(model),'--output_dir',str(work),'--cache_dir',str(REPO/'HF_cache')]
    with LOCK.open('a') as lk:
        fcntl.flock(lk.fileno(),fcntl.LOCK_EX)
        with log.open('a') as f: rc=subprocess.run(mcmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        fcntl.flock(lk.fileno(),fcntl.LOCK_UN)
    if rc: raise RuntimeError('materialize_failed')
    cfg=model/'low_rank_config.json'
    if not cfg.exists(): raise RuntimeError('rank_config_missing')
    rank=json.loads(cfg.read_text()); b=sum(sum(x) for x in rank)*4096; gap=(b-TARGET)/TARGET*100
    rec={'status':'RESOURCE_ONLY_PASS','base_e':e,'B_live':b,'target_B_live':TARGET,'resource_gap_pct':gap,'seed':SEED,'gpu':g,'rank_sha256':hashlib.sha256(json.dumps(rank,sort_keys=True).encode()).hexdigest(),'config_sha256':sha(cfg),'basis_sha256':sha(pts[0]),'updated_at':now()}
    tmp=ROOT/'reports'/f'FDP25_AGGRESSIVE_UNIFORM_RESOURCE_{tag}.json.tmp'; tmp.write_text(json.dumps(rec,indent=2)+'\n'); os.replace(tmp,tmp.with_suffix(''))
    update(e,'MEASURED','RESOURCE_ONLY_PASS',b,gap,'official path; resource-only; no PPL',g)
    shutil.rmtree(work,ignore_errors=True)
def main():
    for e in CANDIDATES:
        if any(r.get('base_e')==f'{e:.4f}' and r.get('status')=='RESOURCE_ONLY_PASS' for r in get_rows()): continue
        attempts=0
        update(e,'QUEUED','PENDING_SPACE_GATE',notes='waits for E>=32GiB and GPU2/3>=40GiB; no PPL')
        while e_free()<MIN_E_FREE or stable_gpu() is None or campaign_busy(): time.sleep(60)
        g=stable_gpu()
        if g is None: continue
        update(e,'RUNNING','RESOURCE_ONLY_RUNNING',notes='official basis+materializer; no PPL',g=g)
        try:
            run_one(e,g)
        except Exception as ex:
            attempts += 1
            shutil.rmtree(SCRATCH/f'eb_{e:.4f}'.replace('.','p'),ignore_errors=True)
            if attempts < 3 and any(x in str(ex) for x in ('basis_failed','basis_missing','materialize_failed')):
                update(e,'RETRY_WAIT','RESOURCE_RACE_RETRY',notes=f'attempt {attempts}/3: {ex}; waiting for stable GPU',g=g)
                time.sleep(60)
                continue
            update(e,'FAILED','RESOURCE_ONLY_FAILURE',notes=f'attempt {attempts}: {ex}',g=g)
            break
if __name__=='__main__': main()
