#!/usr/bin/env python3
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT=Path('/diskF/jiaqizhang/fdp125_final_campaign_v1')
OUT=ROOT/'results/llama/aggressive/official_uniform_resource_match_0870'
WORK=Path('/diskE/jiaqizhang/fdp125_final_campaign_v1_scratch/official_uniform_resource_match_0870')
REPO=Path('/home/bai-project/jiaqizhang/EigenAttn')
PY='/home/bai-project/anaconda3/envs/eigen/bin/python'
MODEL=Path('/diskD/jiaqizhang/Llama-3')
WIKI=Path('/home/bai-project/jiaqizhang/wikitext2')
C4=Path('/diskD/jiaqizhang/llama3_8b_eb045_eval_minimal-20260502T170035Z-3-001/llama3_8b_eb045_eval_minimal/data/c4_ppl_selected_texts.jsonl')
BUILDER=REPO/'EigenAttn/tools/build_eigen_basis_llama_error_budget_offline.py'
MAIN=REPO/'EigenAttn/main_eigen_attn.py'
RUNNER=REPO/'rebuttal_legacy_reproduction/runners/ppl_instrumented.py'
LOCK=ROOT/'materialize.lock'
PROTO=REPO/'rebuttal_legacy_reproduction/LEGACY_EVALUATION_PROTOCOL_LOCK.json'
EB=0.0870
GPU=1
TARGET=159088640
EXPECTED_B=160038912

def now(): return dt.datetime.now().astimezone().isoformat()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n')
    os.replace(tmp,path)
def gpu_free(g):
    out=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True).stdout
    for line in out.splitlines():
        a=[x.strip() for x in line.split(',')]
        if len(a)>=2 and int(a[0])==g: return int(a[1])
    return 0
def ensure_gate():
    if shutil.disk_usage('/diskE').free < 32*1024**3: raise RuntimeError('E_space_gate')
    if gpu_free(GPU) < 40*1024: raise RuntimeError('GPU_space_gate')
    time.sleep(10)
    if gpu_free(GPU) < 40*1024: raise RuntimeError('GPU_space_gate_unstable')
def main():
    if (OUT/'DONE.json').exists(): return
    OUT.mkdir(parents=True,exist_ok=True)
    log=OUT/'run.log'
    write(OUT/'STATUS.json',{'status':'RUNNING','method':'official_eigenattention_uniform','base_e':EB,'target_B_live':TARGET,'gpu':GPU,'seed':2,'at':now()})
    try:
        ensure_gate()
        if WORK.exists(): shutil.rmtree(WORK)
        WORK.mkdir(parents=True,exist_ok=True)
        ebjson=WORK/'per_layer_eb.json'
        write(ebjson,{'error_budget_by_layer':[EB]*32,'meta':{'official_uniform':True,'base_e':EB,'seed':2,'source':'FDP25 resource lock 0870'}})
        basis=WORK/'basis'; model=WORK/'final_model'; basis.mkdir()
        env=dict(os.environ); env.update({'CUDA_VISIBLE_DEVICES':str(GPU),'PYTHONPATH':f'{REPO}:{REPO/"EigenAttn"}','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','C4_DATA_DIR':str(C4.parent)})
        bcmd=[PY,'-u',str(BUILDER),'--model',str(MODEL),'--data_dir',str(WIKI),'--out_dir',str(basis),'--out_prefix','official_uniform_aggressive_eb_0p0870_quality','--per_layer_eb_json',str(ebjson),'--freeze_layer0_eb','--layer0_eb',str(EB),'--nsamples','512','--seqlen','2048','--dtype','bf16','--attn_implementation','eager','--device','cuda:0','--seed','2']
        with log.open('a') as f: r=subprocess.run(bcmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode: raise RuntimeError(f'basis_rc={r.returncode}')
        pts=list(basis.glob('*.pt'))
        if not pts: raise RuntimeError('basis_missing')
        mcmd=[PY,'-u',str(MAIN),'--model',str(MODEL),'--net','Llama-3-8b','--dtype','bf16','--error_budget',str(EB),'--avg_dim','4096','--seqlen','2048','--nsamples','64','--seed','2','--calib_dataset','c4','--compress_only','--attn_implementation','eager','--eigen_basis_path',str(pts[0]),'--save_dir',str(model),'--output_dir',str(WORK),'--cache_dir',str(REPO/'HF_cache')]
        with LOCK.open('a') as lk:
            fcntl.flock(lk.fileno(),fcntl.LOCK_EX)
            with log.open('a') as f: r=subprocess.run(mcmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
            fcntl.flock(lk.fileno(),fcntl.LOCK_UN)
        if r.returncode: raise RuntimeError(f'materialize_rc={r.returncode}')
        rank=json.loads((model/'low_rank_config.json').read_text())
        bl=sum(sum(x) for x in rank)*4096
        write(OUT/'resource_check.json',{'B_live':bl,'target_B_live':TARGET,'expected_B_live':EXPECTED_B,'gap_pct':(bl-TARGET)/TARGET*100,'rank_sha256':hashlib.sha256(json.dumps(rank,sort_keys=True).encode()).hexdigest(),'config_sha256':sha(model/'low_rank_config.json'),'basis_sha256':sha(pts[0]),'at':now()})
        if bl != EXPECTED_B: raise RuntimeError(f'resource_mismatch_B_live={bl}')
        quality={}
        for ds,args in [('wiki',['--dataset','wikitext2','--wikitext2_dir',str(WIKI)]),('c4',['--dataset','c4','--c4_texts',str(C4)])]:
            od=OUT/ds; od.mkdir(parents=True,exist_ok=True)
            cmd=[PY,'-u',str(RUNNER),*args,'--model',str(model),'--tok',str(MODEL),'--out_dir',str(od),'--dtype','bf16','--method','OFFICIAL_UNIFORM_LLAMA_AGGRESSIVE_RESOURCE_MATCH_0870','--evaluation_protocol_hash',sha(PROTO)]
            with log.open('a') as f: r=subprocess.run(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
            if r.returncode: raise RuntimeError(f'{ds}_rc={r.returncode}')
            quality[ds]=json.loads((od/'ppl_aggregate.json').read_text())
        write(OUT/'provenance.json',{'status':'PASS','method':'official_eigenattention_uniform','base_e':EB,'B_live':bl,'target_B_live':TARGET,'resource_gap_pct':(bl-TARGET)/TARGET*100,'quality':quality,'gpu':GPU,'seed':2,'evaluation_protocol_hash':sha(PROTO),'completed_at':now()})
        write(OUT/'DONE.json',{'status':'PASS','B_live':bl,'quality':quality,'completed_at':now()})
    except Exception as ex:
        write(OUT/'STATUS.json',{'status':'FAILED','error':str(ex),'method':'official_eigenattention_uniform','base_e':EB,'gpu':GPU,'at':now()})
    finally:
        shutil.rmtree(WORK,ignore_errors=True)
if __name__=='__main__': main()
