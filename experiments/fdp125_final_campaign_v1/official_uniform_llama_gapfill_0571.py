#!/usr/bin/env python3
import argparse, datetime as dt, hashlib, json, os, shutil, subprocess, fcntl
from pathlib import Path

ROOT=Path('/diskF/jiaqizhang/fdp125_final_campaign_v1')
SCR=Path('/diskE/jiaqizhang/fdp125_final_campaign_v1_scratch/official_uniform_llama_gapfill_0571')
REPO=Path('/home/bai-project/jiaqizhang/EigenAttn'); PY='/home/bai-project/anaconda3/envs/eigen/bin/python'
MODEL=Path('/diskD/jiaqizhang/Llama-3'); WIKI=Path('/home/bai-project/jiaqizhang/wikitext2')
C4=Path('/diskD/jiaqizhang/llama3_8b_eb045_eval_minimal-20260502T170035Z-3-001/llama3_8b_eb045_eval_minimal/data/c4_ppl_selected_texts.jsonl')
BUILDER=REPO/'EigenAttn/tools/build_eigen_basis_llama_error_budget_offline.py'; MAIN=REPO/'EigenAttn/main_eigen_attn.py'; RUNNER=REPO/'rebuttal_legacy_reproduction/runners/ppl_instrumented.py'
LOCK=REPO/'rebuttal_legacy_reproduction/LEGACY_EVALUATION_PROTOCOL_LOCK.json'
E={'light':0.043,'medium':0.0571,'aggressive':0.085}; TARGET={'light':218923008,'medium':199753728,'aggressive':159088640}
def now(): return dt.datetime.now().astimezone().isoformat()
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix('.tmp'); t.write_text(json.dumps(x,indent=2)+'\n'); os.replace(t,p)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--tier',choices=E,required=True); ap.add_argument('--gpu',type=int,required=True); a=ap.parse_args()
 out=ROOT/'results/llama'/a.tier/'official_uniform_gapfill_0571'; work=SCR/a.tier
 if (out/'DONE.json').exists(): return
 if work.exists() and list((work/'basis').glob('*.pt')):
  # Preserve a completed official basis after a downstream engineering failure.
  pass
 elif work.exists(): shutil.rmtree(work)
 work.mkdir(parents=True,exist_ok=True); out.mkdir(parents=True,exist_ok=True); write(out/'STATUS.json',{'status':'RUNNING','tier':a.tier,'gpu':a.gpu,'path':'official_eigenattention_uniform','at':now()})
 env=dict(os.environ); env.update({'CUDA_VISIBLE_DEVICES':str(a.gpu),'PYTHONPATH':f'{REPO}:{REPO/"EigenAttn"}','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','C4_DATA_DIR':str(C4.parent)})
 e=E[a.tier]; eb=work/'per_layer_eb.json'; write(eb,{'error_budget_by_layer':[e]*32,'meta':{'official_uniform':True,'base_e':e}})
 basis=work/'basis'; model=work/'final_model'; log=work/'run.log'
 cmds=[[PY,'-u',str(BUILDER),'--model',str(MODEL),'--data_dir',str(WIKI),'--out_dir',str(basis),'--out_prefix',f'official_uniform_{a.tier}','--per_layer_eb_json',str(eb),'--freeze_layer0_eb','--layer0_eb',str(e),'--nsamples','512','--seqlen','2048','--dtype','bf16','--attn_implementation','eager','--device','cuda:0','--seed','2'],[PY,'-u',str(MAIN),'--model',str(MODEL),'--net','Llama-3-8b','--dtype','bf16','--error_budget',str(e),'--avg_dim','4096','--seqlen','2048','--nsamples','64','--seed','2','--calib_dataset','c4','--compress_only','--attn_implementation','eager','--eigen_basis_path',str(next(basis.glob('*.pt')) if basis.exists() and list(basis.glob('*.pt')) else basis/'official_uniform_placeholder.pt'),'--save_dir',str(model),'--output_dir',str(work),'--cache_dir',str(REPO/'HF_cache')]]
 if True:
  basis.mkdir(exist_ok=True)
  if not list(basis.glob('*.pt')):
   with log.open('a') as f: r=subprocess.run(cmds[0],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
   if r.returncode: raise RuntimeError(f'basis rc={r.returncode}')
  pts=list(basis.glob('*.pt'))
  if not pts: raise RuntimeError('basis missing')
  cmds[1][cmds[1].index('--eigen_basis_path')+1]=str(pts[0])
  with open(ROOT/'materialize.lock','w') as lk:
   fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
   with log.open('a') as f: r=subprocess.run(cmds[1],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
   fcntl.flock(lk.fileno(), fcntl.LOCK_UN)
  if r.returncode: raise RuntimeError(f'materialize rc={r.returncode}')
  rank=json.loads((model/'low_rank_config.json').read_text()); bl=sum(sum(x) for x in rank)*4096
  if bl<=TARGET[a.tier]: raise RuntimeError(f'official resource not above FDP: B_live={bl}')
  quality={}
  for ds,args in [('wiki',['--dataset','wikitext2','--wikitext2_dir',str(WIKI)]),('c4',['--dataset','c4','--c4_texts',str(C4)])]:
   od=out/ds; od.mkdir(parents=True,exist_ok=True); cmd=[PY,'-u',str(RUNNER),*args,'--model',str(model),'--tok',str(MODEL),'--out_dir',str(od),'--dtype','bf16','--method',f'OFFICIAL_UNIFORM_LLAMA_{a.tier}','--evaluation_protocol_hash',sha(LOCK)]
   with log.open('a') as f: r=subprocess.run(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
   if r.returncode: raise RuntimeError(f'{ds} rc={r.returncode}')
   quality[ds]=json.loads((od/'ppl_aggregate.json').read_text())
  prov={'status':'PASS','method':'official_eigenattention_uniform','tier':a.tier,'base_e':e,'B_live':bl,'quality':quality,'gpu':a.gpu,'seed':2}
  write(out/'provenance.json',prov)
  write(out/'DONE.json',{'status':'PASS','B_live':bl,'quality':quality,'completed_at':now()})
  shutil.rmtree(work,ignore_errors=True)
if __name__=='__main__': main()
