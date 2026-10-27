#!/usr/bin/env python3
"""Isolated official-code uniform-allocation LRD diagnostics (user-requested, not Table1 exact)."""
import os,sys,json,hashlib,gzip,datetime,traceback,gc,fcntl,types
from pathlib import Path
BASE=Path('/diskE/jiaqizhang/PALU_OFFICIAL_TABLE1_REPRO_V1')
REPO=Path('/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/Palu')
DATA=BASE/'inputs/wikitext2_raw_v1'
C4=Path('/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/data/c4-validation.00000-of-00008.json.gz')
SHA={'wiki_test':'3ee89cd6a2ab912afd5d01e98867b46a67d9ec7a9eca0910e5e4c3cdd4cc1925','wiki_train':'49280f3dd993cbdfd322186194133d5d1935f81e1c91064c239f2a70ebb02701','c4':'1f25b6af12da84115301d4ee93ea5246c8fea5bb4a2008472794d95b917cc97f'}
for p,k in [(DATA/'test-00000-of-00001.parquet','wiki_test'),(DATA/'train-00000-of-00001.parquet','wiki_train'),(C4,'c4')]:
    if hashlib.sha256(p.read_bytes()).hexdigest()!=SHA[k]: raise RuntimeError('input SHA mismatch '+k)
os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
sys.path.insert(0,str(REPO))
import evaluate
_metric_load=evaluate.load
class Guard:
    def compute(self,*a,**k): raise RuntimeError('zero-shot metric invoked unexpectedly')
evaluate.load=lambda name,*a,**k: Guard() if name=='exact_match' else _metric_load(name,*a,**k)
fht=types.ModuleType('fast_hadamard_transform')
def forbidden(*a,**k): raise RuntimeError('Hadamard invoked in unquantized PPL')
fht.hadamard_transform=forbidden
sys.modules['fast_hadamard_transform']=fht
import datasets,torch,numpy as np
_real_load=datasets.load_dataset
class Rows:
    def __init__(self,text): self.text=text
    def __getitem__(self,key):
        if isinstance(key,slice): return {'text':self.text[key]}
        if key=='text': return self.text
        return {'text':self.text[key]}
def local_load(path,*a,**kw):
    if path=='wikitext' and a and a[0]=='wikitext-2-raw-v1':
        split=kw.get('split')
        file=DATA/('train-00000-of-00001.parquet' if split=='train' else 'test-00000-of-00001.parquet')
        if split not in ('train','test'): raise RuntimeError('unexpected Wiki split')
        return _real_load('parquet',data_files={split:str(file)},split=split)
    if path=='allenai/c4':
        if kw.get('revision')!='607bd4c8450a42878aa9ddc051a65a055450ef87': raise RuntimeError('wrong C4 revision')
        import json as jj
        txt=[]
        with gzip.open(C4,'rt',encoding='utf-8') as f:
            for _,line in zip(range(1100),f): txt.append(jj.loads(line)['text'])
        return Rows(txt)
    return _real_load(path,*a,**kw)
datasets.load_dataset=local_load
_real_save=torch.save
torch.save=lambda obj,*a,**k: _real_save(types.SimpleNamespace(input_ids=obj.input_ids) if obj.__class__.__name__=='TokenizerWrapper' else obj,*a,**k)
from palu.model.modules.svd_linear import HeadwiseLowRankModule
import compress as cp,run_ppl_eval
ROOT=BASE/'uniform_diagnostics'; STATE=BASE/'state/UNIFORM_SWEEP_STATE.json'
LOCKDIR=BASE/'locks/UNIFORM_SWEEP_ACTIVE'; LOCKDIR.mkdir(parents=True,exist_ok=True)
lock=open(BASE/'locks/UNIFORM_SWEEP.lock','w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def savej(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)
state=json.loads(STATE.read_text()) if STATE.exists() else {'cells':{}}
state.update(status='RUNNING',worker_pid=os.getpid(),gpu=0,allocator='uniform',fisher_enabled=False,foreign_jobs_interrupted=0)
active={'run_id':'TABLE1_USER_UNIFORM_SWEEP_20260929','pid':os.getpid(),'gpu':0,'started_utc':utc(),'method':'uniform','foreign_jobs_interrupted':0}
savej(LOCKDIR/'ACTIVE.json',active); savej(STATE,state)
specs=[('J_LRD_GS32_UNIFORM',32,['wikitext2','c4']),('G_LRD_GS4_UNIFORM',4,['wikitext2','c4']),('M_LRD_GS1_UNIFORM',1,['wikitext2','c4']),('G_LRD_GS16_UNIFORM',16,['wikitext2']),('G_LRD_GS8_UNIFORM',8,['wikitext2']),('G_LRD_GS2_UNIFORM',2,['wikitext2'])]
alias=str(BASE/'model_reference/Llama-2-7b-hf'); hold={}
_load=cp.load_model_and_tokenizer
def capload(*a,**k):
    m,t=_load(*a,**k); hold.update(model=m,tokenizer=t); return m,t
cp.load_model_and_tokenizer=capload
_rank=cp.rank_search
def caprank(m,t,a):
    out=_rank(m,t,a); hold.update(selection=out[0],rank_sum=int(out[1]),total_rank=int(out[2])); return out
cp.rank_search=caprank
def allocation_sha(sel):
    raw=json.dumps({k:[int(x) for x in v] for k,v in sorted(sel.items())},separators=(',',':')).encode()
    return hashlib.sha256(raw).hexdigest()
def resource(m,sel,rs,tr,gs):
    mods=[(n,x) for n,x in m.named_modules() if isinstance(x,HeadwiseLowRankModule)]
    fb=0; h=hashlib.sha256(); hist={}
    for _,r in sorted(sel.items()):
        for v in r: hist[str(int(v))]=hist.get(str(int(v)),0)+1
    for n,x in mods:
        for pn,p in sorted(x.named_parameters()):
            fb+=p.numel()*p.element_size(); arr=p.detach().cpu().contiguous().numpy()
            h.update((n+'.'+pn+'|'+str(tuple(arr.shape))+'|'+str(arr.dtype)+'\n').encode()); h.update(memoryview(arr).cast('B'))
    return {'rank_sum':int(rs),'total_rank':int(tr),'compression_ratio':1-float(rs)/tr,'rank_quantum':32,'factor_bytes':int(fb),'cache_bytes_per_token_fp16':int(rs*2),'prefill_cache_bytes_at_2048_fp16':int(rs*4096),'decode_cache_bytes_at_2048_fp16':int(rs*4096),'low_rank_module_count':len(mods),'allocation_sha256':allocation_sha(sel),'realization_sha256':h.hexdigest(),'rank_histogram':hist,'resource_pass':int(rs)==131072 and int(tr)==262144 and fb>0}
for cell,gs,dsets in specs:
    if state.get('cells',{}).get(cell,{}).get('status')=='DONE':
        print('SKIP_ALREADY_DONE',cell,flush=True); continue
    outdir=ROOT/cell; outdir.mkdir(parents=True,exist_ok=True); start=utc()
    state['cells'][cell]={'status':'RUNNING','gs':gs,'allocator':'uniform','fisher_enabled':False,'datasets':dsets,'started_utc':start,'old_results_valid':True}
    savej(STATE,state); hold.clear(); print('CELL_START',cell,'gs',gs,flush=True)
    try:
        args=types.SimpleNamespace(model_id=alias,seed=42,dump_huggingface_model=False,use_cache=True,device='cuda',n_fisher_calib_samples=32,n_whiten_calib_samples=256,calib_dataset='wikitext2',calib_seqlen=1024,head_group_size=gs,param_ratio_target=.5,verbose=False,search_method='uniform',decompose_method='whiten')
        cp.compress(args); m=hold['model']; tok=hold['tokenizer']; sel=hold['selection']; rs=hold['rank_sum']; tr=hold['total_rank']
        rr=resource(m,sel,rs,tr,gs)
        alloc={'schema':'PALU_TABLE1_USER_UNIFORM_ALLOCATION_V1','cell':cell,'gs':gs,'search_method':'uniform','fisher_enabled':False,'seed':42,'allocation_sha256':rr['allocation_sha256'],'rank_sum':rs,'total_rank':tr,'rank_histogram':rr['rank_histogram'],'ranks':{k:[int(v) for v in x] for k,x in sorted(sel.items())}}
        savej(outdir/'allocation.json',alloc)
        if not rr['resource_pass']: raise RuntimeError('exact rank/resource census failed '+str(rr))
        ppls={}
        for d in dsets:
            vals=run_ppl_eval.eval_ppl(m,tok,alias,d,seqlen=2048,device='cuda'); ppls.update(vals); print('CELL_METRIC',cell,d,vals[d],flush=True)
        result={'schema':'PALU_TABLE1_USER_UNIFORM_PPL_RESULT_V1','campaign':'PALU_OFFICIAL_TABLE1_REPRO_V1','cell':cell,'method':'J-LRD' if gs==32 else ('M-LRD' if gs==1 else 'G-LRD'),'group_size':gs,'allocator':'global uniform; Fisher disabled by request','classification':'PROTOCOL_MISMATCH_TO_LIKELY_OFFICIAL_TABLE1_ALLOCATION; user-requested diagnostic','official_repo_commit':'bb22666e2ef96707e8dd21d93fc00146c2e0d615','model_alias':alias,'seed':42,'decomposition':'official whiten/truncation-aware SVD path','sequence_length':2048,'stride':2048,'ppl':ppls,'resource':rr,'dataset_sha256':SHA,'started_utc':start,'completed_utc':utc(),'foreign_jobs_interrupted':0,'historical_results_modified':False}
        digest=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest(); result['result_sha256']=digest
        savej(outdir/'result.json',result)
        state['cells'][cell]={'status':'DONE','gs':gs,'allocator':'uniform','fisher_enabled':False,'ppl':ppls,'resource':rr,'allocation_path':str(outdir/'allocation.json'),'result_path':str(outdir/'result.json'),'result_sha256':digest,'completed_utc':result['completed_utc'],'old_results_valid':True}
        print('CELL_DONE',cell,digest,flush=True)
    except Exception as e:
        fail={'status':'TODO' if 'out of memory' in str(e).lower() else 'FAILED','failure_type':'GPU_MEMORY_RUNTIME_FAILURE' if 'out of memory' in str(e).lower() else type(e).__name__,'reason':str(e),'traceback':traceback.format_exc(),'failed_utc':utc(),'old_results_valid':True}
        state['cells'][cell]=fail; savej(outdir/'FAILURE.json',fail); print('CELL_FAILED',cell,type(e).__name__,str(e),flush=True)
    finally:
        m=hold.pop('model',None); hold.pop('tokenizer',None); hold.pop('selection',None)
        if m is not None: del m
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    savej(STATE,state)
state['status']='COMPLETE' if all(state.get('cells',{}).get(c,{}).get('status')=='DONE' for c,_,_ in specs) else 'PARTIAL_TODO_OR_FAILED'
state['finished_utc']=utc(); state['foreign_jobs_interrupted']=0; savej(STATE,state)
done=BASE/'locks/UNIFORM_SWEEP_DONE'; done.mkdir(parents=True,exist_ok=True)
savej(done/'DONE.json',{'run_id':active['run_id'],'pid':os.getpid(),'finished_utc':utc(),'status':state['status'],'foreign_jobs_interrupted':0})
try: (LOCKDIR/'ACTIVE.json').unlink()
except FileNotFoundError: pass
print('SWEEP_'+state['status'],flush=True)
