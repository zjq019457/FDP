"""Evaluation-only, source-pinned MPT five-task accuracy, full prediction logs."""
import os,sys,json,hashlib,importlib.util,argparse,datetime,math
from pathlib import Path
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def canonical_sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,str(p));m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def dump(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,default=convert)+'\n');os.replace(tmp,p)
def convert(x):
 if hasattr(x,'tolist'):return x.tolist()
 raise TypeError(type(x).__name__)
ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--tier',required=True);ap.add_argument('--arm',required=True);args=ap.parse_args()
mp=Path(args.manifest);manifest=json.loads(mp.read_text());mh=digest(mp);E=Path(manifest['E_root']);F=Path(manifest['F_root'])
if mh!=(E/'FORMAL_ACCURACY_MANIFEST.sha256').read_text().strip():raise RuntimeError('manifest changed')
for path,row in manifest['sources'].items():
 if digest(path)!=row['sha256']:raise RuntimeError('source changed '+path)
for path,sha in {**manifest['runtime_sources'],**manifest['dataset_files']}.items():
 if digest(path)!=sha:raise RuntimeError('scorer/data changed '+path)
cell=next(c for c in manifest['cells'] if c['tier']==args.tier and c['arm']==args.arm)
if not cell['launch_authorized'] or not cell['resource_lock_pass']:raise RuntimeError('cell not authorized')
os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_CACHE=str(F/'cache'),TOKENIZERS_PARALLELISM='false')
import torch,numpy as np
from transformers import AutoTokenizer
from lm_eval import evaluator
from lm_eval.tasks import TaskManager,get_task_dict
from lm_eval.models.huggingface import HFLM
device=torch.device('cuda:0');free,total=torch.cuda.mem_get_info(device)
if free<36*1024**3:raise RuntimeError('GPU_SAFE_SLOT_UNAVAILABLE')
torch.cuda.set_per_process_memory_fraction(.40,device);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(2)
ppl=load('formal_ppl',E/'sources/canonical_ppl.py');ppl.DECOMP=E/'sources/decompose_modules.py'
paths=dict(ppl.TIER_PATHS[args.tier]);paths['relaxed_lock' if args.arm=='relaxed_uniform' else 'fdp_lock']=Path(cell['allocation'])
raw=F/'raw'/args.tier/args.arm;summary_dir=E/'results'/args.tier/args.arm;raw.mkdir(parents=True,exist_ok=True);summary_dir.mkdir(parents=True,exist_ok=True)
model,rs=ppl.realize(args.arm,device,paths)
assert rs==cell['rank_sha256']
protocol=load('formal_protocol',E/'sources/mpt_fdp_protocol_v1.py')
with torch.inference_mode():
 ids=torch.ones((1,2048),dtype=torch.long,device=device)
 cached=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True,return_dict=True)
 census=protocol.actual_cache_census(cached.past_key_values,cell['ranks'])
assert census['b_live_bytes']==cell['actual_b_live'];del cached,ids;torch.cuda.empty_cache()
dump(summary_dir/'RUNTIME_CACHE_BINDING.json',dict(census,manifest_sha256=mh,allocation_sha256=cell['allocation_sha256']))
factors=[]
for li,block in enumerate(model.transformer.blocks):
 ps=[]
 for name,p in block.attn.named_parameters():
  t=p.detach().cpu().contiguous();ps.append({'name':name,'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()});del t
 factors.append({'layer':li,'ranks':cell['ranks'][li],'parameters':ps})
dump(summary_dir/'MATERIALIZATION_BINDING.json',{'manifest_sha256':mh,'layers':factors,'rank_sha256':rs})
tok=AutoTokenizer.from_pretrained(manifest['model_path'],local_files_only=True,use_fast=True,trust_remote_code=False)
lm=HFLM(pretrained=model,tokenizer=tok,batch_size=1,max_length=2048,logits_cache=False,add_bos_token=False)
sys.path.insert(0,str(E/'tasks'));tm=TaskManager(include_path=str(E/'tasks'));task_dict=get_task_dict(list(manifest['tasks']),tm)
scores={};outputs=[]
for name,contract in manifest['tasks'].items():
 dst=raw/(name+'.json');metric=contract['metric']
 if dst.exists():
  prior=json.loads(dst.read_text())
  if prior['manifest_sha256']!=mh or prior['rank_schedule_sha256']!=rs or prior['status']!='PASS':raise RuntimeError('nonmatching previous formal output')
  scores[name]=prior['primary_score'];outputs.append({'path':str(dst),'sha256':digest(dst)});continue
 print('TASK_START',args.tier,args.arm,name,flush=True)
 result=evaluator.simple_evaluate(model=lm,tasks=[name],task_manager=tm,num_fewshot=0,batch_size=1,device='cuda:0',log_samples=True,limit=None,cache_requests=False,bootstrap_iters=0,random_seed=2,numpy_random_seed=2,torch_random_seed=2,fewshot_random_seed=2)
 samples=result['samples'][name];task=task_dict[name];ordered=[]
 for row in samples:
  choices=task.doc_to_choice(row['doc']);ll=np.array([r[0] for r in row['filtered_resps']],dtype=float)
  if not np.isfinite(ll).all():raise RuntimeError('nonfinite choice likelihood')
  gold=task.doc_to_text(row['doc']) if task.multiple_input else task.doc_to_target(row['doc'])
  if isinstance(gold,str):gold=choices.index(gold) if gold in choices else -100
  pred=int(np.argmax(ll));lengths=np.array([len(x) for x in choices],dtype=float)
  pred_norm=int(np.argmax(ll/lengths))
  chosen=pred_norm if metric=='acc_norm' else pred
  independent=float(chosen in gold if isinstance(gold,list) else chosen==gold)
  if independent!=float(row[metric]):raise RuntimeError('stored correctness/prediction parity failed')
  row.update(prediction_index=chosen,prediction_acc_index=pred,prediction_acc_norm_index=pred_norm,scoring_gold=gold,choice_character_lengths=lengths.tolist(),primary_metric=metric)
  ordered.append({'doc_id':row['doc_id'],'doc_sha256':canonical_sha(row['doc'])})
 if len(samples)!=contract['n'] or canonical_sha(ordered)!=contract['ordered_document_sha256']:raise RuntimeError('sample identity/order mismatch')
 score=float(result['results'][name][metric+',none'])
 if abs(np.mean([r[metric] for r in samples])-score)>1e-12:raise RuntimeError('score/sample mismatch')
 rec={'status':'PASS','tier':args.tier,'arm':args.arm,'task':name,'primary_metric':metric,'primary_score':score,'n':len(samples),'samples':samples,'raw_metrics':result['results'][name],'manifest_sha256':mh,'runner_sha256':digest(__file__),'rank_schedule_sha256':rs,'allocation_sha256':cell['allocation_sha256'],'source_binding':'PASS_CURRENT_EXECUTION_PINNED','actual_b_live':cell['actual_b_live'],'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 dump(dst,rec);scores[name]=score;outputs.append({'path':str(dst),'sha256':digest(dst)})
 print('TASK_DONE',args.tier,args.arm,name,score,flush=True)
 summary={'status':'RUNNING' if len(scores)<5 else 'COMPLETE','manifest_sha256':mh,'tier':args.tier,'arm':args.arm,'per_task':scores,'macro':sum(scores.values())/len(scores),'outputs':outputs,'rank_sha256':rs,'actual_b_live':cell['actual_b_live'],'source_binding':'PASS_CURRENT_EXECUTION_PINNED'}
 dump(summary_dir/'ACC_SUMMARY.json',summary)
print('ARM_DONE',args.tier,args.arm,summary['macro'],flush=True)
