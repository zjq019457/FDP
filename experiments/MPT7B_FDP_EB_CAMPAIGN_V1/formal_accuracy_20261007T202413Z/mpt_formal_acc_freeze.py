"""Freeze existing six MPT arms and source-bound evaluation inputs, no inference."""
import os,sys,json,hashlib,shutil,datetime,subprocess,importlib.util
from pathlib import Path
E=Path(sys.argv[1]);F=Path(sys.argv[2]);HERE=Path(__file__).parent
ROOT=Path('/diskE/jiaqizhang/MPT7B_FDP_EB_CAMPAIGN_V1')
AUDIT=ROOT/'audits/repair_20261007T200635Z'
CODE=Path('/home/bai-project/jiaqizhang/EigenAttn/experimental/mpt7b_fdp_eb_pipeline_v1')
INCLUDE=Path('/diskF/jiaqizhang/a1804298693/offline_datasets')
TASKS={'arc_challenge_localjsonl':'acc_norm','arc_easy_localjsonl':'acc_norm','hellaswag_localjsonl':'acc','piqa_localjsonl':'acc','winogrande_localjsonl':'acc'}
COUNTS=[1172,2376,10042,1838,1267]
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def csha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
if (E/'FORMAL_ACCURACY_MANIFEST.json').exists():raise RuntimeError('manifest already frozen')
for p in [E/'sources',E/'tasks',F/'raw',F/'logs',F/'cache'] :p.mkdir(parents=True,exist_ok=True)
sources={}
def snapshot(src,dst):
 shutil.copy2(src,dst);sources[str(dst)]={'sha256':digest(dst),'original':str(src),'original_sha256':digest(src)}
snapshot(CODE/'run_u011_diagnostic_ppl.py',E/'sources/canonical_ppl.py')
snapshot(CODE/'run_mpt_diagnostic_acc.py',E/'sources/historical_accuracy_runner.py')
snapshot('/home/bai-project/jiaqizhang/EigenAttn/EigenAttn_base/models/decompose_modules.py',E/'sources/decompose_modules.py')
snapshot(ROOT/'adapters/mpt_fdp_protocol_v1.py',E/'sources/mpt_fdp_protocol_v1.py')
for name in ['mpt_formal_acc_runner.py','mpt_formal_acc_scheduler.py','mpt_formal_acc_statistics.py','mpt_formal_acc_freeze.py']:
 sources[str(E/name)]={'sha256':digest(E/name),'original':str(E/name),'original_sha256':digest(E/name)}
for t in TASKS:snapshot(INCLUDE/(t+'.yaml'),E/'tasks'/(t+'.yaml'))
snapshot(INCLUDE/'preprocess_winogrande_local.py',E/'tasks/preprocess_winogrande_local.py')
identity_path=ROOT/'identity/MPT7B_CHECKPOINT_CONTENT_MANIFEST_V1.json';identity=read(identity_path)
model=Path(identity['remote_checkpoint_path']);model_files=[]
for r in identity['checkpoint_files']:
 p=model/r['path'];actual=digest(p)
 if actual!=r['sha256']:raise RuntimeError('model file changed '+str(p))
 model_files.append({'path':str(p),'sha256':actual,'bytes':p.stat().st_size})
os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_CACHE=str(F/'cache'),TOKENIZERS_PARALLELISM='false')
import torch,numpy,transformers,lm_eval,datasets,yaml
from lm_eval import evaluator
from lm_eval.api import task as task_module
from lm_eval.models import huggingface
from lm_eval.tasks import TaskManager,get_task_dict
from transformers.models.mpt import modeling_mpt
runtime_sources={str(Path(m.__file__)):digest(Path(m.__file__)) for m in [evaluator,task_module,huggingface,modeling_mpt]}
datasets_hash={};tasks_manifest={}
tm=TaskManager(include_path=str(E/'tasks'));td=get_task_dict(list(TASKS),tm)
for (name,metric),count in zip(TASKS.items(),COUNTS):
 config=yaml.safe_load((E/'tasks'/(name+'.yaml')).read_text().replace('!function ',' '))
 for path in config['dataset_kwargs']['data_files'].values():datasets_hash[path]=digest(path)
 task=td[name];docs=list(task.eval_docs);ordered=[]
 if len(docs)!=count:raise RuntimeError('sample count differs '+name)
 for i,doc in enumerate(docs):
  ordered.append({'doc_id':i,'doc_sha256':csha(doc)})
 sample_path=E/'tasks'/(name+'_SAMPLE_IDS.json');dump(sample_path,ordered)
 tasks_manifest[name]={'metric':metric,'n':count,'yaml_sha256':digest(E/'tasks'/(name+'.yaml')),'sample_manifest':str(sample_path),'sample_manifest_sha256':digest(sample_path),'ordered_document_sha256':csha(ordered)}
cells=[]
for tier in ['U011','U015','U019']:
 for arm in ['relaxed_uniform','fdp']:
  path=ROOT/tier/('10_RELAXED_UNIFORM/RELAXED_UNIFORM_RANK_LOCK.json' if arm=='relaxed_uniform' else '04_SCHEDULE/'+tier+'_FDP_RANK_LOCK.json')
  lock=read(path);dest=E/'sources'/(tier+'_'+arm+'_ALLOCATION.json');snapshot(path,dest)
  census=read(AUDIT/(tier+'_'+arm+'_ACTUAL_CACHE.json'))
  assert census['runtime_rank_sha']==lock['rank_schedule_sha256']
  cells.append({'tier':tier,'arm':arm,'allocation':str(dest),'allocation_sha256':digest(dest),'rank_sha256':lock['rank_schedule_sha256'],'ranks':lock['rank_schedule'],'actual_b_live':census['b_live_bytes'],'runtime_census':str(AUDIT/(tier+'_'+arm+'_ACTUAL_CACHE.json')),'runtime_census_sha256':digest(AUDIT/(tier+'_'+arm+'_ACTUAL_CACHE.json')),'resource_lock_pass':True,'launch_authorized':True})
historical=[]
for tier in ['U011','U015','U019']:
 for arm in ['relaxed_uniform','fdp']:
  for p in sorted((ROOT/tier/'11_ACCURACY'/arm).glob('*.json')):
   historical.append({'path':str(p),'sha256':digest(p),'classification':'HISTORICAL_NON_FORMAL','reason':'historical execution source binding partial; preserved immutable'})
dump(E/'HISTORICAL_ACCURACY_CLASSIFICATION.json',historical)
commit=subprocess.check_output(['git','-C','/home/bai-project/jiaqizhang/EigenAttn/EigenAttn_base','rev-parse','HEAD'],text=True).strip()
status=subprocess.check_output(['git','-C','/home/bai-project/jiaqizhang/EigenAttn/EigenAttn_base','status','--porcelain'],text=True)
(E/'BASE_REPO_DIRTY_STATUS.txt').write_text(status)
manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Evaluation-only rerun; six frozen schedules unchanged; no PPL rerun','status':'FROZEN_CELL_LAUNCH_AUTHORIZED','model_path':str(model),'checkpoint_identity':{'path':str(identity_path),'sha256':digest(identity_path)},'model_files':model_files,'sources':sources,'runtime_sources':runtime_sources,'dataset_files':datasets_hash,'tasks':tasks_manifest,'cells':cells,'git_commit':commit,'git_dirty_status_sha256':digest(E/'BASE_REPO_DIRTY_STATUS.txt'),'versions':{m.__name__:getattr(m,'__version__','unknown') for m in [torch,numpy,transformers,lm_eval,datasets]},'eval':{'num_fewshot':0,'batch_size':1,'max_length':2048,'add_bos_token':False,'seed':2,'logits_cache':False,'full_sample_logging':True},'statistics':{'seed':20261007,'bootstrap':10000,'aggregate':'equal-task macro','paired':'within-task item resampling','McNemar':'two-sided exact binomial; Holm across 15 tests'},'E_root':str(E),'F_root':str(F),'foreign_jobs_interrupted':0,'PPL_status':'PRESERVED_NO_RERUN_SOURCE_BINDING_PARTIAL_NOT_AUTO_PROMOTED'}
dump(E/'FORMAL_ACCURACY_MANIFEST.json',manifest)
(E/'FORMAL_ACCURACY_MANIFEST.sha256').write_text(digest(E/'FORMAL_ACCURACY_MANIFEST.json')+'\n')
print('FROZEN',E,'6 arms,30 tasks',flush=True)
