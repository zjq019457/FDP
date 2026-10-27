import os,pathlib,json,hashlib,subprocess,time,datetime,fcntl,csv,sys
B=pathlib.Path('/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1'); L=B/'longbench_supplemental_4096'; M=L/'LONGBENCH_MANIFEST.json'; P='/home/bai-project/jiaqizhang/.venv/bin/python'; runner=B/'code/run_palu_longbench_supplemental.py'; side=L/'audit/multi_news_pair_scheduler';side.mkdir(parents=True,exist_ok=True); events=side/'events.jsonl'
EXPECTED='9519dc984b48e4815487207b1580b0dc807353be37de8f699a879d35657ed990'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def j(p):
 try:return json.loads(pathlib.Path(p).read_text())
 except:return None
def emit(s,**k):
 d={'utc':now(),'status':s,'foreign_jobs_interrupted':0,**k};events.open('a').write(json.dumps(d,sort_keys=True)+'\n');print(json.dumps(d,sort_keys=True),flush=True)
def gpus():
 t=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader'],text=True)
 return {int(a.strip()):int(b.split()[0]) for a,b in (x.split(',',1) for x in t.splitlines())}
def has_worker():
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   a=[x.decode(errors='replace') for x in p.joinpath('cmdline').read_bytes().split(bytes([0])) if x]
   if any(pathlib.Path(x).name=='run_palu_longbench_supplemental.py' for x in a) and '--task' in a and a[a.index('--task')+1]=='multi_news':return True
  except:pass
 return False
def write_pipe(status,rec):
 q=B/'PALU_PIPELINE_STATE.json';d=j(q) or {};rat=d.setdefault('longbench_supplemental',{}).setdefault('ratio_0.5',{})
 for m in ('PALU_UNIFORM','PALU_FDP_NATIVE'):rat.setdefault(m,{})
 rat['PALU_UNIFORM']['multi_news']={**(rat['PALU_UNIFORM'].get('multi_news') or {}),**rec['uniform']}
 rat['PALU_FDP_NATIVE']['multi_news']={**(rat['PALU_FDP_NATIVE'].get('multi_news') or {}),**rec['fdp_native']}
 d['multi_news_pair_scheduler']={'status':status,'updated_utc':now(),'audit_dir':str(side),'foreign_jobs_interrupted':0};d['foreign_jobs_interrupted']=0
 tmp=q.with_suffix('.json.tmp');tmp.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');os.replace(tmp,q)
fd=os.open(side/'scheduler.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
if sha(M)!=EXPECTED:emit('ABORT_MANIFEST_SHA_MISMATCH',observed=sha(M));sys.exit(2)
md=j(M) or {};task=md.get('task_files',{}).get('multi_news',{});dataset=pathlib.Path(task.get('path','/nonexistent'))
if not task or not dataset.exists() or sha(dataset)!=task.get('sha256'):emit('ABORT_TASK_DATASET_BINDING_MISMATCH');sys.exit(2)
outs={m:L/'ratio_0.5'/m/'multi_news.raw.jsonl' for m in ('PALU_UNIFORM','PALU_FDP_NATIVE')}
if any(p.exists() for p in outs.values()):emit('EXISTING_OUTPUT_NO_OVERWRITE',outputs={k:str(v) for k,v in outs.items()});sys.exit(0)
if has_worker():emit('EXISTING_WORKER_NO_DUPLICATE');sys.exit(0)
rec={'uniform':{'status':'READY_WAITING_FOR_SAFE_GPU','task':'multi_news','method':'uniform','n_samples_expected':task.get('n_samples'),'manifest_file_sha256':sha(M),'dataset_sha256':sha(dataset),'foreign_jobs_interrupted':0},'fdp_native':{'status':'READY_WAITING_FOR_SAFE_GPU','task':'multi_news','method':'fdp_native','n_samples_expected':task.get('n_samples'),'manifest_file_sha256':sha(M),'dataset_sha256':sha(dataset),'foreign_jobs_interrupted':0}}
write_pipe('READY_WAITING_FOR_SAFE_GPU',rec);emit('READY_WAITING_FOR_SAFE_GPU',free_mib=gpus(),required_free_mib_per_gpu=40000,task='multi_news')
while True:
 if any(p.exists() for p in outs.values()):emit('OUTPUT_APPEARED_EXTERNALLY_NO_OVERWRITE');sys.exit(0)
 if has_worker():emit('WORKER_APPEARED_EXTERNALLY_NO_DUPLICATE');sys.exit(0)
 free=gpus();eligible=[i for i in (0,2,1,3) if free.get(i,0)>=40000]
 if len(eligible)>=2:break
 time.sleep(120)
selected=eligible[:2];emit('GPU_SAFE_DISPATCH',gpu_pair=selected,free_mib=free)
procs=[];started={}
for method,gpu in zip(('uniform','fdp_native'),selected):
 runid=f'RATIO0P5_{method.upper()}_MULTINEWS_GPU{gpu}_{datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
 log=L/'logs'/f'{runid}.log';log.parent.mkdir(parents=True,exist_ok=True);f=log.open('ab',buffering=0)
 p=subprocess.Popen([P,'-u',str(runner),'--ratio','0.5','--method',method,'--task','multi_news','--gpu',str(gpu)],cwd=B,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 procs.append((method,p,f,runid,log,gpu));out=outs['PALU_UNIFORM' if method=='uniform' else 'PALU_FDP_NATIVE']
 started[method]={'status':'ACTIVE','run_id':runid,'pid':p.pid,'gpu':gpu,'task':'multi_news','method':method,'started_at_utc':now(),'manifest_file_sha256':sha(M),'dataset_sha256':sha(dataset),'log':str(log),'raw_output':str(out),'n_samples_expected':task.get('n_samples'),'foreign_jobs_interrupted':0,'formal_results_modified':False}
(L/'audit'/f'{runid}_ACTIVE.json').write_text(json.dumps(started[method],indent=2,sort_keys=True)+'\n')
rec=started;write_pipe('ACTIVE',rec);emit('PAIRED_WORKERS_STARTED',workers=started,free_mib_at_launch=free)
while any(p.poll() is None for _,p,_,_,_,_ in procs):
 time.sleep(60)
status={}
for method,p,f,runid,log,gpu in procs:
 rc=p.returncode;f.close();m='PALU_UNIFORM' if method=='uniform' else 'PALU_FDP_NATIVE';out=outs[m];summary=out.with_suffix('.json');audit={}
 if out.exists():
  rows=0;bad=0;ids=set()
  for line in out.open(errors='replace'):
   try:
    x=json.loads(line);rows+=1;i=x.get('id',x.get('sample_id'));ids.add(str(i)) if i is not None else None
   except:bad+=1
  audit={'n_samples_completed':rows,'unique_sample_ids':len(ids),'parse_errors':bad,'raw_jsonl_sha256':sha(out)}
 sm=j(summary) if summary.exists() else None
 status[method]={'status':'COMPLETE' if rc==0 and sm else 'FAILED','exit_code':rc,'result_summary':sm,'raw_audit':audit,'log':str(log),'foreign_jobs_interrupted':0,'formal_results_modified':False}
rec2={'uniform':status.get('uniform',{}),'fdp_native':status.get('fdp_native',{})}
for method in ('uniform','fdp_native'):
 p=outs['PALU_UNIFORM' if method=='uniform' else 'PALU_FDP_NATIVE'];s=j(p.with_suffix('.json')) if p.with_suffix('.json').exists() else None
 if s:rec2[method].update({'score':s.get('score'),'n_samples_completed':s.get('n_samples'),'raw_jsonl_sha256':sha(p) if p.exists() else None,'result_sha256':sha(p.with_suffix('.json')),'status':'COMPLETE'})
complete=all(rec2[m].get('status')=='COMPLETE' for m in ('uniform','fdp_native'));write_pipe('COMPLETE' if complete else 'PAIR_FAILED_OR_INCOMPLETE',rec2);emit('PAIR_COMPLETE' if complete else 'PAIR_FAILED_OR_INCOMPLETE',results=rec2,delta_native_minus_uniform=(rec2['fdp_native'].get('score',0)-rec2['uniform'].get('score',0)) if complete else None)