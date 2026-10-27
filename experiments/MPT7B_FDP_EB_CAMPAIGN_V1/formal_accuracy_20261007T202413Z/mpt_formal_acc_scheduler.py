"""One persistent scheduler, six frozen arms, safe GPU stacking, no foreign mutation."""
import os,sys,json,time,subprocess,fcntl,datetime
from pathlib import Path
E=Path(sys.argv[1]);M=E/'FORMAL_ACCURACY_MANIFEST.json';manifest=json.loads(M.read_text());F=Path(manifest['F_root']);PY=sys.executable
lock=(E/'SCHEDULER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
active={};attempts={};failure_counts={};state={};cells=manifest['cells'];slots={}
def write_state():
 data={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'foreign_jobs_interrupted':0,'active':[{**r,'process':None} for r in active.values()],'states':state,'gpu_snapshot':slots}
 data['active']=[{k:v for k,v in r.items() if k not in ['process','log']} for r in active.values()]
 tmp=E/'SCHEDULER_STATE.json.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,E/'SCHEDULER_STATE.json')
def gpus():
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 result={}
 for line in text.strip().splitlines():
  i,u,free,total,util=[x.strip() for x in line.split(',')];result[int(i)]={'uuid':u,'free_MiB':int(free),'total_MiB':int(total),'util':int(util),'effective_free_MiB':int(free)}
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
 used={}
 for line in apps.strip().splitlines():
  try:p,b=[x.strip() for x in line.split(',')];used[int(p)]=int(b)
  except ValueError:pass
 for r in active.values():
  if r['gpu'] in result:result[r['gpu']]['effective_free_MiB']-=max(0,int(result[r['gpu']]['total_MiB']*.4)-used.get(r['pid'],0))
 return result
for c in cells:
 key=c['tier']+'_'+c['arm'];state[key]='READY'
while True:
 for key,r in list(active.items()):
  rc=r['process'].poll()
  if rc is None:continue
  r['log'].close();del active[key]
  if rc==0:state[key]='DONE'
  else:
   failure_counts[key]=failure_counts.get(key,0)+1
   text=Path(r['log_path']).read_text(errors='replace')[-50000:]
   engineering=any(x in text for x in ['CUDA out of memory','GPU_SAFE_SLOT_UNAVAILABLE','ConnectionResetError'])
   state[key]='READY' if engineering and failure_counts[key]<3 else 'BLOCKED'
   (E/(key+'_attempt'+str(r['attempt'])+'_FAILURE.json')).write_text(json.dumps({'status':'ENGINEERING_FAILURE' if engineering else 'REQUIRES_AUDIT','exit_code':rc,'log':r['log_path'],'retry_queued':state[key]=='READY'},indent=2)+'\n')
 if all(v=='DONE' for v in state.values()):
  write_state();subprocess.run([PY,str(E/'mpt_formal_acc_statistics.py'),str(E)],check=True);break
 try:slots=gpus()
 except Exception as ex:
  (E/'GPU_QUERY_FAILURE.log').open('a').write(str(ex)+'\n');time.sleep(15);continue
 for c in cells:
  key=c['tier']+'_'+c['arm']
  if state[key]!='READY':continue
  possible=[i for i,r in slots.items() if r['effective_free_MiB']>=36864]
  if not possible:break
  gpu=max(possible,key=lambda i:slots[i]['effective_free_MiB']);attempts[key]=attempts.get(key,0)+1
  logpath=F/'logs'/(key+'_attempt'+str(attempts[key])+'.log');fh=logpath.open('a')
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
  cmd=[PY,'-u',str(E/'mpt_formal_acc_runner.py'),'--manifest',str(M),'--tier',c['tier'],'--arm',c['arm']]
  p=subprocess.Popen(cmd,stdout=fh,stderr=subprocess.STDOUT,env=env)
  active[key]={'process':p,'pid':p.pid,'gpu':gpu,'attempt':attempts[key],'log':fh,'log_path':str(logpath),'command':cmd,'launch_free_MiB':slots[gpu]['free_MiB']};state[key]='RUNNING'
  slots[gpu]['effective_free_MiB']-=int(slots[gpu]['total_MiB']*.4)
  print('LAUNCH',key,'GPU',gpu,'PID',p.pid,flush=True)
 write_state()
 if not active and all(v in ['DONE','BLOCKED'] for v in state.values()):break
 time.sleep(15)
