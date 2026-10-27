import os,pathlib,json,hashlib,subprocess,time,datetime,fcntl,sys
R=pathlib.Path('/diskE/jiaqizhang/PALU_FDP_NATIVE_RESIDUAL_SEARCH_V3'); P='/home/bai-project/jiaqizhang/.venv/bin/python'; runner=R/'run_rs_v3.py'; side=R/'runtime_controls/post_tune_controller'; side.mkdir(parents=True,exist_ok=True); events=side/'events.jsonl'
E={'manifest_sha256':'0041bdf6f296c7ce1783e2fb45bc68089c68e8f6652a0924feb647c7dc941c93','method_card_sha256':'dacb622d47629fc4c55215e485f50c30b9064aa3c30df2b469c950d5c4a3b07f','runner_sha256':'1632bab4ccac3442246f2e593068fd6393caa04a9b55834617d124b44c47fdf3'}
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def emit(s,**k):
 d={'utc':utc(),'status':s,'foreign_jobs_interrupted':0,**k};events.open('a').write(json.dumps(d,sort_keys=True)+'\n');print(json.dumps(d,sort_keys=True),flush=True)
def rd(p):
 try:return json.loads(pathlib.Path(p).read_text())
 except:return None
def active():
 out=[]
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   a=[x.decode(errors='replace') for x in p.joinpath('cmdline').read_bytes().split(bytes([0])) if x]
   if any(pathlib.Path(z).name=='run_rs_v3.py' for z in a) and '--phase' in a:out.append({'pid':int(p.name),'phase':a[a.index('--phase')+1] if a.index('--phase')+1<len(a) else 'unknown','args':a})
  except:pass
 return out
def free():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader'],text=True)
 return {int(a.strip()):int(b.split()[0]) for a,b in (x.split(',',1) for x in s.splitlines())}
def gpu():
 f=free()
 for i in (1,0,2,3):
  if f.get(i,0)>=30000:return i,f
 return None,f
def runphase(ph,g):
 path=side/f'{ph}.log';emit(ph.upper()+'_START',gpu=g,free_mib=free(),log=str(path))
 with path.open('ab') as o:r=subprocess.run([P,str(runner),'--gpu',str(g),'--phase',ph,'--arm','all'],cwd=R,stdin=subprocess.DEVNULL,stdout=o,stderr=subprocess.STDOUT)
 emit(ph.upper()+'_EXIT',code=r.returncode,log=str(path));return r.returncode
fd=os.open(side/'controller.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
if hashlib.sha256(runner.read_bytes()).hexdigest()!=E['runner_sha256']:emit('ABORT_RUNNER_SHA_MISMATCH');sys.exit(2)
s=rd(R/'STATE.json') or {}
if any(s.get(k)!=v for k,v in E.items() if k!='runner_sha256'):emit('ABORT_FROZEN_STATE_BINDING_MISMATCH');sys.exit(2)
emit('WATCHING_TUNE',pid=s.get('tune_worker_pid'),state=s.get('status'),live=active())
while active():
 emit('WAIT_TUNE_EXIT',pids=[x['pid'] for x in active()],phases=[x['phase'] for x in active()],tune_done=(R/'CALIBRATION_TUNE/TUNE_DONE.json').exists());time.sleep(300)
t=rd(R/'CALIBRATION_TUNE/TUNE_DONE.json')
if not t or t.get('status')!='TUNE_GRID_COMPLETE':emit('TUNE_EXITED_WITHOUT_DONE',state=rd(R/'STATE.json'));sys.exit(1)
sel=[]
for a in 'ABCD':
 d=rd(R/f'CALIBRATION_TUNE/ARM_{a}_SELECTED.json')
 if not d or d.get('status')!='ARM_TUNE_CONFIG_SELECTED' or not rd(R/'CALIBRATION_TUNE'/d['selected_config_id']/'DONE.json'):emit('SELECTION_GATE_FAILED',arm=a);sys.exit(1)
 sel.append({'arm':a,'config_id':d['selected_config_id'],'strict':d.get('tune_strict_vs_v1'),'J':d.get('final_J_tune')})
nd=len(list((R/'CALIBRATION_TUNE').glob('arm*/DONE.json')))
if t.get('arm_count')!=4 or t.get('config_count')!=nd:emit('GRID_INTEGRITY_FAILED',tune_done=t,done_count=nd);sys.exit(1)
emit('TUNE_GRID_COMPLETE',tune_done=t,selected_arms=sel)
l=rd(R/'CALIBRATION_LOCK/V3_LOCK.json')
if l is None:
 g,f=gpu()
 while g is None:emit('WAIT_SAFE_GPU_LOCK',free_mib=f);time.sleep(120);g,f=gpu()
 if runphase('lock',g)!=0:emit('LOCK_RUNNER_FAILED',state=rd(R/'STATE.json'));sys.exit(1)
 l=rd(R/'CALIBRATION_LOCK/V3_LOCK.json')
if not l or l.get('status')!='V3_CALIBRATION_LOCK_FROZEN':emit('NO_V3_GO_QUALITY',lock=l,formal_skipped=True);sys.exit(0)
a=l.get('chosen_arm');ad=rd(R/f'CALIBRATION_LOCK/ARM_{a}_LOCK.json') if a else None;ci=l.get('lock_ci95_vs_v1',[]);delta=l.get('lock_delta_J_vs_v1')
if not ad or not ad.get('strict_lock_pass') or delta is None or float(delta)>=0 or len(ci)!=2 or float(ci[1])>=0:emit('LOCK_STRICT_GATE_FAILED',lock=l,arm_lock=ad,formal_skipped=True);sys.exit(1)
if not (R/'FORMAL_PPL/DONE.json').exists():
 g,f=gpu()
 while g is None:emit('WAIT_SAFE_GPU_FORMAL',free_mib=f);time.sleep(120);g,f=gpu()
 if runphase('formal',g)!=0:emit('FORMAL_RUNNER_FAILED',state=rd(R/'STATE.json'));sys.exit(1)
emit('FORMAL_DONE',result=rd(R/'FORMAL_PPL/DONE.json'))