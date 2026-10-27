#!/home/bai-project/jiaqizhang/.venv/bin/python
import os,sys,json,time,glob,hashlib,pathlib,subprocess,datetime,fcntl,shutil
B=pathlib.Path('/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1')
LB=B/'longbench_supplemental_4096'; SIDE=LB/'audit/multi_news_pair_scheduler'
MAN=LB/'LONGBENCH_MANIFEST.json'; STATE=B/'PALU_PIPELINE_STATE.json'
RUNNER=B/'code/run_palu_longbench_supplemental.py'
PY='/home/bai-project/jiaqizhang/.venv/bin/python'
MAN_SHA='9519dc984b48e4815487207b1580b0dc807353be37de8f699a879d35657ed990'
SEM_SHA='65c0304ee4e7059fb8eabd621edc9e4f8bf374f8a76594cf3642783456fe76c4'
DATA_SHA='f8817c00af317c2d7f81128f334bcfe5780c6c2a515d5c96811aa0ddf8b5ee6b'
Q_SHA='6f3b59993abbffbd932825b20ff036197c5be6ff63523ab46e94281bc24eecb1'
RANK=32640; CACHE=133693440; MIN_FREE=40000
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def j(p):
    try:return json.loads(pathlib.Path(p).read_text())
    except:return None
def emit(status,**kw):
    r={'utc':now(),'status':status,'foreign_jobs_interrupted':0,**kw}
    with (SIDE/'events_serial.jsonl').open('a') as f:f.write(json.dumps(r,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(r,sort_keys=True),flush=True)
def gpu():
    s=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.free','--format=csv,noheader'],text=True)
    return [x.strip() for x in s.splitlines()]
def free_mib(i=1):
    s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader'],text=True)
    return {int(a.strip()):int(b.split()[0]) for a,b in (x.split(',',1) for x in s.splitlines())}.get(i,0)
def procs():
    s=subprocess.check_output(['ps','-eo','user,pid,ppid,stat,etime,%cpu,args'],text=True)
    keys=('run_rs_v3.py','run_palu_longbench_supplemental.py','mie_v2_q256','palu_multinews_pair_scheduler','build_eigen_basis')
    return [x for x in s.splitlines() if any(k in x for k in keys)]
def worker_exists():
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            a=[x.decode(errors='replace') for x in p.joinpath('cmdline').read_bytes().split(bytes([0])) if x]
            if any(pathlib.Path(x).name=='run_palu_longbench_supplemental.py' for x in a):
                if '--task' in a and a[a.index('--task')+1]=='multi_news':return True
        except Exception:pass
    return False
def audit_raw(p):
    n=bad=0;ids=[]
    for line in pathlib.Path(p).open(errors='replace'):
        try:
            x=json.loads(line);n+=1;v=x.get('id',x.get('sample_id'))
            if v is None:v=hashlib.sha256(line.encode()).hexdigest()
            ids.append(str(v))
        except Exception:bad+=1
    return {'parsed':n,'parse_errors':bad,'unique_sample_ids':len(set(ids)),'duplicates':len(ids)-len(set(ids))}
def update_pipe(status,records=None,snapshot=None):
    d=j(STATE) or {}; lane=d.setdefault('longbench_supplemental',{}).setdefault('ratio_0.5',{})
    lane.setdefault('PALU_UNIFORM',{});lane.setdefault('PALU_FDP_NATIVE',{})
    if records:
        for method,rec in records.items():
            k='PALU_UNIFORM' if method=='uniform' else 'PALU_FDP_NATIVE'
            lane[k]['multi_news']=rec
    d['multi_news_pair_scheduler']={'status':status,'mode':'serial_same_gpu1','min_free_mib':MIN_FREE,'updated_utc':now(),'audit_dir':str(SIDE),'foreign_jobs_interrupted':0,'formal_results_modified':False}
    d['foreign_jobs_interrupted']=0;d['formal_results_modified']=False;d['last_audit_utc']=now()
    if snapshot is not None:d['last_remote_scheduling_snapshot']=snapshot
    tmp=STATE.with_suffix('.json.tmp');tmp.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');os.replace(tmp,STATE)
def record_verified(method,task):
    k='PALU_UNIFORM' if method=='uniform' else 'PALU_FDP_NATIVE'
    raw=LB/'ratio_0.5'/k/(task+'.raw.jsonl');sm=raw.with_suffix('.json')
    if not raw.exists() or not sm.exists():return None
    s=j(sm);a=audit_raw(raw)
    expected=200 if task=='multi_news' else 500
    if not s:return None
    return {'status':'COMPLETE','task':task,'method':method,'ratio':0.5,'score':s.get('score'),'n_samples_expected':expected,'n_samples_completed':a['parsed'],'unique_sample_ids':a['unique_sample_ids'],'parse_errors':a['parse_errors'],'duplicates':a['duplicates'],'raw_jsonl_sha256':sha(raw),'result_sha256':sha(sm),'manifest_file_sha256':MAN_SHA,'manifest_sha':s.get('manifest_sha'),'q_sha':s.get('q_sha'),'schedule_sha':s.get('schedule_sha'),'native_rank':s.get('native_rank'),'formal_cache_bytes':s.get('formal_cache_bytes'),'gpu':s.get('gpu'),'validity':s.get('validity'),'formal_results_modified':False,'foreign_jobs_interrupted':0}
def refresh_completed():
    records={}
    for t in ('lcc','repobench-p'):
        for m in ('uniform','fdp_native'):
            r=record_verified(m,t)
            if r and r['validity']=='VALID' and r['n_samples_completed']==r['n_samples_expected'] and r['unique_sample_ids']==r['n_samples_expected'] and not r['parse_errors'] and not r['duplicates']:
                records[f'{m}_{t}']=r
    partial={}
    for m in ('uniform','fdp_native'):
        r=record_verified(m,'qmsum')
        if r:partial[m]=r;continue
        k='PALU_UNIFORM' if m=='uniform' else 'PALU_FDP_NATIVE'
        raw=LB/'ratio_0.5'/k/'qmsum.raw.jsonl'
        if raw.exists():
            a=audit_raw(raw);partial[m]={'status':'ACTIVE_OR_PARTIAL_NO_SCORE','task':'qmsum','method':m,'n_samples_expected':200,'n_samples_completed':a['parsed'],'unique_sample_ids':a['unique_sample_ids'],'parse_errors':a['parse_errors'],'duplicates':a['duplicates'],'raw_jsonl_sha256':sha(raw),'formal_results_modified':False,'foreign_jobs_interrupted':0}
    return records,partial
def validate(method,rc,runid,log):
    k='PALU_UNIFORM' if method=='uniform' else 'PALU_FDP_NATIVE'
    raw=LB/'ratio_0.5'/k/'multi_news.raw.jsonl';sm=raw.with_suffix('.json')
    a=audit_raw(raw) if raw.exists() else {'parsed':0,'parse_errors':0,'unique_sample_ids':0,'duplicates':0}
    s=j(sm) if sm.exists() else None
    wanted='PALU_UNIFORM' if method=='uniform' else 'PALU_FDP_NATIVE'
    ok=bool(rc==0 and s and s.get('validity')=='VALID' and s.get('method')==wanted and s.get('ratio')==0.5 and s.get('n_samples')==200 and s.get('manifest_sha')==SEM_SHA and s.get('q_sha')==Q_SHA and s.get('native_rank')==RANK and s.get('formal_cache_bytes')==CACHE and a['parsed']==200 and a['unique_sample_ids']==200 and a['parse_errors']==0 and a['duplicates']==0)
    return {'status':'COMPLETE' if ok else 'TODO_BLOCKED_INCOMPLETE','task':'multi_news','method':method,'ratio':0.5,'run_id':runid,'gpu':1,'exit_code':rc,'score':s.get('score') if s else None,'n_samples_expected':200,'n_samples_completed':a['parsed'],'unique_sample_ids':a['unique_sample_ids'],'parse_errors':a['parse_errors'],'duplicates':a['duplicates'],'raw_jsonl_sha256':sha(raw) if raw.exists() else None,'result_sha256':sha(sm) if sm.exists() else None,'manifest_file_sha256':sha(MAN),'manifest_sha':s.get('manifest_sha') if s else None,'dataset_sha256':DATA_SHA,'q_sha':s.get('q_sha') if s else Q_SHA,'schedule_sha':s.get('schedule_sha') if s else None,'native_rank':s.get('native_rank') if s else RANK,'formal_cache_bytes':s.get('formal_cache_bytes') if s else CACHE,'validity':s.get('validity') if s else None,'raw_audit':a,'log':str(log),'formal_results_modified':False,'foreign_jobs_interrupted':0},ok
def main():
    SIDE.mkdir(parents=True,exist_ok=True)
    fd=os.open(SIDE/'scheduler.lock',os.O_CREAT|os.O_RDWR,0o600)
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:emit('BLOCKED_LOCK_HELD');return
    if sha(MAN)!=MAN_SHA:emit('ABORT_MANIFEST_SHA_MISMATCH',observed=sha(MAN));return
    md=j(MAN) or {};task=md.get('task_files',{}).get('multi_news',{});ds=pathlib.Path(task.get('path','/nonexistent'))
    if not ds.exists() or sha(ds)!=DATA_SHA or task.get('sha256')!=DATA_SHA or task.get('n_samples')!=200:emit('ABORT_DATASET_BINDING_MISMATCH');return
    outs={m:LB/'ratio_0.5'/('PALU_UNIFORM' if m=='uniform' else 'PALU_FDP_NATIVE')/'multi_news.raw.jsonl' for m in ('uniform','fdp_native')}
    existing={m:str(p) for m,p in outs.items() if p.exists() or p.with_suffix('.json').exists()}
    if existing:emit('ABORT_EXISTING_OUTPUT_NO_OVERWRITE',outputs=existing);return
    if worker_exists():emit('ABORT_EXISTING_WORKER_NO_DUPLICATE');return
    du=shutil.disk_usage('/diskE')
    if du.free<10*1024**3:emit('BLOCKED_DISK_MARGIN',free_bytes=du.free);return
    completed,qpartial=refresh_completed()
    snapshot={'observed_utc':now(),'gpu_rows':gpu(),'gpu_free_mib':{int(a.strip()):int(b.split()[0]) for a,b in (x.split(',',1) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader'],text=True).splitlines())},'diskE_free_bytes':du.free,'processes':procs(),'gpu2_extra_pid_352305_identity':'EigenAttn build_eigen_basis_llama_error_budget_offline_fdp125_cpu_stage_v1.py; not touched','manifest_file_sha256':sha(MAN),'dataset_sha256':sha(ds),'foreign_jobs_interrupted':0}
    update_pipe('ACTIVE_SERIAL_SINGLE_GPU1',snapshot=snapshot)
    emit('SERIAL_PAIR_READY',selected_gpu=1,free_mib=free_mib(1),already_completed=completed,qmsum_partial=qpartial,snapshot=snapshot)
    records={}
    for method in ('uniform','fdp_native'):
        raw=outs[method];sm=raw.with_suffix('.json')
        if raw.exists() or sm.exists():emit('ABORT_OUTPUT_APPEARED_NO_OVERWRITE',method=method);update_pipe('PAIR_INCOMPLETE_NO_OVERWRITE',records);return
        if worker_exists():emit('ABORT_WORKER_APPEARED_NO_DUPLICATE',method=method);update_pipe('PAIR_INCOMPLETE_WORKER_CONFLICT',records);return
        while free_mib(1)<MIN_FREE:
            update_pipe('READY_WAITING_FOR_SAFE_GPU1',records)
            emit('WAIT_GPU1_MARGIN',method=method,free_mib=free_mib(1),required_free_mib=MIN_FREE)
            time.sleep(60)
            if raw.exists() or sm.exists() or worker_exists():emit('ABORT_EXTERNAL_CONFLICT',method=method);return
        runid=f'RATIO0P5_{method.upper()}_MULTINEWS_GPU1_SERIAL_{datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
        log=LB/'logs'/f'{runid}.log';log.parent.mkdir(parents=True,exist_ok=True);lf=log.open('ab',buffering=0)
        proc=subprocess.Popen([PY,'-u',str(RUNNER),'--ratio','0.5','--method',method,'--task','multi_news','--gpu','1'],cwd=str(B),stdin=subprocess.DEVNULL,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
        active={'status':'ACTIVE','task':'multi_news','method':method,'ratio':0.5,'run_id':runid,'pid':proc.pid,'gpu':1,'started_at_utc':now(),'manifest_file_sha256':sha(MAN),'dataset_sha256':sha(ds),'q_sha':Q_SHA,'n_samples_expected':200,'raw_output':str(raw),'log':str(log),'formal_results_modified':False,'foreign_jobs_interrupted':0}
        records[method]=active
        update_pipe('ACTIVE_SERIAL_SINGLE_GPU1',records)
        (SIDE/f'{runid}_ACTIVE.json').write_text(json.dumps(active,indent=2,sort_keys=True)+'\n')
        emit('WORKER_STARTED',worker=active,free_mib=free_mib(1))
        while proc.poll() is None:time.sleep(60)
        rc=proc.returncode;lf.close();rec,ok=validate(method,rc,runid,log);records[method]=rec
        update_pipe('ACTIVE_SERIAL_SINGLE_GPU1' if ok else 'TODO_BLOCKED_INCOMPLETE_PRESERVED',records)
        (SIDE/f'{runid}_DONE.json' if ok else SIDE/f'{runid}_FAILED.json').write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
        emit('METHOD_COMPLETE' if ok else 'METHOD_FAILED_PRESERVED_PARTIAL',result=rec)
        if not ok:return
    u=records['uniform'];n=records['fdp_native'];delta=round(float(n['score'])-float(u['score']),6)
    update_pipe('COMPLETE',records)
    emit('PAIR_COMPLETE',uniform_score=u['score'],fdp_native_score=n['score'],delta_native_minus_uniform=delta,n_samples=200,resource={'native_rank':RANK,'formal_cache_bytes':CACHE},manifest_file_sha256=sha(MAN),q_sha=Q_SHA,uniform_schedule_sha=u['schedule_sha'],fdp_native_schedule_sha=n['schedule_sha'])
if __name__=='__main__':
    try:main()
    except Exception as e:
        emit('CONTROLLER_EXCEPTION',error=repr(e))
        try:update_pipe('TODO_BLOCKED_CONTROLLER_EXCEPTION')
        except Exception:pass
        raise

