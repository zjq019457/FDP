#!/usr/bin/env python3
import os,sys,argparse,pathlib,json,hashlib,datetime,traceback,fcntl,math,platform
P=argparse.ArgumentParser(); P.add_argument('--seed',type=int,required=True); P.add_argument('--gpu',type=int,required=True); A=P.parse_args()
if A.seed not in (0,1,2) or A.gpu not in (0,1,2): raise SystemExit('invalid seed/gpu')
os.environ['CUDA_VISIBLE_DEVICES']=str(A.gpu); os.environ['HF_HUB_OFFLINE']='1'; os.environ['WANDB_DISABLED']='true'; os.environ.setdefault('OMP_NUM_THREADS','8')
import torch
V3=pathlib.Path('/diskE/jiaqizhang/PALU_FDP_NATIVE_RESIDUAL_SEARCH_V3'); O=V3/'RANDOM_MATCHED_CONTROL_V1'; S=O/f'seed_{A.seed}'; RUN=O/'runs'/f'seed_{A.seed}'; RUN.mkdir(parents=True,exist_ok=True)
def read(p): return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def write_atomic(p,x):
 p=pathlib.Path(p); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8'); tmp.replace(p)
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
lockf=open(RUN/'worker.lock','a+')
try: fcntl.flock(lockf.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError: raise SystemExit('WORKER_LOCKED')
statusp=RUN/'STATUS.json'; resultp=RUN/'FORMAL_RESULT.json'; failedp=RUN/'FAILED.json'
if resultp.exists() or failedp.exists() or statusp.exists(): raise SystemExit('REFUSE_DUPLICATE_OR_UNKNOWN_PRIOR_STATE')
prot=read(O/'RANDOM_MATCHED_GENERATION_PROTOCOL.json'); protlock=read(O/'RANDOM_MATCHED_PROTOCOL_LOCK.json'); agg=read(O/'RANDOM_MATCHED_AGGREGATE_MANIFEST.json')
if sha_file(O/'RANDOM_MATCHED_GENERATION_PROTOCOL.json')!=protlock['protocol_sha256'] or protlock['quality_evaluation_started'] is not False: raise RuntimeError('frozen protocol hash/status mismatch')
if agg.get('status')!='ALL_SEEDS_RESOURCE_PASS_QUALITY_NOT_STARTED' or not agg.get('all_random_matched_resource_pass'): raise RuntimeError('aggregate resource gate is not PASS')
row=next(x for x in agg['seed_manifests'] if x['seed']==A.seed)
if sha_file(S/'ACTION_MANIFEST.json')!=row['action_manifest_sha256'] or sha_file(S/'final_schedule.json')!=row['schedule_sha256']: raise RuntimeError('seed manifest SHA mismatch')
action=read(S/'ACTION_MANIFEST.json'); schedule=read(S/'final_schedule.json')
if action.get('status')!='RANDOM_MATCHED_RESOURCE_PASS' or schedule.get('final_allocation_sha256')!=row['allocation_sha256']: raise RuntimeError('per-seed resource/allocation gate mismatch')
source=pathlib.Path(prot['source']['v1_native_schedule_path']); v1doc=read(source); start={k:[int(y) for y in v] for k,v in sorted(v1doc['selection'].items())}; rand={k:[int(y) for y in v] for k,v in sorted(schedule['selection'].items())}
if sha_file(source)!=prot['source']['v1_native_schedule_file_sha256']: raise RuntimeError('V1 schedule file changed')
formalpath=V3/'FORMAL_PPL'/'DONE.json'; formal=read(formalpath)
if sha_file(formalpath)!=prot['source']['v3_formal_done_sha256'] or formal.get('formal_manifest_sha256')!=prot['formal_evaluation']['same_formal_manifest_sha256']: raise RuntimeError('V3 formal binding changed')
if formal.get('status')!='V3_FORMAL_PPL_COMPLETE' or formal.get('resource_match')!='EXACT': raise RuntimeError('V3 formal is not an exact-resource completed baseline')
# Verify read-only sources and the frozen runner/evaluator identities.
v3runner=V3/'run_rs_v3.py'; ofrunner=pathlib.Path(prot['source']['formal_quality_runner_path'])
if sha_file(v3runner)!=prot['source']['v3_runner_sha256'] or sha_file(ofrunner)!=prot['source']['formal_quality_runner_sha256']: raise RuntimeError('frozen runner SHA mismatch')
write_atomic(statusp,{'status':'RUNNING','seed':A.seed,'physical_gpu':A.gpu,'pid':os.getpid(),'started_utc':now(),'protocol_sha256':protlock['protocol_sha256'],'action_manifest_sha256':row['action_manifest_sha256'],'foreign_jobs_interrupted':0})
try:
 sys.path.insert(0,str(V3)); import run_rs_v3 as v3
 engine,profile=v3.load_modules(); frozen,split,verified_start=v3.validate_frozen_inputs(engine)
 if {k:list(v) for k,v in verified_start.items()}!=start: raise RuntimeError('V3 verified V1 start differs from bound random-control source')
 model_parts=v3.load_runtime(engine,profile,start); model,tok=model_parts[0],model_parts[1]
 formal_module=__import__('run_official_formal_quality')
 device=next(model.parameters()).device; dtype=str(next(model.parameters()).dtype)
 torch.cuda.synchronize()
 local_v1=formal_module.windows(model,tok)
 v3_selection={k:[int(y) for y in vals] for k,vals in formal['v3'].get('selection',{}).items()} if formal['v3'].get('selection') else read(V3/'CALIBRATION_LOCK'/'V3_LOCK.json')['selection']
 v3_selection={k:[int(y) for y in vals] for k,vals in sorted(v3_selection.items())}
 v3.install_selection(profile,model_parts,start,v3_selection)
 local_v3=formal_module.windows(model,tok)
 v3.install_selection(profile,model_parts,v3_selection,rand)
 local_random=formal_module.windows(model,tok)
 torch.cuda.synchronize()
 def exact(a,b): return a.get('wiki')==b.get('wiki') and a.get('c4')==b.get('c4')
 def result_short(x): return {k:x[k] for k in ['J','wiki_nll','c4_nll','wiki_ppl','c4_ppl','wiki_chunks','c4_chunks','wiki_tokens','c4_tokens']}
 def deltas(x,y): return {'wiki':[float(b)-float(a) for a,b in zip(x['wiki'],y['wiki'])],'c4':[float(b)-float(a) for a,b in zip(x['c4'],y['c4'])]}
 def score_delta(d): return 0.5*(sum(d['wiki'])/len(d['wiki'])+sum(d['c4'])/len(d['c4']))
 def paired_ci(d,seed): return v3.bootstrap_ci(d,seed)
 vf1=formal['v1_native']; vf3=formal['v3']
 if len(local_random['wiki'])!=166 or len(local_random['c4'])!=256: raise RuntimeError('formal sample count differs from bound manifest')
 d_rv1=deltas(vf1,local_random); d_v3r=deltas(local_random,vf3)
 bs=2026092600+2*A.seed
 resource=v3.resource(rand)
 if resource!=prot['match_targets']['exact_final_resource']: raise RuntimeError('runtime resource census mismatch')
 gpuinfo={'visible_physical_gpu':A.gpu,'name':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),'torch_cuda_version':torch.version.cuda,'memory_total_bytes':torch.cuda.get_device_properties(0).total_memory}
 payload={
  'status':'RANDOM_MATCHED_FORMAL_COMPLETE','experiment':'PALU_V3_RANDOM_MATCHED_CONTROL_V1','seed':A.seed,'ratio':0.5,
  'protocol_path':str(O/'RANDOM_MATCHED_GENERATION_PROTOCOL.json'),'protocol_sha256':protlock['protocol_sha256'],
  'action_manifest_path':str(S/'ACTION_MANIFEST.json'),'action_manifest_sha256':row['action_manifest_sha256'],
  'action_sequence_sha256':action['action_sequence_sha256'],'allocation_sha256':action['final_allocation_sha256'],
  'formal_manifest_sha256':formal['formal_manifest_sha256'],'v3_formal_result_sha256':sha_file(formalpath),
  'v3_lock_sha256':sha_file(V3/'CALIBRATION_LOCK'/'V3_LOCK.json'),'v1_schedule_file_sha256':sha_file(source),
  'runner_sha256':sha_file(pathlib.Path(__file__)),'v3_runner_sha256':sha_file(v3runner),'official_quality_runner_sha256':sha_file(ofrunner),
  'model_path':frozen['model'],'model_parameter_dtype':dtype,'device':str(device),'gpu':gpuinfo,
  'runtime':{'python':platform.python_version(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,
    'allow_tf32_matmul':bool(torch.backends.cuda.matmul.allow_tf32),'cudnn_allow_tf32':bool(torch.backends.cudnn.allow_tf32)},
  'V1_native_replayed':result_short(local_v1),'V3_replayed_on_same_device':result_short(local_v3),'Random_i':result_short(local_random),
  'replay_vs_frozen':{'V1_exact':exact(local_v1,vf1),'V3_exact':exact(local_v3,vf3),
    'V1_max_abs_chunk_delta':{k:max(abs(float(a)-float(b)) for a,b in zip(local_v1[k],vf1[k])) for k in ['wiki','c4']},
    'V3_max_abs_chunk_delta':{k:max(abs(float(a)-float(b)) for a,b in zip(local_v3[k],vf3[k])) for k in ['wiki','c4']}},
  'paired_comparisons':{
    'Random_i_minus_V1_frozen':{'delta_J':score_delta(d_rv1),'ci95':paired_ci(d_rv1,bs),'bootstrap_seed':bs,'bootstrap_n':v3.BOOT_N},
    'V3_frozen_minus_Random_i':{'delta_J':score_delta(d_v3r),'ci95':paired_ci(d_v3r,bs+1),'bootstrap_seed':bs+1,'bootstrap_n':v3.BOOT_N}},
  'paired_chunk_deltas':{'Random_i_minus_V1_frozen':d_rv1,'V3_frozen_minus_Random_i':d_v3r},
  'resource_census':{'V1':v3.resource(start),'Random_i':resource,'V3':v3.resource(v3_selection),'status':'EXACT_MATCH'},
  'generation_match':{'action_count':action['action_count'],'transfer_volume_rank':action['movement_census']['transfer_volume_rank'],
    'final_l1_rank_vs_v1':action['movement_census']['final_l1_rank_vs_v1'],'factor_bytes_directional':action['movement_census']['directional_donor_factor_bytes_moved_fp16'],
    'resource_pass':action['legality_audit']['resource_pass']},
  'formal_quality_used_to_select':False,'foreign_jobs_interrupted':0,'started_utc':read(statusp)['started_utc'],'completed_utc':now()
 }
 # Validate sample-level alignments and all exact-resource checks before committing the result.
 if len(d_rv1['wiki'])!=len(vf1['wiki']) or len(d_v3r['c4'])!=len(vf3['c4']): raise RuntimeError('paired sample alignment failure')
 write_atomic(resultp,payload); payload['result_sha256']=sha_file(resultp); write_atomic(statusp,{**payload,'status':'DONE'})
 print(json.dumps({'status':'DONE','seed':A.seed,'gpu':A.gpu,'J':local_random['J'],'Wiki_PPL':local_random['wiki_ppl'],'C4_PPL':local_random['c4_ppl'],'Random_minus_V1':payload['paired_comparisons']['Random_i_minus_V1_frozen'],'V3_minus_Random':payload['paired_comparisons']['V3_frozen_minus_Random_i'],'V1_exact':payload['replay_vs_frozen']['V1_exact'],'V3_exact':payload['replay_vs_frozen']['V3_exact'],'result_sha256':payload['result_sha256'],'foreign_jobs_interrupted':0},indent=2),flush=True)
except BaseException as e:
 err={'status':'FAILED','failure_type':'RUNTIME_OR_ENGINEERING_FAILURE','reason':repr(e),'traceback':traceback.format_exc(),'old_v1_v3_results_still_valid':True,'seed':A.seed,'gpu':A.gpu,'protocol_sha256':protlock['protocol_sha256'],'foreign_jobs_interrupted':0,'failed_utc':now()}
 write_atomic(failedp,err); write_atomic(statusp,err); raise
