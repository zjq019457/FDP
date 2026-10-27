import json, hashlib, sys, subprocess
from pathlib import Path
import numpy as np
ROOT=Path('/diskE/jiaqizhang/MPT7B_FDP_EB_CAMPAIGN_V1')
OUT=Path(sys.argv[1])
CODE=Path('/home/bai-project/jiaqizhang/EigenAttn/experimental/mpt7b_fdp_eb_pipeline_v1')
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ranksha(s): return hashlib.sha256(json.dumps({'rank_kq_rank_v_by_layer':s},sort_keys=True,separators=(',',':')).encode()).hexdigest()
tasks={'arc_challenge_localjsonl':'acc_norm','arc_easy_localjsonl':'acc_norm','hellaswag_localjsonl':'acc','piqa_localjsonl':'acc','winogrande_localjsonl':'acc'}
contract={'seed':20261007,'replicates':10000,'rng':'NumPy PCG64','accuracy':'batch32; task order ARC-C,ARC-E,Hella,PIQA,Wino; paired within-task items; equal-task macro','ppl':'batch128; Wiki draws then C4 draws in each batch; equal dataset mean NLL','metric_contract':tasks}
(OUT/'REPAIR_STATISTICS_PROTOCOL.json').write_text(json.dumps(contract,indent=2)+'\n')
source={str(p):sha(p) for p in CODE.glob('*.py')}
source[str(CODE/'run_mpt_acc_seq.sh')]=sha(CODE/'run_mpt_acc_seq.sh')
(OUT/'CURRENT_SOURCE_HASHES.json').write_text(json.dumps(source,indent=2)+'\n')
rows=[]
for tier in ['U011','U015','U019']:
    base=ROOT/tier; lockpath=base/'10_RELAXED_UNIFORM/RELAXED_UNIFORM_RANK_LOCK.json'; lock=read(lockpath)
    s=lock['rank_schedule']; rs=ranksha(s)
    acc=[]; files=[]
    for t,m in tasks.items():
        up=base/'11_ACCURACY/relaxed_uniform'/f'{t}.json'; fp=base/'11_ACCURACY/fdp'/f'{t}.json'
        u,f=read(up),read(fp)
        assert u['rank_schedule_sha256']==rs
        assert u['primary_metric']==f['primary_metric']==m
        assert [x['doc_id'] for x in u['samples']]==[x['doc_id'] for x in f['samples']]
        a=np.array([x[m] for x in u['samples']],float); b=np.array([x[m] for x in f['samples']],float)
        assert abs(a.mean()-u['primary_score'])<1e-12
        assert abs(b.mean()-f['primary_score'])<1e-12
        acc.append((a,b)); files.extend([{'path':str(p),'sha':sha(p)} for p in [up,fp]])
    rng=np.random.default_rng(contract['seed']); boot=[]
    for start in range(0,10000,32):
        n=min(32,10000-start); z=np.zeros(n)
        for a,b in acc:
            d=b-a; z+=d[rng.integers(0,len(d),(n,len(d)))].mean(1)/5
        boot.extend(z.tolist())
    point=float(np.mean([np.mean(b-a) for a,b in acc]))
    pair=[]
    for ds in ['wiki','c4']:
        uf=sorted((base/'10_RELAXED_UNIFORM'/ds).glob('*.json'))
        ff=sorted((base/'09_DIAGNOSTIC_QUALITY/fdp'/ds).glob('*.json'))
        assert len(uf)==len(ff)
        dv=[]
        for up,fp in zip(uf,ff):
            u,f=read(up),read(fp)
            assert (u['window_id'],u['input_ids_sha256'],u['token_count'])==(f['window_id'],f['input_ids_sha256'],f['token_count'])
            dv.append(f['mean_nll']-u['mean_nll']);files.extend([{'path':str(p),'sha':sha(p)} for p in [up,fp]])
        pair.append(np.array(dv))
    rng=np.random.default_rng(contract['seed']); pb=[]
    for start in range(0,10000,128):
        n=min(128,10000-start); z=np.zeros(n)
        for d in pair: z+=.5*d[rng.integers(0,len(d),(n,len(d)))].mean(1)
        pb.extend(z.tolist())
    history=read(base/'10_RELAXED_UNIFORM/PAIRED_BOOTSTRAP.json')
    row={'band':tier,'ACC_ARM_IDENTITY':'RELAXED','ACC_UNIFORM_CONFIG_SHA':sha(lockpath),'ACC_UNIFORM_RANK_SHA':rs,'ACC_UNIFORM_RANK_SUM':sum(map(sum,s)),'ACC_UNIFORM_THEORETICAL_BYTES':sum(map(sum,s))*4096,'epsilon':lock['relaxed_uniform_epsilon'],'runner_chain':{'launcher':str(CODE/'run_mpt_acc_seq.sh'),'accuracy':str(CODE/'run_mpt_diagnostic_acc.py'),'materializer':str(CODE/'run_u011_diagnostic_ppl.py'),'config':str(lockpath)},'arm_identity_evidence':'launcher --arm relaxed_uniform; accuracy calls ppl.realize(args.arm); realize loads relaxed_lock; all five output rank SHAs exact match','accuracy_delta_pp':point*100,'accuracy_ci_pp':(np.quantile(boot,[.025,.975])*100).tolist(),'ppl_delta_J':float(np.mean([d.mean() for d in pair])),'ppl_ci':np.quantile(pb,[.025,.975]).tolist(),'historical_bootstrap':history,'bootstrap_reconciliation':'Frozen new seed and batching differ from historical seed2; no interval equality forced.','relaxed_generation':'UNRESOLVED: declaration and common thresholds present, historical generation source not recovered','source_binding':'PARTIAL: current source hashes resolved, historical execution source SHA/git commit absent from result files','result_hashes':files}
    row['historical_J_ci']=history['delta_J']['ci95']
    row['J_ci_endpoint_difference']=(np.array(row['ppl_ci'])-np.array(row['historical_J_ci'])).tolist()
    row['current_base_git_commit']=subprocess.check_output(['git','-C','/home/bai-project/jiaqizhang/EigenAttn/EigenAttn_base','rev-parse','HEAD'],text=True).strip()
    tag={'U011':'0011','U015':'0015','U019':'0019'}[tier]
    attempt=2 if tier=='U011' else 1
    rawpath=Path('/diskF/jiaqizhang/MPT7B_FDP_EB_CAMPAIGN_V1/resource_stages')/f'EB{tag}_UNIFORM_RESOURCE_ATTEMPT{attempt}'/f'MPT7B_UNIFORM_EB{tag}_STRICT_RESOURCE_SCHEDULE_V1.json'
    raw=read(rawpath)
    row['raw_uniform_comparison']={'config':str(rawpath),'config_sha':sha(rawpath),'rank_sha':ranksha(raw['rank_schedule']),'rank_sum':sum(map(sum,raw['rank_schedule'])),'predicted_bytes':sum(map(sum,raw['rank_schedule']))*4096,'same_as_accuracy_schedule':raw['rank_schedule']==s}
    row['input_binding_manifest']={'path':str(ROOT/'input_bindings/MPT7B_EVAL_INPUT_BINDING_V1.json'),'sha':sha(ROOT/'input_bindings/MPT7B_EVAL_INPUT_BINDING_V1.json')}
    row['checkpoint_manifest']={'path':str(ROOT/'identity/MPT7B_CHECKPOINT_CONTENT_MANIFEST_V1.json'),'sha':sha(ROOT/'identity/MPT7B_CHECKPOINT_CONTENT_MANIFEST_V1.json')}
    row['task_yaml_current_shas']={t:sha(Path('/diskF/jiaqizhang/a1804298693/offline_datasets')/(t+'.yaml')) for t in tasks if (Path('/diskF/jiaqizhang/a1804298693/offline_datasets')/(t+'.yaml')).exists()}
    row['per_layer_native_spectral_threshold']=[x.get('threshold') for x in lock.get('per_layer',[])]
    row['per_layer_rank_schedule']=s
    row['post_search_layer_specific_rank_repair']='UNRESOLVED_NO_GENERATION_SOURCE'
    replay_path=OUT/'COMMON_EPSILON_NATIVE_TRIAL_REPLAY.json'
    if replay_path.exists():
        replay=next(x for x in read(replay_path) if x['band']==tier)
        row['relaxed_generation']='Original source unrecovered; independent shared-epsilon native-trial replay matches 32/32 layers.'
        row['schedule_semantics']='TRUE_UNIFORM_EB' if replay['matched_layers']==32 else 'UNRESOLVED'
        row['post_search_layer_specific_rank_repair']='None required for exact replay; original execution history unresolved.'
    row['per_result_source_binding']=[dict(x,runner_sha=source[str(CODE/('run_mpt_diagnostic_acc.py' if '/11_ACCURACY/' in x['path'] else 'run_u011_diagnostic_ppl.py'))],binding_status='PARTIAL_HISTORICAL_EXECUTION_SHA_NOT_RECORDED') for x in files]
    rows.append(row)
(OUT/'REPAIR_EXISTING_OUTPUTS.json').write_text(json.dumps({'contract':contract,'bands':rows,'J_SOURCE_BINDING':'PASS','J_source':str(CODE/'run_u011_diagnostic_ppl.py')+': main(), summary J = 0.5*(wiki[log_ppl]+c4[log_ppl])','formal_reuse_allowed':False},indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ['band','ACC_ARM_IDENTITY','ACC_UNIFORM_RANK_SUM','ACC_UNIFORM_THEORETICAL_BYTES','accuracy_delta_pp','accuracy_ci_pp','ppl_delta_J','ppl_ci']} for r in rows],indent=2),flush=True)
