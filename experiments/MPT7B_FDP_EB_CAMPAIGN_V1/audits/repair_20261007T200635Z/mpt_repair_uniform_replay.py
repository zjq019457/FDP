"""Resource-only replay of common-epsilon schedules from preserved native trials."""
import sys,json,importlib.util,hashlib
from pathlib import Path
out=Path(sys.argv[1]);root=Path('/diskE/jiaqizhang/MPT7B_FDP_EB_CAMPAIGN_V1')
code=Path('/home/bai-project/jiaqizhang/EigenAttn/experimental/mpt7b_fdp_eb_pipeline_v1/native_resource_geometry_audit.py')
s=importlib.util.spec_from_file_location('geometry',code);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
records=[]
for tier,tag,attempt in [('U011','0011',2),('U015','0015',1),('U019','0019',1)]:
    lock=json.loads((root/tier/'10_RELAXED_UNIFORM/RELAXED_UNIFORM_RANK_LOCK.json').read_text());epsilon=lock['relaxed_uniform_epsilon'];rows=[]
    for layer in range(32):
        trials=m.trials_for(tag,tier,attempt,layer);selected=None
        for t in trials:
            if t['relative_error']>epsilon:break
            selected=t
        rank=None if selected is None else [selected['rank_kq'],selected['rank_v']]
        rows.append({'layer':layer,'epsilon':epsilon,'replayed_rank':rank,'stored_rank':lock['rank_schedule'][layer],'match':rank==lock['rank_schedule'][layer],'candidate_source':'preserved native threshold/error trials; first-failure stopping'})
    records.append({'band':tier,'epsilon':epsilon,'matched_layers':sum(r['match'] for r in rows),'per_layer':rows,'generation_code_recovered':False,'note':'Replay is independently computed; it does not prove the original historical generation procedure.'})
(out/'COMMON_EPSILON_NATIVE_TRIAL_REPLAY.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ['band','epsilon','matched_layers']} for r in records],indent=2))
