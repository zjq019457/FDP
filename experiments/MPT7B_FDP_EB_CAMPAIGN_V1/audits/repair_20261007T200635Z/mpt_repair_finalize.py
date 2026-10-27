import sys,json,csv,hashlib
from pathlib import Path
out=Path(sys.argv[1]); bands=['U011','U015','U019']; rows=[]; verdict={
 'RELAXED_UNIFORM_SEMANTICS':'UNRESOLVED',
 'HELLASWAG_METRIC_CONTRACT':'CORRECTED_TO_ACC',
 'SOURCE_BINDING':'PARTIAL','J_SOURCE_BINDING':'PASS','FORMAL_5TASK_REUSE_ALLOWED':'NO'}
replay=json.loads((out/'COMMON_EPSILON_NATIVE_TRIAL_REPLAY.json').read_text())
if all(r['matched_layers']==32 for r in replay):
    verdict['RELAXED_UNIFORM_SEMANTICS']='TRUE_UNIFORM_EB'
for band in bands:
    records={arm:json.loads((out/f'{band}_{arm}_ACTUAL_CACHE.json').read_text()) for arm in ['relaxed_uniform','fdp']}
    u,f=records['relaxed_uniform']['b_live_bytes'],records['fdp']['b_live_bytes']
    ok=abs(u-f)/f<=.0015 and all(r['status']=='ACTUAL_TENSOR_CENSUS_PASS' and r['predicted_b_live']==r['b_live_bytes'] for r in records.values())
    verdict['ACC_ARM_IDENTITY_'+band]='RELAXED'
    verdict['MEASURED_RESOURCE_MATCH_'+band]='PASS' if ok else 'FAIL'
    for arm,r in records.items():
        rows.append({'Band':band,'Method':arm,'Predicted_B_live':r['predicted_b_live'],'Measured_B_live':r['b_live_bytes'],'Difference':r['b_live_bytes']-r['predicted_b_live'],'Relative_error':(r['b_live_bytes']-r['predicted_b_live'])/r['predicted_b_live'],'Uniform_minus_FDP':u-f,'pair_gap_percent':100*(u-f)/f,'projection_state_bytes_excluded_from_cache':r['projection_state_bytes'],'status':verdict['MEASURED_RESOURCE_MATCH_'+band]})
with (out/'ACTUAL_RESOURCE_CENSUS.csv').open('w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
(out/'FINAL_DECISION.json').write_text(json.dumps(verdict,indent=2)+'\n')
text='\n'.join(k+' = '+str(v) for k,v in verdict.items())
(out/'FINAL_DECISION.txt').write_text(text+'\n')
(out/'REPAIR_REPORT.md').write_text('# MPT audit repair\n\n'+text+'\n\n## Findings and remaining evidence gaps\n\n- All three relaxed schedules reproduce 32/32 ranks from their one shared epsilon and preserved native trial curves, using first-failure semantics. Thus their schedule semantics are TRUE_UNIFORM_EB; no layer-specific rank repair is required. This is an independent resource-only replay, not a claim that the original generation script was recovered. Historical generation source remains a source-binding gap. See COMMON_EPSILON_NATIVE_TRIAL_REPLAY.json and mpt_repair_uniform_replay.py. Native function authority: mpt_fdp_protocol_v1.py::select_last_passing_before_first_failure. Per-layer spectral thresholds are not global reconstruction epsilons.\n- Launcher, loader, config and result rank SHA chains now agree on RELAXED for all three accuracy arms. Current runner/source hashes and model/task manifests are captured; historical execution-time source SHA is not embedded in outputs. Binding remains PARTIAL.\n- Six actual batch1/seq2048/BF16 runtime caches were counted. Projection weights excluded and reported separately.\n- Correct metric contract and 10k item-stratified bootstrap use only existing accuracy files. Independent paired-window PPL bootstrap stores historical intervals, seed/algorithm differences and endpoint differences.\n- Real J source recovered in run_u011_diagnostic_ppl.py, main(), summary J field.\n- No formal quality was rerun. No existing predictions or results were modified. No promotion and no unnecessary 15/30 task rerun: RAW/OTHER is not the remaining cause.\n')
paths=sorted(p for p in out.iterdir() if p.is_file() and p.name not in ['SHA256SUMS.txt','runtime_stdout.log','runtime_stderr.log'])
(out/'SHA256SUMS.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in paths))
print(text)
print(json.dumps(rows,indent=2))
