#!/usr/bin/env python3
import torch, json, hashlib, os
from pathlib import Path
F=Path('/diskF/jiaqizhang/fdp125_tacl_llama2_exact_3comp_pass1_v1_20260930')
root=F/'llama2_e060_full_capacity_basis_only_v1'
full_path=root/'llama2_e060_full_basis_capacity_only_eb0p06.pt'
report_path=F/'eb070_eb090_shared_decomp/llama2_7b_uniform_pass1_eb070_eb090_report.json'
old_selected_path=F/'llama2_e060_fullbasis_uniform_same_rank_audit_v1/llama2_e060_fullbasis_uniform_selected_rank.pt'
out=Path('/dev/shm/llama2_uniform_e059_attempt6/selected_basis.pt')
expected_rank_sha='3f4e59be8cf981abb5c6c272702eb42e75729f9167677e29846f957e7935eaf4'
full=torch.load(full_path,map_location='cpu',weights_only=False)
report=json.load(open(report_path))
old_selected=torch.load(old_selected_path,map_location='cpu',weights_only=False)
# Verify this exact selection transform against the existing Uniform-.060 selected basis.
old_ranks=json.load(open(F/'llama2_e060_fullbasis_uniform_same_rank_audit_v1/rank_binding.json'))['layers']
verify_max=0.0; verify_count=0
for row in old_ranks:
    l=row['layer']; ck=row['rank_kq']//32; cv=row['rank_v']//32
    pre=f'base_model.model.model.layers.{l}.self_attn.'
    for side in ('q_proj_basis','k_proj_basis'):
        key=pre+side; x=full[key][:,:ck]; y=old_selected[key]
        if x.shape!=y.shape: raise RuntimeError(f'old QK shape mismatch {key} {x.shape} {y.shape}')
        d=(x-y).abs().max().item(); verify_max=max(verify_max,d); verify_count+=1
        if d!=0: raise RuntimeError(f'old QK tensor mismatch {key} max_abs={d}')
    key=pre+'v_proj_basis'
    idx=torch.cat([torch.arange(h*128,h*128+cv) for h in range(32)])
    x=full[key].index_select(1,idx); y=old_selected[key]
    if x.shape!=y.shape: raise RuntimeError(f'old V shape mismatch {key} {x.shape} {y.shape}')
    d=(x-y).abs().max().item(); verify_max=max(verify_max,d); verify_count+=1
    if d!=0: raise RuntimeError(f'old V tensor mismatch {key} max_abs={d}')
# Derive EB=.059 ranks from the frozen Uniform threshold trace. Do not inspect quality.
ranks=[]; selected=[]
for layer in report['layers']:
    run=next((r for r in layer['runs'] if abs(float(r['error_budget'])-.07)<1e-12),None)
    if run is None: raise RuntimeError(f"layer {layer['layer']} missing .07 trace")
    passed=[p for p in run['threshold_trace'] if float(p['err'])<=.059]
    if not passed: raise RuntimeError(f"layer {layer['layer']} has no passing rank")
    p=passed[-1]
    pair=[int(p['rank_kq']),int(p['rank_v'])]
    ranks.append(pair)
    selected.append({'layer':int(layer['layer']),'threshold':float(p['threshold']),'error':float(p['err']),'rank_kq':pair[0],'rank_v':pair[1]})
rank_sha=hashlib.sha256(json.dumps(ranks,separators=(',',':')).encode()).hexdigest()
if rank_sha!=expected_rank_sha: raise RuntimeError(f'rank SHA mismatch {rank_sha}')
if len(ranks)!=32 or any(k%32 or v%32 for k,v in ranks): raise RuntimeError('rank lattice violation')
# Reuse the verified canonical layout: Q/K prefix; V columns gathered per head.
selected_basis={}
for l,(rk,rv) in enumerate(ranks):
    ck=rk//32; cv=rv//32; pre=f'base_model.model.model.layers.{l}.self_attn.'
    for side in ('q_proj_basis','k_proj_basis'):
        key=pre+side; selected_basis[key]=full[key][:,:ck].contiguous()
    key=pre+'v_proj_basis'
    idx=torch.cat([torch.arange(h*128,h*128+cv) for h in range(32)])
    selected_basis[key]=full[key].index_select(1,idx).contiguous()
torch.save(selected_basis,out)
b_live=sum(sum(x) for x in ranks)*4096
if b_live!=789577728: raise RuntimeError(f'B_live mismatch {b_live}')
meta={'status':'SELECTED_BASIS_BUILT_FROM_FROZEN_EB059_SCHEDULE','rank_sha256':rank_sha,'rank_sum':sum(sum(x) for x in ranks),'B_live_bytes':b_live,'verification_against_existing_EB060_selected_basis':{'tensor_count':verify_count,'mismatch_count':0,'max_abs':verify_max},'full_basis_sha256':__import__('subprocess').check_output(['sha256sum',str(full_path)],text=True).split()[0],'source_report_sha256':__import__('subprocess').check_output(['sha256sum',str(report_path)],text=True).split()[0],'source_report':str(report_path),'selection_rule':'for each layer, last threshold-trace row with error <= 0.059 from frozen Uniform .07/.09 decomposition report','selected_rows':selected,'basis_output':str(out)}
mp=Path('/diskF/jiaqizhang/fdp125_tacl_llama2_exact_3comp_pass1_v1_20260930/LLAMA2_UNIFORM_EB059_STRICT_MATCH_V1/attempt6/basis_manifest.json')
mp.write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
