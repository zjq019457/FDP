"""Final unchanged-allocation and basis binding gate, no inference or search."""
import sys,json,hashlib,datetime
from pathlib import Path
E=Path(sys.argv[1]);manifest=json.loads((E/'FORMAL_ACCURACY_MANIFEST.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
audit=Path('/diskE/jiaqizhang/MPT7B_FDP_EB_CAMPAIGN_V1/audits/repair_20261007T200635Z');cells=[];bases=[]
for c in manifest['cells']:
 census=json.loads(Path(c['runtime_census']).read_text())
 assert sha(c['allocation'])==census['allocation_sha']
 assert sha(E/'sources/canonical_ppl.py')==census['runner_sha']
 assert c['ranks']==[[r['rank_kq'],r['rank_v']] for r in census['per_layer']]
 cells.append({'tier':c['tier'],'arm':c['arm'],'allocation_unchanged':True,'measured_resource_binding':'PASS'})
for tier,tag,attempt in [('U011','0011',2),('U015','0015',1),('U019','0019',1)]:
 root=Path('/diskF/jiaqizhang/MPT7B_FDP_EB_CAMPAIGN_V1/resource_stages')/f'EB{tag}_UNIFORM_RESOURCE_ATTEMPT{attempt}'
 for li in range(32):
  p=root/f'layer_{li:02d}_resource_search.json';r=json.loads(p.read_text());bp=root/f'layer_{li:02d}_basis_and_uniform_rank.pt';actual=sha(bp)
  assert actual==r['basis_tensor_file_sha256']
  bases.append({'tier':tier,'layer':li,'basis':str(bp),'basis_sha256':actual,'basis_report':str(p),'basis_report_sha256':sha(p)})
gate={'status':'PASS','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':sha(E/'FORMAL_ACCURACY_MANIFEST.json'),'cells':cells,'basis_inputs':bases,'historical_quality_modified':False,'PPL_restarted':False,'external_comparisons':'WAIT_FOR_NEW_FORMAL_5TASK_ANCHOR'}
(E/'PRELAUNCH_BINDING.json').write_text(json.dumps(gate,indent=2)+'\n')
(E/'PRELAUNCH_BINDING.sha256').write_text(sha(E/'PRELAUNCH_BINDING.json')+'\n')
print('PRELAUNCH_PASS six unchanged schedules;96 basis files bound',flush=True)
