"""Post-evaluation paired task-stratified bootstrap and exact McNemar/Holm."""
import sys,json,hashlib,math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
E=Path(sys.argv[1]);manifest=json.loads((E/'FORMAL_ACCURACY_MANIFEST.json').read_text());F=Path(manifest['F_root'])
mh=hashlib.sha256((E/'FORMAL_ACCURACY_MANIFEST.json').read_bytes()).hexdigest();rows=[];tests=[]
for tier in ['U011','U015','U019']:
 pairs=[];scores={'relaxed_uniform':{},'fdp':{}}
 for task,contract in manifest['tasks'].items():
  paths=[F/'raw'/tier/arm/(task+'.json') for arm in ['relaxed_uniform','fdp']]
  if not all(p.exists() for p in paths):raise RuntimeError('formal output not complete')
  u,f=[json.loads(p.read_text()) for p in paths]
  assert u['manifest_sha256']==f['manifest_sha256']==mh
  assert [(r['doc_id'],r['doc_hash'],r['scoring_gold']) for r in u['samples']]==[(r['doc_id'],r['doc_hash'],r['scoring_gold']) for r in f['samples']]
  metric=contract['metric'];a=np.array([r[metric] for r in u['samples']]);b=np.array([r[metric] for r in f['samples']]);pairs.append(b-a)
  for arm,rec in [('relaxed_uniform',u),('fdp',f)]:scores[arm][task]=rec['primary_score']
  win=int(((a==0)&(b==1)).sum());loss=int(((a==1)&(b==0)).sum());n=win+loss
  tests.append({'tier':tier,'task':task,'metric':metric,'n':len(a),'fdp_wins':win,'fdp_losses':loss,'p':float(binomtest(win,n,.5).pvalue) if n else 1.})
 rng=np.random.default_rng(manifest['statistics']['seed']);boot=[]
 for start in range(0,10000,32):
  n=min(32,10000-start);d=np.zeros(n)
  for x in pairs:d+=x[rng.integers(0,len(x),(n,len(x)))].mean(1)/5
  boot.extend(d.tolist())
 rows.append({'tier':tier,'uniform_macro':float(np.mean(list(scores['relaxed_uniform'].values()))),'fdp_macro':float(np.mean(list(scores['fdp'].values()))),'delta_pp':float(np.mean([d.mean() for d in pairs]))*100,'paired_95_ci_pp':(np.quantile(boot,[.025,.975])*100).tolist(),'per_task':scores})
order=sorted(range(len(tests)),key=lambda i:tests[i]['p']);previous=0.
for j,i in enumerate(order):
 previous=max(previous,min(1.,tests[i]['p']*(len(tests)-j)));tests[i]['p_holm_15']=previous
result={'status':'FORMAL_5TASK_COMPLETE','manifest_sha256':mh,'metric_contract':{t:v['metric'] for t,v in manifest['tasks'].items()},'statistics':manifest['statistics'],'bands':rows,'McNemar':tests,'raw_output_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (F/'raw').glob('*/*/*.json')},'external_anchor_status':'FROZEN_ONLY_FOR_THESE_MPT_SCHEDULES_AND_THIS_SUITE'}
(E/'FORMAL_5TASK_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
(E/'FORMAL_5TASK_RESULTS.sha256').write_text(hashlib.sha256((E/'FORMAL_5TASK_RESULTS.json').read_bytes()).hexdigest()+'\n')
print(json.dumps(rows,indent=2),flush=True)
