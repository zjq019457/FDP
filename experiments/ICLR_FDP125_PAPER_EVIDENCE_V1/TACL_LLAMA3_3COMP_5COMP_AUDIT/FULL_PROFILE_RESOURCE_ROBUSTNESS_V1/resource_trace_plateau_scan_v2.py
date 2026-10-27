#!/home/bai-project/anaconda3/envs/eigen/bin/python
import csv,hashlib,json,math
from pathlib import Path
E=Path('/diskE/jiaqizhang/ICLR_FDP125_PAPER_EVIDENCE_V1/TACL_LLAMA3_3COMP_5COMP_AUDIT/FULL_PROFILE_RESOURCE_ROBUSTNESS_V1')
F=Path('/diskF/jiaqizhang/llama3_full_resource_robustness_v1')
LOCK=json.loads((E/'FULL_PROFILE_RESOURCE_ROBUSTNESS_PROTOCOL_LOCK_V1.1.json').read_text())
def sha_obj(x): return hashlib.sha256(json.dumps(x,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def eps_for(base,q):
 raw=[float(base)/(1+1.2*float(x)) for x in q]; mean=sum(raw)/len(raw)
 eps=[v*float(base)/mean for v in raw]
 eps[0]=math.exp(.75*math.log(float(base))+.25*math.log(eps[0]))
 eps[0]=float(base)
 return eps
def ranks_for(base,q,traces):
 eps=eps_for(base,q); ranks=[]
 for i,tr in enumerate(traces):
  best=[1024,1024]
  for row in tr:
   if float(row['err'])<=eps[i] and float(row['threshold'])>=.30: best=[int(row['rank_kq']),int(row['rank_v'])]
   else: break
  ranks.append(best)
 return ranks
def b_live(r): return sum(k+v for k,v in r)*4096
def run(tier):
 cfg=LOCK['tier_inputs'][tier]
 out=F/'envelope'/tier.lower()
 rp=out/f"llama3_{tier.lower()}_resource_envelope_perlayer_eb_report.json"
 bp=out/f"llama3_{tier.lower()}_resource_envelope_perlayer_eb.pt"
 if not rp.is_file() or not bp.is_file(): raise FileNotFoundError('envelope artifacts incomplete')
 rep=json.loads(rp.read_text()); traces=[x['runs'][0]['threshold_trace'] for x in rep['layers']]
 if len(traces)!=32 or any(not t for t in traces): raise RuntimeError('trace layer count/empties')
 start=float(cfg['scalar_original']); end=float(cfg['envelope_scalar']); target=int(cfg['target_b_live_bytes']); q=cfg['q_minmax']
 provp=F.parent/'fdp_component_subset_search_v2'/'results'/tier.lower()/'11001'/'provenance.json'
 prov=json.loads(provp.read_text())
 anchor=ranks_for(start,q,traces)
 if anchor!=prov['final_rank_vector'] or sha_obj(anchor)!=prov['final_rank_sha'] or b_live(anchor)!=prov['b_live']:
  raise RuntimeError('envelope trace failed exact original-Full anchor replay')
 fac=eps_for(1.0,q); events={start,end}
 for i,tr in enumerate(traces):
  for row in tr:
   x=float(row['err'])/fac[i]
   if start<x<end: events.add(x)
 events=sorted(events); segs=[]
 for lo,hi in zip(events[:-1],events[1:]):
  mid=(lo+hi)/2; ranks=ranks_for(mid,q,traces); sig=tuple(tuple(x) for x in ranks)
  if segs and segs[-1]['sig']==sig and abs(segs[-1]['hi']-lo)<1e-12:
   segs[-1]['hi']=hi; segs[-1]['mid']=(segs[-1]['lo']+hi)/2
  else: segs.append({'lo':lo,'hi':hi,'mid':mid,'sig':sig,'ranks':ranks,'B':b_live(ranks),'sha':sha_obj(ranks)})
 for x in (start,end):
  ranks=ranks_for(x,q,traces); sig=tuple(tuple(a) for a in ranks)
  if not any(s['sig']==sig for s in segs): segs.append({'lo':x,'hi':x,'mid':x,'sig':sig,'ranks':ranks,'B':b_live(ranks),'sha':sha_obj(ranks)})
 segs.sort(key=lambda s:(s['mid'],s['sha']))
 merged=[]
 for s in segs:
  if merged and merged[-1]['sig']==s['sig'] and s['lo']<=merged[-1]['hi']+1e-12:
   merged[-1]['hi']=max(merged[-1]['hi'],s['hi']); merged[-1]['mid']=(merged[-1]['lo']+merged[-1]['hi'])/2
  else: merged.append(s)
 if any(merged[i+1]['B']>merged[i]['B'] for i in range(len(merged)-1)): raise RuntimeError('B_live not monotone nonincreasing')
 above=[s for s in merged if s['B']>=target]; below=[s for s in merged if s['B']<=target]
 adjacent=[]; selected=None; state='NEEDS_ENVELOPE_EXTENSION'
 if above and below:
  a=max(above,key=lambda s:(s['B'],-s['mid'])); b=max(below,key=lambda s:(s['B'],-s['mid']))
  adjacent=[a] if a['sig']==b['sig'] else [a,b]
  def key(s): return (abs(s['B']-target),0 if s['B']>=target else 1,abs(s['mid']-start),s['sha'])
  selected=min(adjacent,key=key)
  gap=selected['B']-target; pct=100*gap/target
  state='RESOURCE_CANDIDATE_PASS' if abs(pct)<=.15 else 'RESOURCE_TARGET_OUTSIDE_0.15_PERCENT'
 payload={'schema':'LLAMA3_FULL_PROFILE_RESOURCE_PLATEAU_SCAN_V1','tier':tier,'status':state,'profile_sha256':cfg['profile_sha256'],'q_sha256':cfg['q_sha256'],'original_scalar':start,'envelope_scalar':end,'target_B_live_bytes':target,'envelope_report_path':str(rp),'envelope_report_sha256':sha_file(rp),'envelope_basis_sha256':sha_file(bp),'original_Full_anchor':{'exact_rank_match':True,'rank_sha256':sha_obj(anchor),'B_live_bytes':b_live(anchor)},'plateaus':[],'adjacent_candidates':[],'selected':None}
 for s in merged:
  payload['plateaus'].append({'scalar_interval':[s['lo'],s['hi']],'scalar_midpoint':s['mid'],'B_live_bytes':s['B'],'gap_bytes':s['B']-target,'gap_pct_vs_target':100*(s['B']-target)/target,'rank_sha256':s['sha'],'ranks':s['ranks']})
 for s in adjacent:
  payload['adjacent_candidates'].append({'scalar_interval':[s['lo'],s['hi']],'scalar_midpoint':s['mid'],'B_live_bytes':s['B'],'gap_bytes':s['B']-target,'gap_pct_vs_target':100*(s['B']-target)/target,'rank_sha256':s['sha'],'ranks':s['ranks']})
 if selected:
  gap=selected['B']-target; pct=100*gap/target; eps=eps_for(selected['mid'],q)
  ep=F/tier.lower()/f'{tier.lower()}_selected_epsilon.json'
  ep.write_text(json.dumps({'error_budget_by_layer':eps,'source_profile_sha256':cfg['profile_sha256'],'q_sha256':cfg['q_sha256'],'base_scalar':selected['mid'],'lambda':1.2,'rho':.75,'layer0_freeze':True},indent=2))
  candidate={'schema':'LLAMA3_FULL_PROFILE_RESOURCE_SCHEDULE_CANDIDATE_V1','tier':tier,'status':'PENDING_FINAL_OFFICIAL_BUILDER_REPLAY','arm_name':f'FULL_RESOURCE_{tier}_LOW','paper_label':'Full profile, resource-controlled','target_B_live_bytes':target,'candidate_B_live_bytes':selected['B'],'gap_bytes':gap,'gap_pct_vs_target':pct,'base_scalar':selected['mid'],'scalar_plateau':[selected['lo'],selected['hi']],'rank_vector':selected['ranks'],'rank_sha256':selected['sha'],'q_sha256':cfg['q_sha256'],'profile_sha256':cfg['profile_sha256'],'epsilon_input_path':str(ep),'epsilon_input_sha256':sha_file(ep),'quality_blind':True}
  payload['selected']=candidate
  (E/f'{tier.lower()}_selected_resource_schedule_candidate.json').write_text(json.dumps(candidate,indent=2))
 jp=E/f'{tier.lower()}_resource_plateau_scan.json'; jp.write_text(json.dumps(payload,indent=2))
 cp=E/f'{tier.lower()}_resource_plateau_scan.csv'
 with cp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['scalar_lo','scalar_hi','scalar_mid','rank_sha256','B_live_bytes','gap_bytes','gap_pct_vs_target','adjacent'])
  w.writeheader()
  # Plateau objects use `sha`; rank_sha256 is only the serialized field name.
  adj={x['sha'] for x in adjacent}
  for s in merged: w.writerow({'scalar_lo':s['lo'],'scalar_hi':s['hi'],'scalar_mid':s['mid'],'rank_sha256':s['sha'],'B_live_bytes':s['B'],'gap_bytes':s['B']-target,'gap_pct_vs_target':100*(s['B']-target)/target,'adjacent':s['sha'] in adj})
 print(tier,state,'plateaus',len(merged),'selected',json.dumps(payload['selected'],separators=(',',':')))
 print('SHA',sha_file(jp),sha_file(cp))
if __name__=='__main__':
 for t in ('MEDIUM','AGGRESSIVE'):
  rc=E/f'{t.lower()}_envelope_builder.rc'
  if not rc.exists() or rc.read_text().strip()!='0': raise SystemExit(t+' envelope not PASS')
 run('MEDIUM'); run('AGGRESSIVE')
