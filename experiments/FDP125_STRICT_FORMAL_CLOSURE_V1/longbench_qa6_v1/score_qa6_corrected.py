#!/home/bai-project/anaconda3/envs/eigen/bin/python
import argparse,hashlib,json,os,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def atomic(p,obj):
 t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n'); os.replace(t,p)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--pred_root',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--out_dir',required=True); a=ap.parse_args()
 sysroot=Path('/home/bai-project/jiaqizhang/EigenAttn/slim'); sys.path.insert(0,str(sysroot)); import longbench_eval as lbe
 mf=json.load(open(a.manifest,encoding='utf-8')); assert mf['LONGBENCH_PROTOCOL_AUDIT']=='PASS' and mf['launch_authorized'] is True
 tasks=mf['dataset']['task_order']; counts={t:int(mf['dataset']['task_metadata'][t]['sample_count']) for t in tasks}; total=sum(counts.values())
 result={'status':'PASS','protocol_status':'PASS','launch_authorized':True,'manifest_sha256':mf.get('corrected_manifest_payload_sha256',mf.get('manifest_sha256')),'manifest_file_sha256':sha(a.manifest),'task_order':tasks,'expected_total_samples':total,'task_counts':counts,'methods':{}}
 for method in ['official_uniform','fdp125_direct']:
  md=Path(a.pred_root)/method; scores={}; missing=[]
  for task in tasks:
   p=md/(task+'.jsonl'); rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []
   if len(rows)!=counts[task]: missing.append({'task':task,'expected':counts[task],'actual':len(rows),'path':str(p)}); continue
   score=lbe.scorer(task,[x.get('pred','') for x in rows],[x.get('answers') or [] for x in rows],[int(x.get('length') or 0) for x in rows],next((x.get('all_classes') for x in rows if x.get('all_classes') is not None),None))
   scores[task]={'score':float(score),'sample_count':len(rows),'prediction_sha256':sha(p)}
  mr={'tasks':scores,'sample_count':sum(x['sample_count'] for x in scores.values()),'expected_sample_count':total,'resource':mf['methods'][method]}
  if missing: result['status']='INCOMPLETE'; mr['missing']=missing
  else: mr['macro_exact']=sum(x['score'] for x in scores.values())/len(tasks); mr['macro']=round(mr['macro_exact'],2)
  result['methods'][method]=mr
 out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); atomic(out/'FDP125_LONGBENCH_QA6_SCORES_CORRECTED.json',result); print(json.dumps({'status':result['status'],'official_macro':result['methods']['official_uniform'].get('macro'),'fdp125_macro':result['methods']['fdp125_direct'].get('macro'),'expected_total_samples':total},ensure_ascii=False))
if __name__=='__main__': main()
