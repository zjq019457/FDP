import json, os, sys
from pathlib import Path
gpu, mode = sys.argv[1:3]
os.environ['CUDA_VISIBLE_DEVICES']=gpu
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
base=Path('/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1')
root=base/'quality/llama2/ratio_0.5/PALU_V3_POSTHOC_ZEROSHOT_V1'
v3=Path('/diskE/jiaqizhang/PALU_FDP_NATIVE_RESIDUAL_SEARCH_V3')
sys.path.insert(0,str(base/'code')); sys.path.insert(0,'/home/bai-project/jiaqizhang/ReCalKV')
import run_palu_formal_quality as q
import run_palu_formal_zeroshot as zs
m=json.loads((root/'PAIRED_ZEROSHOT_MANIFEST.json').read_text())
if mode=='v1':
 rec=m['methods']['v1_native_formal_reference']; schedule=Path(rec['schedule_path']); doc=json.loads(schedule.read_text()); doc['schedule_sha']=rec['schedule_semantic_sha256']; selection=doc['selection']; method_root=Path(rec['output_dir'])
elif mode=='v3':
 rec=m['methods']['v3_selected']; schedule=Path(rec['lock_path']); doc=json.loads(schedule.read_text()); doc['schedule_sha']=doc['allocation_sha256']; selection=doc['selection']; method_root=Path(rec['output_dir'])
else: raise ValueError(mode)
q.FACTOR_CACHE='/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1/cache/official_whiten_factors_rank512.pt'
q.load_schedule=lambda _ratio,_method:(schedule,doc,selection)
runtime=q.load_runtime(0.5,'fdp_native')
q.load_runtime=lambda _ratio,_method:runtime
zs.formal.BRANCH=method_root
sys.argv=[str(zs.__file__),'--ratio','0.5','--method','fdp_native']
zs.main()
