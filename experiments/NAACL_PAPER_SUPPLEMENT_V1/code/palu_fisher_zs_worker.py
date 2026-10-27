"""PaLu ratio-0.5 Fisher (official PaLu allocation, anchor schedule) six-task zero-shot with per-sample logs.
Identical to run_paired_zeroshot_worker.py (used for the V1/V3 paired run) except that the schedule is the
official Fisher anchor schedule loaded by the campaign's own load_schedule(0.5, 'fisher'); same factor cache,
same zero-shot script, tasks and settings. Output goes to a new directory; nothing existing is modified."""
import os, sys
from pathlib import Path
gpu = sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
base = Path('/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1')
out = Path('/diskF/jiaqizhang/NAACL_PAPER_SUPPLEMENT_V1/palu_fisher_zs')
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(base / 'code')); sys.path.insert(0, '/home/bai-project/jiaqizhang/ReCalKV')
import run_palu_formal_quality as q
import run_palu_formal_zeroshot as zs
# factor cache: campaign default (the 8-group override used for V1/V3 is shape-incompatible with the Fisher anchor schedule)
schedule, doc, selection = q.load_schedule(0.5, 'fisher')
q.load_schedule = lambda _ratio, _method: (schedule, doc, selection)
runtime = q.load_runtime(0.5, 'fisher')
q.load_runtime = lambda _ratio, _method: runtime
zs.formal.BRANCH = out
sys.argv = [str(zs.__file__), '--ratio', '0.5', '--method', 'fisher']
zs.main()
