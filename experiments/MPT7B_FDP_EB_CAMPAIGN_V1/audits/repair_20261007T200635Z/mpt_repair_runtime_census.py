"""Read frozen schedules and census the existing canonical MPT runtime only."""
import os, sys, json, hashlib, importlib.util, gc, traceback
from pathlib import Path
import torch

ROOT = Path('/diskE/jiaqizhang/MPT7B_FDP_EB_CAMPAIGN_V1')
OUT = Path(sys.argv[1])
SCRIPT = Path('/home/bai-project/jiaqizhang/EigenAttn/experimental/mpt7b_fdp_eb_pipeline_v1/run_u011_diagnostic_ppl.py')
def load(name, path):
    s = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(s)
    sys.modules[name] = m; s.loader.exec_module(m); return m
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
ppl = load('repair_canonical_ppl', SCRIPT)
protocol = load('repair_mpt_protocol', ROOT / 'adapters/mpt_fdp_protocol_v1.py')
torch.manual_seed(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
device = torch.device('cuda:0')
torch.cuda.set_per_process_memory_fraction(.40, device)
for tier in ['U011', 'U015', 'U019']:
    for arm in ['relaxed_uniform', 'fdp']:
        dst = OUT / (tier + '_' + arm + '_ACTUAL_CACHE.json')
        if dst.exists(): continue
        free, total = torch.cuda.mem_get_info()
        if free < 36 * 1024**3: raise RuntimeError('safe GPU slot lost before realization')
        try:
            model, rank_sha = ppl.realize(arm, device, ppl.TIER_PATHS[tier])
            lock = ppl.TIER_PATHS[tier]['relaxed_lock' if arm == 'relaxed_uniform' else 'fdp_lock']
            sched = json.loads(lock.read_text())['rank_schedule']
            # Forward actual compressed runtime; do not construct dummy cache tensors.
            ids = torch.full((1, 2048), 1, dtype=torch.long, device=device)
            with torch.inference_mode():
                result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, return_dict=True)
            rec = protocol.actual_cache_census(result.past_key_values, sched)
            assert rec['dtypes'] == ['torch.bfloat16']
            rec.update(tier=tier, arm=arm, predicted_b_live=protocol.predicted_cache_bytes(sched),
                       runner_sha=sha(SCRIPT), allocation_file=str(lock), allocation_sha=sha(lock),
                       runtime_rank_sha=rank_sha, peak_allocated_bytes=torch.cuda.max_memory_allocated())
            # Materialized projection weights are separately accounted, never in KV B_live.
            rec['projection_state_bytes'] = sum(
                p.numel()*p.element_size() for n,p in model.named_parameters()
                if '.attn.Wqkv.' in n or '.attn.out_proj.' in n)
            rec['basis_state_note'] = 'Basis loaded on CPU during canonical realization, not retained as runtime cache.'
            dst.write_text(json.dumps(rec, indent=2) + '\n')
            print(tier, arm, rec['b_live_bytes'], flush=True)
            del result, model, ids; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        except Exception:
            (OUT / (tier+'_'+arm+'_FAILURE.txt')).write_text(traceback.format_exc())
            raise
