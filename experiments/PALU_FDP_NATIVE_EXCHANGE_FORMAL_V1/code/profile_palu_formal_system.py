import argparse
import json
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import run_palu_formal_quality as formal


def timed_forward(model, ids, repeats, use_cache):
    device = next(model.parameters()).device
    ids = ids.to(device)
    for _ in range(2):
        with torch.no_grad():
            # Match the already validated formal scorer exactly.  This
            # official wrapper's positional input path is the supported one.
            model.model(ids)
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            model.model(ids)
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratio', type=float, default=0.5, choices=[0.5])
    ap.add_argument('--method', required=True, choices=['uniform', 'fisher', 'fdp_prior', 'fdp_native'])
    args = ap.parse_args()
    started = time.time()
    model, tokenizer, schedule_path, schedule_doc, selection, wiki_ids, _c4_ids = formal.load_runtime(args.ratio, args.method)
    torch.cuda.reset_peak_memory_stats()
    # formal.load_runtime returns tokenizer ids with the leading batch axis.
    prefill_ids = wiki_ids[:, :formal.SEQ]
    decode_ids = prefill_ids[:, -1:]
    prefill_ms = timed_forward(model, prefill_ids, 5, False)
    decode_ms = timed_forward(model, decode_ids, 10, False)
    torch.cuda.synchronize()
    resource = formal.resource(selection)
    doc = {
        'status': 'SYSTEM_PROFILE_COMPLETE',
        'ratio': args.ratio,
        'method': f'PALU_{args.method.upper()}',
        'schedule_sha': schedule_doc['schedule_sha'],
        'resource': resource,
        'gpu_visible_index': 0,
        'measurement': {
            'prefill_definition': 'one 2048-token forward, use_cache=false, official model forward path',
            'decode_definition': 'one-token forward proxy, use_cache=false; not a persistent-cache kernel benchmark',
            'prefill_latency_ms': prefill_ms,
            'decode_proxy_latency_ms': decode_ms,
            'prefill_latency_ms_mean': sum(prefill_ms) / len(prefill_ms),
            'decode_proxy_latency_ms_mean': sum(decode_ms) / len(decode_ms),
            'peak_memory_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_memory_reserved_bytes': torch.cuda.max_memory_reserved(),
            'wall_seconds': time.time() - started,
        },
        'cost_audit': {
            'profile_cost_measured': True,
            'exchange_calibration_cost': 'see frozen calibration logs; not rerun here',
            'formal_quality_runtime_overhead': 'see formal quality cell logs; model/evaluator load included in wall time',
        },
        'historical_results_modified': False,
    }
    out = formal.BRANCH / 'quality' / 'llama2' / 'ratio_0.5' / f'PALU_{args.method.upper()}' / 'system_profile.json'
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': doc['status'], 'method': doc['method'], 'prefill_mean_ms': doc['measurement']['prefill_latency_ms_mean'], 'decode_proxy_mean_ms': doc['measurement']['decode_proxy_latency_ms_mean'], 'peak_reserved_bytes': doc['measurement']['peak_memory_reserved_bytes'], 'out': str(out)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
