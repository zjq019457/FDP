#!/usr/bin/env python3
"""FDP_MEDIUM_LATENCY_V1 — EigenAttention-style (Table 5) generation latency, Llama-3-8B Medium.

Batch 1, bf16, eager attention, greedy decode. For each context length: prefill the fixed WikiText-2 test prefix,
then decode NEW tokens one at a time with the model's own KV cache. Timing uses CUDA events:
  * per-layer attention latency: forward pre/post hooks on every decoder layer's self_attn (decode steps only)
  * per-token decode latency and prefill latency (whole forward)
  * peak allocated memory during the decode phase
WARMUP full runs are discarded; REPS measured runs are reported (median and IQR over runs).
usage: latency_bench.py <label> <dense|model_dir> <out.json> [ctx ...]"""
import json, statistics, sys, time
from pathlib import Path

import torch

sys.path.insert(0, "/home/bai-project/jiaqizhang/EigenAttn/EigenAttn")
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

TOK = "/diskD/jiaqizhang/Llama-3"; WIKI = "/home/bai-project/jiaqizhang/wikitext2/test-00000-of-00001.parquet"
label, src, out = sys.argv[1:4]
CTXS = [int(x) for x in sys.argv[4:]] or [2048, 8192]
NEW, WARMUP, REPS = 128, 1, 5

tok = AutoTokenizer.from_pretrained(TOK, use_fast=False, local_files_only=True)
if src == "dense":
    model = AutoModelForCausalLM.from_pretrained(TOK, torch_dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True, attn_implementation="eager")
    ranks = None
else:
    from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn
    ranks = json.loads((Path(src) / "low_rank_config.json").read_text())
    cfg = AutoConfig.from_pretrained(TOK, attn_implementation="eager", local_files_only=True)
    model = LlamaForCausalLM_EigenAttn.from_pretrained(src, config=cfg, low_rank_config=ranks, torch_dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True)
model.eval()

import pyarrow.parquet as pq  # noqa: E402
text = "\n\n".join(pq.read_table(WIKI).column("text").to_pylist())
ids_all = tok(text, return_tensors="pt").input_ids[0]
attn_mods = [l.self_attn for l in model.model.layers]
REC = {"on": False, "ev": []}


def pre(m, a, k):
    if REC["on"]:
        e = torch.cuda.Event(enable_timing=True); e.record(); REC["ev"].append([m._li, e, None])


def post(m, a, k, o):
    if REC["on"]:
        e = torch.cuda.Event(enable_timing=True); e.record(); REC["ev"][-1][2] = e


for i, m in enumerate(attn_mods):
    m._li = i
    m.register_forward_pre_hook(pre, with_kwargs=True); m.register_forward_hook(post, with_kwargs=True)


@torch.no_grad()
def one_run(ctx):
    x = ids_all[:ctx].unsqueeze(0).to("cuda:0")
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); o = model(input_ids=x, use_cache=True); e.record()
    pkv = o.past_key_values; nxt = o.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize(); prefill = s.elapsed_time(e)
    mem_after_prefill = torch.cuda.memory_allocated()
    REC["ev"] = []; REC["on"] = True
    step_ev = []
    for _ in range(NEW):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); o = model(input_ids=nxt, past_key_values=pkv, use_cache=True); b.record()
        pkv = o.past_key_values; nxt = o.logits[:, -1:].argmax(-1); step_ev.append((a, b))
    REC["on"] = False
    torch.cuda.synchronize()
    tok_ms = [a.elapsed_time(b) for a, b in step_ev]
    per_layer = [[] for _ in attn_mods]
    for li, a, b in REC["ev"]: per_layer[li].append(a.elapsed_time(b))
    return {"prefill_ms": prefill, "decode_ms_per_token_mean": statistics.mean(tok_ms), "decode_tokens_per_s": 1000 * NEW / sum(tok_ms),
            "attn_ms_per_layer_mean": statistics.mean(statistics.mean(v) for v in per_layer),
            "attn_ms_by_layer": [statistics.mean(v) for v in per_layer], "attn_share_of_decode": sum(statistics.mean(v) for v in per_layer) / statistics.mean(tok_ms),
            "peak_alloc_GiB_decode": torch.cuda.max_memory_allocated() / 2 ** 30, "alloc_after_prefill_GiB": mem_after_prefill / 2 ** 30}


res = {"label": label, "source": src, "ranks_sum": (sum(a + b for a, b in ranks) if ranks else 2 * 1024 * 32),
       "B_live": (sum(a + b for a, b in ranks) * 4096 if ranks else 268435456), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "protocol": {"batch": 1, "dtype": "bf16", "attn": "eager", "new_tokens": NEW, "warmup_runs": WARMUP, "measured_runs": REPS, "timer": "CUDA events"}, "ctx": {}}
for ctx in CTXS:
    for _ in range(WARMUP): one_run(ctx)
    runs = [one_run(ctx) for _ in range(REPS)]
    def agg(k):
        v = sorted(r[k] for r in runs); q = statistics.quantiles(v, n=4) if len(v) >= 2 else [v[0]] * 3
        return {"median": statistics.median(v), "q1": q[0], "q3": q[2], "runs": v}
    res["ctx"][str(ctx)] = {k: agg(k) for k in ("prefill_ms", "decode_ms_per_token_mean", "decode_tokens_per_s", "attn_ms_per_layer_mean", "attn_share_of_decode",
                                                 "peak_alloc_GiB_decode", "alloc_after_prefill_GiB")}
    res["ctx"][str(ctx)]["attn_ms_by_layer_median"] = [statistics.median(r["attn_ms_by_layer"][i] for r in runs) for i in range(len(attn_mods))]
    print(label, ctx, {k: round(res["ctx"][str(ctx)][k]["median"], 4) for k in ("decode_ms_per_token_mean", "attn_ms_per_layer_mean", "prefill_ms", "peak_alloc_GiB_decode")}, flush=True)
Path(out).write_text(json.dumps(res, indent=1))
