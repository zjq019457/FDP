import os
import sys
import time
import torch

sys.path.insert(0, "/diskF/jiaqizhang/fdp_lowrank_operator_zoo_v1/repos/eigenattention_upstream")
sys.path.insert(0, "/home/bai-project/jiaqizhang/EigenAttn/EigenAttn")
sys.path.insert(0, "/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection")

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from dev_evaluator import DevEvaluator, _compat_rope
from formula_engine import TIER_CONF, load_traces, oracle_lookup
from step1_diagnose import DEFAULT_PROMPTS, build_inputs, unpatch
from step1_diagnose_llama import patch_full_llama, make_cmp_diag_forward_llama

device = "cuda:3"
dtype = torch.bfloat16

print("[1] Loading tokenizer & base model...")
tok = AutoTokenizer.from_pretrained("/diskD/jiaqizhang/Llama-3", local_files_only=True, use_fast=False)
input_ids = build_inputs(tok, DEFAULT_PROMPTS[:2], 512, device)

base = AutoModelForCausalLM.from_pretrained(
    "/diskD/jiaqizhang/Llama-3",
    local_files_only=True,
    torch_dtype=dtype,
    device_map=device
)
base.eval()
base.config.use_cache = False

print("[2] Running FULL model Phase 1...")
full_store = {}
saved = patch_full_llama(base, full_store)
try:
    with torch.no_grad():
        base.model(input_ids)
finally:
    unpatch(saved)
print(f"Captured {len(full_store)} layers from FULL.")

for li, d in full_store.items():
    for k, v in list(d.items()):
        if isinstance(v, torch.Tensor):
            d[k] = v.detach().to("cpu")

del base
torch.cuda.empty_cache()

print("[3] Initializing DevEvaluator for Light tier...")
evaluator = DevEvaluator(
    base_model_path="/diskD/jiaqizhang/Llama-3",
    basis_pt_path="/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb045/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt",
    device=device,
    dtype=dtype
)

traces = load_traces(TIER_CONF["light"]["report"])
base_uniform_ranks = []
for li in range(32):
    rk, rv = oracle_lookup(traces[li], 0.045)
    qk = f"base_model.model.model.layers.{li}.self_attn.q_proj_basis"
    vk = f"base_model.model.model.layers.{li}.self_attn.v_proj_basis"
    q_max = evaluator.basis_dict[qk].shape[1] * 8
    v_max = evaluator.basis_dict[vk].shape[1]
    base_uniform_ranks.append([min(rk, q_max), min(rv, v_max)])

evaluator.set_ranks(base_uniform_ranks)

def run_diagnostic_pass(cmp_model, full_store, input_ids):
    cmp_store = {}
    metrics = {}
    saved = []
    for li, layer in enumerate(cmp_model.model.layers):
        attn = layer.self_attn
        saved.append((attn, attn.forward))
        attn.forward = make_cmp_diag_forward_llama(attn, cmp_store, full_store, metrics, li)
    try:
        with torch.no_grad():
            cmp_model.model(input_ids)
    finally:
        unpatch(saved)
    return metrics

print("[4] Timing 5 diagnostic forward passes...")
t0 = time.time()
for step in range(5):
    t_step = time.time()
    fs_copy = {li: {k: v for k, v in full_store[li].items()} for li in full_store}
    m = run_diagnostic_pass(evaluator.cmp_model, fs_copy, input_ids)
    print(f"  Pass {step+1}: {time.time()-t_step:.3f}s, L0: Z_rel={m[0]['Z_rel_mse']:.4e}, Av_rel={m[0]['score_only_AV_rel_mse']:.4e} | L31: Z_rel={m[31]['Z_rel_mse']:.4e}")

total_time = time.time() - t0
print(f"Total time for 5 passes: {total_time:.3f}s (avg: {total_time/5:.3f}s/pass)")
