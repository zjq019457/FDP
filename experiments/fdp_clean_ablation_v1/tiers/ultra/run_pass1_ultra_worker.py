#!/usr/bin/env python3
import os
import sys
import json
import time
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

UPSTREAM = Path("/diskF/jiaqizhang/fdp_lowrank_operator_zoo_v1/repos/eigenattention_upstream")
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

# RoPE device fix
_original_rope_forward = LlamaRotaryEmbedding.forward
def _compat_rope_forward(self, x, position_ids):
    if hasattr(self, "inv_freq") and torch.is_tensor(self.inv_freq):
        if self.inv_freq.device != position_ids.device:
            self.inv_freq = self.inv_freq.to(position_ids.device)
    return _original_rope_forward(self, x, position_ids)
LlamaRotaryEmbedding.forward = _compat_rope_forward

from decompose.eigen_attn import eigenattn
from models.LMClass import LMClass
from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn

class Logger:
    def info(self, msg):
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}", flush=True)

def main():
    model_path = "/diskD/jiaqizhang/Llama-3"
    save_dir = "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/pass1_uniform/final_model"
    scalar_eb = 0.120
    nsamples = 64
    seqlen = 2048
    avg_dim = 32
    seed = 2

    log = Logger()
    log.info(f"=== Starting Pass 1 Uniform e={scalar_eb} on GPU {os.environ.get('CUDA_VISIBLE_DEVICES')} ===")
    
    model_args = SimpleNamespace(
        model=model_path,
        cache_dir="/home/bai-project/jiaqizhang/EigenAttn/EigenAttn/HF_cache",
        calib_dataset="wikitext2",
        nsamples=nsamples,
        seed=seed,
        batch_size=2,
        attn_implementation="eager",
        net="Llama-3",
        load_low_rank=False,
        load_peft_model=False,
        peft_model_path=None,
        save_dir=None,
        output_dir=None,
        evaluate_baseline=False,
        avg_dim=avg_dim,
        error_budget=scalar_eb,
        fine_tune=False,
        model_family="Llama",
    )
    
    torch.manual_seed(seed)
    lm = LMClass(model_args)
    lm.seqlen = seqlen
    
    cache_path = Path(model_args.cache_dir) / f"dataloader_Llama_wikitext2_{nsamples}.cache"
    log.info(f"Loading calibration from {cache_path}")
    dataloader = torch.load(cache_path, map_location="cpu")
    
    model_args.eigen_attn_params = {
        "threshold": 1.0,
        "avg_dim_features": avg_dim,
        "error_budget": scalar_eb,
    }
    
    tick = time.time()
    eigenattn(lm, model_args, dataloader, log)
    elapsed = time.time() - tick
    log.info(f"Decomposition completed in {elapsed:.2f} s")
    
    layers = lm.model.model.layers
    low_rank_config = []
    for i in range(len(layers)):
        low_rank_config.append([int(layers[i].self_attn.rank_kq), int(layers[i].self_attn.rank_v)])
        
    tot_rank = sum(k + v for k, v in low_rank_config)
    b_live = seqlen * 2 * tot_rank
    rank_sha = hashlib.sha256(json.dumps(low_rank_config, sort_keys=True).encode("utf-8")).hexdigest()
    
    log.info(f"Pass 1 Ranks extracted: total={tot_rank}, B_live={b_live}, SHA={rank_sha}")
    
    log.info(f"Materializing compressed model to {save_dir}...")
    config = AutoConfig.from_pretrained(model_path, attn_implementation="eager")
    model2 = LlamaForCausalLM_EigenAttn.from_pretrained(
        model_path,
        config=config,
        low_rank_config=low_rank_config,
        device_map="cpu",
        torch_dtype=torch.bfloat16
    )
    layers2 = model2.model.layers
    for i in range(len(layers2)):
        layers2[i].self_attn.k_proj_low.weight.data = layers[i].self_attn.k_proj_low.weight.data
        layers2[i].self_attn.k_proj_up.weight.data = layers[i].self_attn.k_proj_up.weight.data
        layers2[i].self_attn.v_proj_low.weight.data = layers[i].self_attn.v_proj_low.weight.data
        layers2[i].self_attn.o_proj_up.weight.data = layers[i].self_attn.o_proj_up.weight.data
        
    save_p = Path(save_dir)
    save_p.mkdir(parents=True, exist_ok=True)
    model2.save_pretrained(save_p)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(save_p)
    
    with open(save_p / "low_rank_config.json", "w") as f:
        json.dump(low_rank_config, f, indent=2)
    torch.save(torch.tensor(low_rank_config, dtype=torch.long), save_p / "low_rank_config.pt")
    
    meta = {
        "status": "PASS1_COMPLETE",
        "error_budget": scalar_eb,
        "total_rank_sum": tot_rank,
        "B_live": b_live,
        "rank_sha": rank_sha,
        "elapsed_seconds": elapsed,
        "low_rank_config": low_rank_config
    }
    with open(save_p / "pass1_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Pass 1 successfully saved to {save_dir}!")

if __name__ == "__main__":
    main()
