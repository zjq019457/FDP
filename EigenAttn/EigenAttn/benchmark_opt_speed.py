import torch
import time
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# 1. 全局配置
# =========================
DTYPE = torch.float16
DEVICE = "cuda"
DECODE_STEPS = 128
WARMUP_STEPS = 10

# =========================
# 2. 构造稳定 prompt（OPT 友好）
# =========================
def build_prompt(tokenizer, length):
    text = "The history of artificial intelligence is"
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    if input_ids.shape[1] < length:
        pad = input_ids[:, -1:].repeat(1, length - input_ids.shape[1])
        input_ids = torch.cat([input_ids, pad], dim=1)

    return input_ids.to(DEVICE)

# =========================
# 3. 核心测速函数（decode）
# =========================
@torch.no_grad()
def benchmark_decode(model, input_ids):
    model.eval()

    # warmup
    for _ in range(WARMUP_STEPS):
        _ = model(input_ids, use_cache=True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.time()

    for _ in range(DECODE_STEPS):
        out = model(input_ids, use_cache=True)
        next_token = out.logits[:, -1:].argmax(dim=-1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    torch.cuda.synchronize()
    end = time.time()

    total_time = end - start

    return {
        "tokens_per_second": DECODE_STEPS / total_time,
        "ms_per_token": 1000.0 * total_time / DECODE_STEPS,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }

# =========================
# 4. KV cache 显存估算（理论）
# =========================
def estimate_kv_cache(model, seq_len):
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_head
    bytes_per_elem = 2  # fp16

    kv_bytes = (
        n_layer * 2 * n_head * seq_len * head_dim * bytes_per_elem
    )
    return kv_bytes / 1024 / 1024

# =========================
# 5. 跑单个模型
# =========================
def run_model(model_path, tag, context_len):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
        device_map="auto",
        local_files_only=True
    )

    input_ids = build_prompt(tokenizer, context_len)

    stats = benchmark_decode(model, input_ids)
    kv_mb = estimate_kv_cache(model, context_len)

    return {
        "model": tag,
        "path": model_path,
        "context_length": context_len,
        "decode_steps": DECODE_STEPS,
        "tokens_per_second": stats["tokens_per_second"],
        "ms_per_token": stats["ms_per_token"],
        "peak_memory_mb": stats["peak_memory_mb"],
        "estimated_kv_cache_mb": kv_mb,
    }

# =========================
# 6. 主函数
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--eigen", type=str, required=True)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--out", type=str, default="opt_speed_results.json")
    args = parser.parse_args()

    results = []

    print("\n===== Baseline OPT (original) =====")
    results.append(run_model(args.baseline, "baseline", args.context))

    print("\n===== EigenAttention OPT =====")
    results.append(run_model(args.eigen, "eigen", args.context))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== Final Results =====")
    print(json.dumps(results, indent=2))

    print("\n===== FLOPs Insight =====")
    print(
        "Baseline OPT: QKV projection O(3 * d^2)\n"
        "EigenAttention: low-rank projection O(3 * d * r)\n"
        "Speedup comes from reduced projection + memory traffic."
    )

if __name__ == "__main__":
    main()
