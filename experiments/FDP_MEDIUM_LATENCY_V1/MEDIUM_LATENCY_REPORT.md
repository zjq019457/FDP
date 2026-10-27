# MEDIUM_LATENCY_REPORT — EigenAttention-style generation latency (Table 5 analogue), Llama-3-8B Medium

Measured 2026-10-10 on one exclusive NVIDIA A100 80GB PCIe (no other GPU processes; torch 2.8.0+cu128). Batch 1, bf16, eager attention,
greedy decode of 128 tokens after prefilling a fixed WikiText-2 test prefix of 2,048 or 8,064 tokens (8,064 + 128 = 8,192 = max positions).
CUDA-event timing; 1 warm-up run discarded, 5 measured runs; medians reported (IQRs in `results/latency_*.json`).
Per-layer attention latency = forward pre/post hooks on every decoder layer's self-attention during decode steps.
Models: FDP-EB and the matched Uniform comparator were re-materialized from the verified cost-benchmark bases with the frozen command;
rank SHA f0c46275… (FDP, 199,753,728 B) and low_rank_config SHA db207077… (Uniform, 199,917,568 B) checked before timing.

| Context | Metric | Dense | Uniform (199.92M) | FDP-EB (199.75M) | FDP vs Uniform |
|---|---|---|---|---|---|
| 2048 | Attention latency per layer per token (ms) | 0.288 | 0.416 | 0.427 | +2.7% |
| 2048 | Decode latency per token (ms) | 20.147 | 25.808 | 26.384 | +2.2% |
| 2048 | Decode throughput (tok/s) | 49.636 | 38.747 | 37.902 | -2.2% |
| 2048 | Prefill latency (ms) | 281.553 | 287.998 | 288.945 | +0.3% |
| 2048 | Peak allocated memory (GiB) | 16.699 | 16.407 | 16.408 | +0.0% |
| 8064 | Attention latency per layer per token (ms) | 0.593 | 0.862 | 0.865 | +0.3% |
| 8064 | Decode latency per token (ms) | 29.918 | 38.643 | 38.738 | +0.2% |
| 8064 | Decode throughput (tok/s) | 33.424 | 25.878 | 25.814 | -0.2% |
| 8064 | Prefill latency (ms) | 2525.035 | 2698.141 | 2723.723 | +0.9% |
| 8064 | Peak allocated memory (GiB) | 35.826 | 35.340 | 35.340 | -0.0% |

## Reading
- **FDP-EB vs Uniform (the relevant comparison):** decode latency +2.2% at 2K and +0.25% at 8K; attention +2.7% / +0.3%; per-layer FDP/Uniform ratios 1.006–1.044 (2K) and 0.97–1.07 (8K). The 2K difference is within the run-to-run IQR overlap. Non-uniform per-layer ranks therefore add no material inference latency at equal total KV bytes.
- **EigenAttention vs Dense:** in this research implementation (unfused eager low-rank projections) both compressed models are slower than Dense (attention about +45%). This mirrors the direction of the EigenAttention paper's Llama-3-70B row (+8%) but is an implementation property; do not report it as an FDP effect or as an optimized-kernel number.
- **Memory:** the batch-1 peak is dominated by weights and prefill activations; the KV saving is visible but small (16.41 vs 16.70 GiB at 2K, 35.34 vs 35.83 GiB at 8K).
