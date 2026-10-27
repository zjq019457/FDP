#!/usr/bin/env python3
"""R8 system metrics: prefill + decode latency/throughput for Dense, Uniform, FDP on Llama-3-8B.
Protocol: warmup=10, repeat=30, lengths=[(2048,128),(8192,128)], batch=1, exclusive GPU."""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "EigenAttn"))
from common import artifact_dir, atomic_write_json, file_hash_or_empty, gate_payload, now_iso, sha256_file, write_pass_gate  # noqa: E402

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

OUT = artifact_dir("r8_system_metrics", link_under=HERE / "artifacts" / "r8_system_metrics")
MODEL = "/diskD/jiaqizhang/Llama-3"
UNI_MODEL = str(artifact_dir("r2_uniform_eb045", link_under=HERE / "artifacts" / "r2_uniform_eb045") / "final_model")
FDP_MODEL = str(artifact_dir("r3_fgeb045", link_under=HERE / "artifacts" / "r3_fgeb045") / "final_model")

WARMUP = 10
REPEAT = 30
CONFIGS = [(2048, 128), (8192, 128)]
BATCH = 1
DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def _median(xs: List[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    m = n // 2
    return xs[m] if n % 2 == 1 else 0.5 * (xs[m - 1] + xs[m])


def _load_dense():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=DTYPE, device_map={"": DEVICE},
        attn_implementation="eager", local_files_only=True,
    )
    model.eval()
    model.config.use_cache = True
    return model


def _load_eigen(model_dir: str):
    from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn
    from transformers import AutoConfig
    lr_json = Path(model_dir) / "low_rank_config.json"
    low = json.loads(lr_json.read_text(encoding="utf-8"))
    if isinstance(low, dict) and "low_rank_config" in low:
        low = low["low_rank_config"]
    cfg = AutoConfig.from_pretrained(MODEL, attn_implementation="eager", local_files_only=True)
    cfg.use_cache = True
    model = LlamaForCausalLM_EigenAttn.from_pretrained(
        model_dir, config=cfg, low_rank_config=low,
        torch_dtype=DTYPE, low_cpu_mem_usage=True,
        device_map={"": DEVICE}, attn_implementation="eager", local_files_only=True,
    )
    model.eval()
    return model


def _bench_model(model, label: str) -> List[Dict[str, Any]]:
    vocab = model.get_input_embeddings().weight.shape[0]
    rows = []
    for prefill_len, decode_steps in CONFIGS:
        # Prefill benchmark
        ids = torch.randint(0, vocab, (BATCH, prefill_len), device=DEVICE)
        for _ in range(WARMUP):
            with torch.no_grad():
                out = model(input_ids=ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_times = []
        for _ in range(REPEAT):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(input_ids=ids, use_cache=True)
            torch.cuda.synchronize()
            prefill_times.append(time.perf_counter() - t0)

        # Decode benchmark
        past = out.past_key_values
        next_tok = torch.randint(0, vocab, (BATCH, 1), device=DEVICE)
        for _ in range(WARMUP):
            with torch.no_grad():
                model(input_ids=next_tok, past_key_values=past, use_cache=True)
        torch.cuda.synchronize()
        decode_times = []
        for _ in range(REPEAT):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(input_ids=next_tok, past_key_values=past, use_cache=True)
            torch.cuda.synchronize()
            decode_times.append(time.perf_counter() - t0)

        # Peak memory
        torch.cuda.reset_peak_memory_stats(DEVICE)
        with torch.no_grad():
            model(input_ids=ids, use_cache=True)
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)

        rows.append({
            "method": label,
            "prefill_len": prefill_len,
            "decode_steps": decode_steps,
            "prefill_median_ms": round(_median(prefill_times) * 1000, 2),
            "decode_median_ms": round(_median(decode_times) * 1000, 2),
            "prefill_tokens_per_sec": round(prefill_len / _median(prefill_times), 1),
            "peak_memory_mb": round(peak_mb, 1),
        })
        del out, past
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []

    print(f"[{now_iso()}] Loading Dense model...")
    model = _load_dense()
    all_rows.extend(_bench_model(model, "Dense"))
    del model; gc.collect(); torch.cuda.empty_cache()

    print(f"[{now_iso()}] Loading Uniform eb045 model...")
    model = _load_eigen(UNI_MODEL)
    all_rows.extend(_bench_model(model, "Uniform_eb045"))
    del model; gc.collect(); torch.cuda.empty_cache()

    print(f"[{now_iso()}] Loading FGeb045 FDP model...")
    model = _load_eigen(FDP_MODEL)
    all_rows.extend(_bench_model(model, "FGeb045_FDP"))
    del model; gc.collect(); torch.cuda.empty_cache()

    atomic_write_json(OUT / "R8_SYSTEM_METRICS.json", {
        "generated_at": now_iso(),
        "warmup": WARMUP,
        "repeat": REPEAT,
        "batch": BATCH,
        "device": DEVICE,
        "rows": all_rows,
    })

    # Markdown
    md = ["# R8 System Metrics", "", f"- generated_at: `{now_iso()}`",
          f"- warmup: {WARMUP}, repeat: {REPEAT}, batch: {BATCH}", ""]
    md.append("| Method | Prefill Len | Prefill ms | Decode ms | Tok/s | Peak MB |")
    md.append("|--------|------------|-----------|----------|-------|---------|")
    for r in all_rows:
        md.append(f"| {r['method']} | {r['prefill_len']} | {r['prefill_median_ms']} | "
                  f"{r['decode_median_ms']} | {r['prefill_tokens_per_sec']} | {r['peak_memory_mb']} |")
    (OUT / "R8_SYSTEM_METRICS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (HERE / "R8_SYSTEM_METRICS.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    payload = gate_payload(
        name="R8_SYSTEM_METRICS_PASS",
        passed=True,
        upstream_hashes={"metrics": sha256_file(OUT / "R8_SYSTEM_METRICS.json")},
        runner_hash=sha256_file(Path(__file__)),
        protocol_hash=file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
        evaluation_hash=file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
        summary="R8 system metrics complete",
    )
    write_pass_gate(OUT / "R8_SYSTEM_METRICS_PASS.DONE", payload)
    write_pass_gate(HERE / "R8_SYSTEM_METRICS_PASS.DONE", payload)
    print(json.dumps({"passed": True, "rows": len(all_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
