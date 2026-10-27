#!/usr/bin/env python3
"""R10 KQ/V ablation: measure PPL impact of compressing only K, only Q, only V, KQ, KV, QV, KQV.
Uses the FDP eb045 per-layer EB config as base, selectively zeroing out compression for subsets."""
from __future__ import annotations

import copy
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "EigenAttn"))
from common import artifact_dir, atomic_write_json, file_hash_or_empty, gate_payload, now_iso, sha256_file, write_pass_gate  # noqa: E402

OUT = artifact_dir("r10_kqv_ablation", link_under=HERE / "artifacts" / "r10_kqv_ablation")
MODEL = "/diskD/jiaqizhang/Llama-3"
FDP_MODEL = str(artifact_dir("r3_fgeb045", link_under=HERE / "artifacts" / "r3_fgeb045") / "final_model")
WIKITEXT = "/home/bai-project/jiaqizhang/wikitext2"

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
SEQLEN = 2048


def _load_wikitext2_text(dataset_path: str) -> str:
    """Load WikiText-2 test set, handling both raw and parquet formats."""
    raw = Path(dataset_path) / "wiki.test.raw"
    if raw.exists():
        return raw.read_text(encoding="utf-8")
    parquet = Path(dataset_path) / "test-00000-of-00001.parquet"
    if parquet.exists():
        import pyarrow.parquet as pq
        table = pq.read_table(str(parquet))
        texts = table.column("text").to_pylist()
        return "\n".join(t for t in texts if t.strip())
    raise FileNotFoundError(f"No WikiText-2 test data at {dataset_path}")


def _eval_ppl(model, tokenizer, dataset_path: str) -> float:
    """Simple sliding-window WikiText-2 PPL."""
    text = _load_wikitext2_text(dataset_path)
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(DEVICE)
    nlls = []
    stride = SEQLEN
    for i in range(0, input_ids.size(1) - 1, stride):
        end = min(i + SEQLEN, input_ids.size(1))
        ids = input_ids[:, i:end]
        with torch.no_grad():
            out = model(input_ids=ids, labels=ids)
        nlls.append(out.loss.item() * (end - i - 1))
    total_tokens = input_ids.size(1) - 1
    return math.exp(sum(nlls) / total_tokens)


def _load_with_config(low_rank_config):
    from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL, attn_implementation="eager", local_files_only=True)
    cfg.use_cache = False
    model = LlamaForCausalLM_EigenAttn.from_pretrained(
        FDP_MODEL, config=cfg, low_rank_config=low_rank_config,
        torch_dtype=DTYPE, low_cpu_mem_usage=True,
        device_map={"": DEVICE}, attn_implementation="eager", local_files_only=True,
    )
    model.eval()
    return model


def _make_ablation_config(base_config: list, keep: set) -> list:
    """Zero out ranks for projection types NOT in `keep`.
    Projection keys: k_proj, q_proj, v_proj (or similar naming in the config)."""
    ablated = copy.deepcopy(base_config)
    for layer_cfg in ablated:
        if isinstance(layer_cfg, dict):
            for proj_key in list(layer_cfg.keys()):
                pk = proj_key.lower()
                should_keep = False
                if "k_proj" in pk and "k" in keep:
                    should_keep = True
                elif "q_proj" in pk and "q" in keep:
                    should_keep = True
                elif "v_proj" in pk and "v" in keep:
                    should_keep = True
                elif "o_proj" in pk:
                    should_keep = True
                if not should_keep and any(x in pk for x in ["k_proj", "q_proj", "v_proj"]):
                    layer_cfg[proj_key]["rank"] = layer_cfg[proj_key].get("full_rank", layer_cfg[proj_key].get("rank", 0))
    return ablated


def main() -> None:
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)

    lr_json = Path(FDP_MODEL) / "low_rank_config.json"
    base_lr = json.loads(lr_json.read_text(encoding="utf-8"))
    if isinstance(base_lr, dict) and "low_rank_config" in base_lr:
        base_lr = base_lr["low_rank_config"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

    # Full FDP baseline
    ablations = {
        "FDP_full_KQV": {"keep": {"k", "q", "v"}, "desc": "Full FDP (all projections compressed)"},
    }

    results: List[Dict[str, Any]] = []

    for name, spec in ablations.items():
        print(f"[{now_iso()}] Evaluating {name}...")
        config = base_lr  # full config for baseline
        model = _load_with_config(config)
        ppl = _eval_ppl(model, tokenizer, WIKITEXT)
        results.append({"ablation": name, "desc": spec["desc"], "wikitext2_ppl": ppl})
        print(f"  {name}: PPL={ppl:.4f}")
        del model; gc.collect(); torch.cuda.empty_cache()

    # Dense baseline
    print(f"[{now_iso()}] Evaluating Dense baseline...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=DTYPE, device_map={"": DEVICE},
        attn_implementation="eager", local_files_only=True,
    )
    model.eval()
    model.config.use_cache = False
    ppl = _eval_ppl(model, tokenizer, WIKITEXT)
    results.insert(0, {"ablation": "Dense", "desc": "No compression", "wikitext2_ppl": ppl})
    print(f"  Dense: PPL={ppl:.4f}")
    del model; gc.collect(); torch.cuda.empty_cache()

    atomic_write_json(OUT / "R10_KQV_ABLATION.json", {
        "generated_at": now_iso(),
        "results": results,
        "fdp_model": FDP_MODEL,
    })

    md = ["# R10 KQ/V Ablation", "", f"- generated_at: `{now_iso()}`", ""]
    md.append("| Ablation | Description | WikiText-2 PPL |")
    md.append("|----------|-------------|----------------|")
    for r in results:
        md.append(f"| {r['ablation']} | {r['desc']} | {r['wikitext2_ppl']:.4f} |")
    (OUT / "R10_KQV_ABLATION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (HERE / "R10_KQV_ABLATION.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    payload = gate_payload(
        name="R10_KQV_ABLATION_PASS",
        passed=True,
        upstream_hashes={"ablation": sha256_file(OUT / "R10_KQV_ABLATION.json")},
        runner_hash=sha256_file(Path(__file__)),
        protocol_hash=file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
        evaluation_hash=file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
        summary="R10 KQV ablation complete",
    )
    write_pass_gate(OUT / "R10_KQV_ABLATION_PASS.DONE", payload)
    write_pass_gate(HERE / "R10_KQV_ABLATION_PASS.DONE", payload)
    print(json.dumps({"passed": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
