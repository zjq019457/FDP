"""CPU-only structural smoke for the isolated MPT EigenAttention adapter.

This is a synthetic small-config operator/cache test plus a read-only check of
the frozen 7B config. It is not a quality evaluation or a 7B checkpoint parity
result.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.mpt.configuration_mpt import MptConfig
from transformers.models.mpt.modeling_mpt import MptAttention
from mpt_fdp_protocol_v1 import validate_mpt_config

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MPT_MODEL_DIR", "/diskF/jiaqizhang/models/新建文件夹"))
EIGENATTN_BASE = Path(os.environ.get(
    "EIGENATTN_BASE", "/home/bai-project/jiaqizhang/EigenAttn/EigenAttn_base"
))
ADAPTER_FILE = HERE / "modelling_mpt_eigen_attn_v1.py"
sys.path.insert(0, str(EIGENATTN_BASE))
spec = importlib.util.spec_from_file_location("mpt_eigen_adapter_v1", ADAPTER_FILE)
adapter = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(adapter)


def alibi_mask(seq_len: int, max_len: int, heads: int, device: str = "cpu"):
    bias = adapter.build_mpt_alibi_tensor(heads, max_len, device=device)
    mask = torch.triu(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
    return bias, mask


def tensor_metrics(a: torch.Tensor, b: torch.Tensor):
    diff = (a.float() - b.float()).abs()
    denom = a.float().norm().clamp_min(1e-12)
    return {
        "shape_a": list(a.shape), "shape_b": list(b.shape),
        "max_abs": float(diff.max().item()),
        "relative_l2": float((diff.norm() / denom).item()),
        "cosine": float(torch.nn.functional.cosine_similarity(
            a.float().reshape(1, -1), b.float().reshape(1, -1), dim=1
        ).item()),
    }


def main():
    frozen_cfg = AutoConfig.from_pretrained(str(MODEL_DIR), local_files_only=True, trust_remote_code=False)
    identity = validate_mpt_config(frozen_cfg)
    adapter_source = ADAPTER_FILE.read_text(encoding="utf-8")
    cache_fix_source_check = {
        "correct_key_sequence_axis_occurrences": adapter_source.count("past_key_values[0][0].shape[2]"),
        "old_tuple_last_element_bug_occurrences": adapter_source.count("past_key_values[0][-1]"),
    }

    torch.manual_seed(4102026)
    cfg = MptConfig(
        d_model=32, n_heads=4, n_layers=1, expansion_ratio=2,
        max_seq_len=16, vocab_size=64, learned_pos_emb=False,
        use_cache=True,
        attn_config={
            "alibi": True, "attn_impl": "torch", "attn_pdrop": 0.0,
            "softmax_scale": None, "alibi_bias_max": 8,
            "attn_type": "multihead_attention", "clip_qkv": None,
        },
    )
    dense = MptAttention(cfg).eval()
    full = adapter.MptEigenAttention(cfg, rank_kq=32, rank_v=32).eval()
    with torch.no_grad():
        full.Wqkv_low.weight.copy_(dense.Wqkv.weight)
        full.out_proj_up.weight.copy_(dense.out_proj.weight)

        x = torch.randn(1, 6, 32)
        bias, mask = alibi_mask(6, 16, 4)
        y_dense, _, cache_dense = dense(x, position_bias=bias, attention_mask=mask)
        y_full, _, cache_full = full(x, position_bias=bias, attention_mask=mask)
        full_forward = tensor_metrics(y_dense, y_full)

        # Incremental attention/cache parity at the attention-module boundary.
        next_x = torch.randn(1, 1, 32)
        next_bias, _ = alibi_mask(1, 16, 4)
        next_mask = torch.zeros((1, 1, 1, 7), dtype=torch.bool)
        yd_inc, _, cd_inc = dense(
            next_x, position_bias=next_bias, past_key_value=cache_dense,
            attention_mask=next_mask,
        )
        yf_inc, _, cf_inc = full(
            next_x, position_bias=next_bias, past_key_value=cache_full,
            attention_mask=next_mask,
        )
        incremental = tensor_metrics(yd_inc, yf_inc)
        cache_shapes_fullrank = {
            "dense_k": list(cache_dense[0].shape),
            "dense_v": list(cache_dense[1].shape),
            "eigen_k": list(cache_full[0].shape),
            "eigen_v": list(cache_full[1].shape),
            "incremental_eigen_k": list(cf_inc[0].shape),
            "incremental_eigen_v": list(cf_inc[1].shape),
        }

        compressed = adapter.MptEigenAttention(cfg, rank_kq=16, rank_v=12).eval()
        yc, _, cc = compressed(x, position_bias=bias, attention_mask=mask)
        compressed_shapes = {
            "output": list(yc.shape),
            "k_cache": list(cc[0].shape),
            "v_cache": list(cc[1].shape),
            "rank_kq": 16,
            "rank_v": 12,
        }
        expected_compressed = (list(yc.shape) == [1, 6, 32]
            and list(cc[0].shape) == [1, 4, 6, 4]
            and list(cc[1].shape) == [1, 4, 6, 3])

        # Exercise the adapted transformer's own past-length path, not only
        # the attention module's cache concatenation.
        tiny_transformer = adapter.MptModel_EigenAttn(
            low_rank_config=[[16, 12]], config=cfg
        ).eval()
        first = tiny_transformer(
            input_ids=torch.tensor([[1, 2, 3]]), use_cache=True, return_dict=True
        )
        second = tiny_transformer(
            input_ids=torch.tensor([[4]]), past_key_values=first.past_key_values,
            use_cache=True, return_dict=True,
        )
        outer_cache_forward = {
            "first_hidden_shape": list(first.last_hidden_state.shape),
            "second_hidden_shape": list(second.last_hidden_state.shape),
            "cached_k_shape": list(second.past_key_values[0][0].shape),
            "cached_v_shape": list(second.past_key_values[0][1].shape),
            "finite": bool(torch.isfinite(second.last_hidden_state).all().item()),
        }
        prepared = adapter.MptForCausalLM_EigenAttn.prepare_inputs_for_generation(
            object(), input_ids=torch.tensor([[1, 2, 3, 4]]),
            past_key_values=first.past_key_values, use_cache=True,
        )
        generation_slice = {
            "prepared_input_shape": list(prepared["input_ids"].shape),
            "prepared_input_ids": prepared["input_ids"].tolist(),
        }

    report = {
        "schema": "MPT7B_NATIVE_OPERATOR_SMOKE_V1",
        "status": "SYNTHETIC_CPU_OPERATOR_AND_CACHE_SMOKE_PASS" if (
            full_forward["max_abs"] < 1e-6 and incremental["max_abs"] < 1e-6
            and expected_compressed and outer_cache_forward["finite"]
            and generation_slice["prepared_input_ids"] == [[4]]
        ) else "FAIL",
        "scope_warning": "small synthetic config only; not a checkpoint-weight parity or quality result",
        "frozen_checkpoint_config": identity,
        "synthetic_config": {"layers": 1, "hidden_size": 32, "heads": 4, "context": 16},
        "full_rank_dense_vs_eigen_attention": full_forward,
        "full_rank_incremental_cache_attention": incremental,
        "full_rank_cache_shapes": cache_shapes_fullrank,
        "compressed_cache_shapes": compressed_shapes,
        "outer_transformer_incremental_cache_forward": outer_cache_forward,
        "prepare_inputs_for_generation_cache_slice": generation_slice,
        "cache_fix_source_check": cache_fix_source_check,
        "outer_transformer_incremental_cache_forward_tested": True,
        "adapter_sha256": hashlib.sha256(ADAPTER_FILE.read_bytes()).hexdigest(),
        "official_source_path": "/home/bai-project/jiaqizhang/EigenAttn/EigenAttn_base/models/modelling_mpt_eigen_attn.py",
        "official_source_sha256": "89ee4efbf17da3e2b0734ac74315ea0f9bc491e6ab1be5a48dfd1a909924b3fd",
        "fixes": [
            "read past sequence length from K tensor axis 2 in MptModel_EigenAttn",
            "read past sequence length from K tensor axis 2 in prepare_inputs_for_generation",
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    report_path = os.environ.get("MPT_SMOKE_REPORT")
    if report_path:
        output = Path(report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
