#!/usr/bin/env python3
"""Instrumented PPL runner: records window-level NLL without changing legacy scorer semantics.

Copies the narrow loss path from EigenAttn/sanity_ppl_dual_protocol.py (read-only original).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Isolation: do not modify EigenAttn sources; import them read-only.
_REPO = Path(__file__).resolve().parents[2]
_EIGEN = _REPO / "EigenAttn"
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_EIGEN))

from common import (  # noqa: E402
    PYTHON_BIN,
    atomic_write_json,
    file_hash_or_empty,
    merge_window_dir_to_jsonl,
    now_iso,
    sha256_file,
    sha256_json,
    sha256_tensor_ids,
    sha256_text,
    stop_requested,
    window_is_complete,
    write_window_atomic,
)

RUNNER_PATH = Path(__file__).resolve()
RUNNER_HASH = sha256_file(RUNNER_PATH)


def _load_c4_windows(texts_jsonl: Path, tok, seqlen: int) -> List[Dict[str, Any]]:
    windows = []
    with texts_jsonl.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj["text"]
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
            # Selected windows were cut for the original tokenizer (often Llama).
            # Other tokenizers (e.g. OPT) may yield fewer tokens; pad with eos / truncate.
            if ids.shape[1] > seqlen:
                ids = ids[:, :seqlen]
            elif ids.shape[1] < seqlen:
                pad_id = tok.eos_token_id
                if pad_id is None:
                    pad_id = tok.pad_token_id
                if pad_id is None:
                    raise RuntimeError(
                        f"C4 window {i} tokenized to {ids.shape[1]} < seqlen={seqlen} "
                        "and tokenizer has no eos/pad id"
                    )
                import torch

                n_pad = seqlen - ids.shape[1]
                ids = torch.cat(
                    [ids, ids.new_full((1, n_pad), int(pad_id))],
                    dim=1,
                )
            windows.append(
                {
                    "window_id": f"c4_{i:05d}",
                    "source_record_id": str(obj.get("row", i)),
                    "token_start": int(obj.get("token_start", 0)),
                    "token_end": int(obj.get("token_end", seqlen)),
                    "input_ids": ids,
                    "dataset": "c4",
                    "split": "validation_selected",
                }
            )
    return windows


def _load_wikitext_windows(tok, seqlen: int, seed: int = 2) -> List[Dict[str, Any]]:
    from datautils import get_loaders
    from sanity_AB_opt30b import _wikitext_te_to_tensor

    _, te = get_loaders(
        "wikitext2",
        nsamples=128,
        seed=seed,
        seqlen=seqlen,
        model=tok.name_or_path if hasattr(tok, "name_or_path") else None,
    )
    te = _wikitext_te_to_tensor(te)
    # te shape [1, N]; legacy: nsamples = numel // seqlen, non-overlapping chunks
    n = te.numel() // seqlen
    windows = []
    for i in range(n):
        batch = te[:, i * seqlen : (i + 1) * seqlen]
        windows.append(
            {
                "window_id": f"wiki_{i:05d}",
                "source_record_id": f"wikitext2_test_chunk_{i}",
                "token_start": i * seqlen,
                "token_end": (i + 1) * seqlen,
                "input_ids": batch,
                "dataset": "wikitext2",
                "split": "test",
            }
        )
    return windows


def eval_windows_legacy_llama(
    model,
    windows: List[Dict[str, Any]],
    *,
    device,
    seqlen: int,
    dataset: str,
    method: str,
    model_hash: str,
    basis_hash: str,
    rank_plan_hash: str,
    evaluation_protocol_hash: str,
    out_dir: Path,
    limit: int = -1,
    skip_nonfinite: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """WikiText-style HF forward CE on float logits (sanity_ppl_dual_protocol Wiki path).
    For C4 Llama path in dual_protocol: CE on native logits without .float() on logits,
    and no skip — we branch by dataset.
    """
    import torch

    model.eval()
    use_cache = getattr(model.config, "use_cache", False) if hasattr(model, "config") else False
    if hasattr(model, "config"):
        model.config.use_cache = False

    loss_fct = torch.nn.CrossEntropyLoss()
    window_dir = out_dir / "windows"
    expected = {
        "runner_hash": RUNNER_HASH,
        "model_hash": model_hash,
    }
    total_nll_sum = 0.0
    token_count_sum = 0
    used = 0
    skipped = 0

    n_windows = len(windows) if limit < 0 else min(len(windows), limit)

    for i in range(n_windows):
        if stop_requested():
            raise SystemExit("STOP_REQUESTED")
        w = windows[i]
        wid = w["window_id"]
        if window_is_complete(window_dir, wid, expected):
            rec = json.loads((window_dir / f"{wid}.json").read_text(encoding="utf-8"))
            if rec.get("skipped"):
                skipped += 1
                continue
            total_nll_sum += float(rec["total_nll"])
            token_count_sum += int(rec["token_count"])
            used += 1
            continue

        batch = w["input_ids"].to(device)
        assert batch.shape[-1] == seqlen, (batch.shape, seqlen)
        input_ids_sha = sha256_tensor_ids(batch)
        attn = torch.ones_like(batch, dtype=torch.bool)
        attn_sha = sha256_tensor_ids(attn.long())

        with torch.no_grad():
            logits = model(input_ids=batch, use_cache=False).logits
            shift_logits = logits[:, :-1, :]
            shift_labels = batch[:, 1:].to(logits.device)
            if dataset == "c4":
                # Legacy C4 Llama: CE on logits dtype as-is (no logits.float())
                loss = loss_fct(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                )
            else:
                # WikiText: logits.float() CE
                loss = loss_fct(
                    shift_logits.reshape(-1, shift_logits.size(-1)).float(),
                    shift_labels.reshape(-1),
                )
            loss_f = loss.float()
            finite = bool(torch.isfinite(loss_f).all().item())
            # token_count denominator matches legacy: seqlen per used chunk
            # total_nll = mean_token_nll * seqlen  (== loss * seqlen)
            total_nll = float((loss_f * seqlen).item()) if finite else float("nan")
            mean_nll = float(loss_f.item()) if finite else float("nan")

        skipped_flag = False
        if dataset == "wikitext2" and skip_nonfinite and not finite:
            skipped_flag = True
            skipped += 1
        elif not finite:
            raise RuntimeError(f"nonfinite loss on window {wid}; fail-closed")

        rec = {
            "dataset": dataset,
            "split": w["split"],
            "window_id": wid,
            "source_record_id": w["source_record_id"],
            "token_start": w["token_start"],
            "token_end": w["token_end"],
            "token_count": seqlen if not skipped_flag else 0,
            "input_ids_sha256": input_ids_sha,
            "attention_mask_sha256": attn_sha,
            "total_nll": total_nll if not skipped_flag else None,
            "mean_nll": mean_nll if not skipped_flag else None,
            "skipped": skipped_flag,
            "method": method,
            "model_hash": model_hash,
            "basis_hash": basis_hash,
            "rank_plan_hash": rank_plan_hash,
            "evaluation_protocol_hash": evaluation_protocol_hash,
            "runner_hash": RUNNER_HASH,
        }
        write_window_atomic(window_dir, wid, rec)
        if not skipped_flag:
            total_nll_sum += total_nll
            token_count_sum += seqlen
            used += 1

    if hasattr(model, "config"):
        model.config.use_cache = use_cache

    if token_count_sum == 0:
        ppl = float("nan")
    else:
        ppl = math.exp(total_nll_sum / token_count_sum)

    agg = {
        "dataset": dataset,
        "ppl": ppl,
        "total_nll_sum": total_nll_sum,
        "token_count_sum": token_count_sum,
        "used_windows": used,
        "skipped_windows": skipped,
        "n_windows_considered": n_windows,
        "seqlen": seqlen,
        "method": method,
        "model_hash": model_hash,
        "runner_hash": RUNNER_HASH,
        "recomputed_from": "exp(sum(total_nll)/sum(token_count))",
        "timestamp": now_iso(),
    }
    return ppl, agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["wikitext2", "c4"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--c4_texts", default="")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--method", default="dense")
    ap.add_argument("--net", default="", help="optional: llama|opt|qwen|...; auto from config if empty")
    ap.add_argument("--basis_hash", default="")
    ap.add_argument("--rank_plan_hash", default="")
    ap.add_argument("--evaluation_protocol_hash", default="")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--wikitext2_dir", default="/home/bai-project/jiaqizhang/wikitext2")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.wikitext2_dir:
        os.environ["WIKITEXT2_DATA_DIR"] = os.path.abspath(args.wikitext2_dir)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tok = AutoTokenizer.from_pretrained(args.tok, use_fast=False, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model_hash = sha256_text(args.model + "|" + file_hash_or_empty(Path(args.model) / "config.json"))
    model_dir = Path(args.model)
    lr_json = model_dir / "low_rank_config.json"
    lr_pt = model_dir / "low_rank_config.pt"
    is_compressed = lr_json.exists() or lr_pt.exists() or (args.method not in ("dense", ""))
    if is_compressed and (lr_json.exists() or lr_pt.exists()):
        if lr_json.exists():
            low = json.loads(lr_json.read_text(encoding="utf-8"))
            if isinstance(low, dict) and "low_rank_config" in low:
                low = low["low_rank_config"]
        else:
            pt = torch.load(lr_pt, map_location="cpu")
            low = [[int(x[0]), int(x[1])] for x in pt]
        cfg = AutoConfig.from_pretrained(args.tok, attn_implementation="eager", local_files_only=True)
        cfg.use_cache = False
        net = (args.net or getattr(cfg, "model_type", "") or args.method or "").lower()
        if not net:
            net = str(args.tok).lower()
        load_kwargs = dict(
            config=cfg,
            low_rank_config=low,
            device_map={"": args.device} if args.device != "cpu" else "cpu",
            torch_dtype=dtype,
            local_files_only=True,
        )
        if "qwen" in net:
            from models.modelling_qwen2_eigen_attn import Qwen2ForCausalLM_EigenAttn

            if not hasattr(cfg, "attention_bias"):
                cfg.attention_bias = True
            if not hasattr(cfg, "mlp_bias"):
                cfg.mlp_bias = False
            model = Qwen2ForCausalLM_EigenAttn.from_pretrained(str(model_dir), **load_kwargs)
        elif "opt" in net:
            from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

            model = OPTForCausalLM_EigenAttn.from_pretrained(str(model_dir), **load_kwargs)
        else:
            from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn

            if not hasattr(cfg, "attention_bias"):
                cfg.attention_bias = False
            if not hasattr(cfg, "mlp_bias"):
                cfg.mlp_bias = False
            model = LlamaForCausalLM_EigenAttn.from_pretrained(str(model_dir), **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            local_files_only=True,
            device_map={"": args.device} if args.device != "cpu" else "cpu",
            trust_remote_code=True,
        )
    model.eval()

    if args.dataset == "c4":
        texts = Path(args.c4_texts)
        if not texts.is_file():
            raise SystemExit(f"missing c4 texts: {texts}")
        windows = _load_c4_windows(texts, tok, args.seqlen)
    else:
        # get_loaders needs model path string for tokenizer sometimes
        class _TokProxy:
            name_or_path = args.tok

        windows = _load_wikitext_windows(tok if False else _TokProxy(), args.seqlen, seed=args.seed)
        # Fix: pass tok path via env / datautils; re-load properly
        from datautils import get_loaders
        from sanity_AB_opt30b import _wikitext_te_to_tensor

        _, te = get_loaders(
            "wikitext2",
            nsamples=128,
            seed=args.seed,
            seqlen=args.seqlen,
            model=args.tok,
        )
        te = _wikitext_te_to_tensor(te)
        n = te.numel() // args.seqlen
        windows = []
        for i in range(n):
            batch = te[:, i * args.seqlen : (i + 1) * args.seqlen]
            windows.append(
                {
                    "window_id": f"wiki_{i:05d}",
                    "source_record_id": f"wikitext2_test_chunk_{i}",
                    "token_start": i * args.seqlen,
                    "token_end": (i + 1) * args.seqlen,
                    "input_ids": batch,
                    "dataset": "wikitext2",
                    "split": "test",
                }
            )

    ppl, agg = eval_windows_legacy_llama(
        model,
        windows,
        device=args.device if args.device != "cpu" else "cpu",
        seqlen=args.seqlen,
        dataset=args.dataset,
        method=args.method,
        model_hash=model_hash,
        basis_hash=args.basis_hash,
        rank_plan_hash=args.rank_plan_hash,
        evaluation_protocol_hash=args.evaluation_protocol_hash or sha256_text("legacy_eval_v1"),
        out_dir=out_dir,
        limit=args.limit,
        skip_nonfinite=(args.dataset == "wikitext2"),
    )

    n_merged = merge_window_dir_to_jsonl(out_dir / "windows", out_dir / "ppl_windows.jsonl")
    atomic_write_json(out_dir / "ppl_aggregate.json", agg)
    audit = {
        "runner": str(RUNNER_PATH),
        "runner_hash": RUNNER_HASH,
        "python": PYTHON_BIN,
        "merged_windows": n_merged,
        "aggregate": agg,
        "original_scorer_source": str(_EIGEN / "sanity_ppl_dual_protocol.py"),
        "original_scorer_sha256": file_hash_or_empty(_EIGEN / "sanity_ppl_dual_protocol.py"),
        "notes": [
            "WikiText uses HF-forward float CE path from sanity_ppl_dual_protocol.eval_ppl_wikitext_hf_forward_llama_style",
            "C4 Llama uses CE without logits.float() matching eval_ppl_c4_legacy_opt_compatible non-OPT branch",
            "token_count per window = seqlen; aggregate = exp(sum(total_nll)/sum(token_count))",
        ],
        "timestamp": now_iso(),
    }
    atomic_write_json(out_dir / "ppl_raw_audit.json", audit)
    md = [
        "# PPL Raw Output Interface Audit",
        "",
        f"- generated_at: `{now_iso()}`",
        f"- runner_hash: `{RUNNER_HASH}`",
        f"- dataset: `{args.dataset}`",
        f"- ppl: `{ppl}`",
        f"- used_windows: `{agg['used_windows']}`",
        f"- token_count_sum: `{agg['token_count_sum']}`",
        f"- total_nll_sum: `{agg['total_nll_sum']}`",
        "",
        "## Invariants",
        "",
        "- Aggregate recomputable as `exp(sum(total_nll)/sum(token_count))`.",
        "- Original EigenAttn scorer source files were not modified.",
        "- Per-window atomic write: `.tmp` → fsync → rename → `.DONE`.",
        "",
    ]
    (out_dir / "PPL_RAW_OUTPUT_INTERFACE_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(agg, ensure_ascii=False))


if __name__ == "__main__":
    main()
