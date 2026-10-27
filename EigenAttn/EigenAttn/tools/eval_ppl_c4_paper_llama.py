#!/usr/bin/env python3
"""
C4 perplexity for EigenAttn-compressed Llama (final_model/).

Matches EigenAttn_base ``main_eigen_attn.evaluate()`` when ``eval_ppl`` is set:
  - C4 tensor from EigenAttn_base ``datautils.get_loaders("c4", ...)`` (offline shards).
  - Llama forward: ``outputs = lm.model.model(batch)``, ``logits = lm.model.lm_head(outputs[0])``.
  - Loss: ``CrossEntropyLoss()`` on flattened logits/labels (no logit clamp).
  - PPL: ``exp(sum(loss_i * seqlen) / (nsamples * seqlen))`` with ``nsamples = testenc.numel() // seqlen``.

Requires sibling checkout ``EigenAttn/EigenAttn_base`` (only ``datautils.py`` is loaded from it via importlib).

If ``final_model/generation_config.json`` mistakenly contains ``low_rank_config``, transformers can crash when
logging ``GenerationConfig``. This script strips that key (one-time backup to
``generation_config.json.bak_before_eval_ppl``) before ``from_pretrained``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig

from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn  # noqa: E402

_INNER = Path(__file__).resolve().parents[1]
_BASE = _INNER.parent / "EigenAttn_base"

# Repo ``models`` must resolve to this tree (EigenAttn/EigenAttn), not EigenAttn_base.
# Never insert EigenAttn_base *before* _INNER — it shadows ``models.LMClass`` and breaks Llama load.
if str(_INNER) not in sys.path:
    sys.path.insert(0, str(_INNER))

_du_path = _BASE / "datautils.py"
if not _du_path.is_file():
    raise SystemExit(f"EigenAttn_base datautils not found: {_du_path}")
_spec = importlib.util.spec_from_file_location("_eigenattn_base_datautils", _du_path)
_base_du = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_base_du)
get_loaders = _base_du.get_loaders

def _load_low_rank_list(compressed_model: Path, override: str) -> list:
    if override.strip():
        p = Path(override.strip())
        if not p.is_file():
            raise FileNotFoundError(f"low_rank_config_path not found: {p}")
        if p.suffix == ".pt":
            raw = torch.load(p, map_location="cpu")
        else:
            raw = json.loads(p.read_text(encoding="utf-8"))
    else:
        j = compressed_model / "low_rank_config.json"
        t = compressed_model / "low_rank_config.pt"
        if j.is_file():
            raw = json.loads(j.read_text(encoding="utf-8"))
        elif t.is_file():
            raw = torch.load(t, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"Need low_rank_config.json or .pt under {compressed_model}, or pass --low_rank_config_path"
            )
    if isinstance(raw, torch.Tensor):
        raw = raw.tolist()
    return raw


def _sanitize_generation_config_json(cmp_dir: Path, *, allow_disk_fix: bool) -> None:
    """Remove non-generation keys (e.g. ``low_rank_config``) so HF can load+log safely."""
    path = cmp_dir / "generation_config.json"
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if "low_rank_config" not in data:
        return
    if not allow_disk_fix:
        raise RuntimeError(
            f"{path} contains invalid key 'low_rank_config'. Omit --no_fix_generation_config_json "
            f"(default writes backup then strips key), or edit the file manually."
        )
    bak = cmp_dir / "generation_config.json.bak_before_eval_ppl"
    if not bak.is_file():
        bak.write_bytes(path.read_bytes())
    del data["low_rank_config"]
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_compressed_llama_eigenattn(
    *,
    model_base: str,
    compressed_model: str,
    cache_dir: str,
    attn_implementation: str,
    low_rank_config_path: str,
    fix_generation_config_json: bool,
) -> LlamaForCausalLM_EigenAttn:
    """Load CMP checkpoint; optionally strip bogus keys from ``generation_config.json`` on disk."""
    cmp_dir = Path(compressed_model)
    _sanitize_generation_config_json(cmp_dir, allow_disk_fix=fix_generation_config_json)
    config = AutoConfig.from_pretrained(
        model_base,
        attn_implementation=attn_implementation,
        cache_dir=cache_dir,
    )
    low_rank_config = _load_low_rank_list(cmp_dir, low_rank_config_path)
    gc_logger = logging.getLogger("transformers.generation.configuration_utils")
    prev = gc_logger.level
    gc_logger.setLevel(logging.ERROR)
    try:
        model = LlamaForCausalLM_EigenAttn.from_pretrained(
            str(cmp_dir),
            config=config,
            low_rank_config=low_rank_config,
            device_map="cpu",
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
        )
    finally:
        gc_logger.setLevel(prev)
    return model


@torch.no_grad()
def eval_ppl_c4_eigenattn_base_llama(
    model: nn.Module,
    testenc: torch.Tensor,
    device: torch.device,
    seqlen: int,
    limit: int,
) -> tuple[float, dict]:
    """Mirror EigenAttn_base ``evaluate()`` Llama branch for dataset ``c4``."""
    nsamples = testenc.numel() // seqlen
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()
    nlls: list[torch.Tensor] = []
    for i in tqdm(range(nsamples), desc="c4-ppl"):
        batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(device)
        outputs = model.model(batch)
        hidden_states = outputs[0]
        logits = model.lm_head(hidden_states)
        shift_logits = logits[:, :-1, :]
        shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:].to(model.lm_head.weight.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        nlls.append(loss.float() * seqlen)
        if limit >= 0 and i == limit:
            break

    model.config.use_cache = use_cache
    if len(nlls) == 0:
        return float("nan"), {"nsamples": nsamples, "used_chunks": 0}

    # Base code uses full ``nsamples`` in denominator even when ``limit`` stops early;
    # keep that behaviour only when we consumed all chunks (typical ``limit=-1``).
    used = len(nlls)
    if limit < 0 and used == nsamples:
        denom = float(nsamples * seqlen)
    else:
        denom = float(used * seqlen)
    ppl = torch.exp(torch.stack(nlls).sum() / denom)
    return ppl.item(), {"nsamples": nsamples, "used_chunks": used, "denom_tokens": int(denom), "seqlen": seqlen}


def main() -> None:
    if not _BASE.is_dir():
        raise SystemExit(f"EigenAttn_base not found at {_BASE} (clone sibling next to EigenAttn/).")

    ap = argparse.ArgumentParser(description="C4 PPL — EigenAttn_base data + paper Llama forward path")
    ap.add_argument("--model_base", required=True, help="Dense HF dir for tokenizer/config (e.g. Llama-3)")
    ap.add_argument("--compressed_model", required=True, help="final_model directory with weights + low_rank_config")
    ap.add_argument("--attn_implementation", default="eager")
    ap.add_argument("--cache_dir", default="./HF_cache")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--limit", type=int, default=-1, help="Max chunk index inclusive; -1 = all")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="First arg to get_loaders(c4); controls train calib size only, not val tensor length",
    )
    ap.add_argument("--c4_data_dir", default=os.environ.get("C4_DATA_DIR", "/diskE/jiaqizhang/c4-en/en"))
    ap.add_argument("--c4_train_glob", default="c4-train.*.json.gz")
    ap.add_argument("--c4_val_glob", default="c4-validation.*.json.gz")
    ap.add_argument("--c4_pool_docs", type=int, default=200_000)
    ap.add_argument("--c4_scan_lines", type=int, default=200_000)
    ap.add_argument("--c4_n_val_chunks", type=int, default=256)
    ap.add_argument("--c4_val_seed", type=int, default=0)
    ap.add_argument("--c4_train_file", default=None)
    ap.add_argument("--c4_val_file", default=None)
    ap.add_argument("--low_rank_config_path", default="", help="Optional explicit low_rank_config.json or .pt")
    ap.add_argument(
        "--no_fix_generation_config_json",
        action="store_true",
        help="Do not strip invalid keys from final_model/generation_config.json (fails if low_rank_config is embedded)",
    )
    args = ap.parse_args()

    _, testenc = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
        model=args.model_base,
        c4_data_dir=args.c4_data_dir,
        c4_train_glob=args.c4_train_glob,
        c4_val_glob=args.c4_val_glob,
        c4_pool_docs=args.c4_pool_docs,
        c4_scan_lines=args.c4_scan_lines,
        c4_n_val_chunks=args.c4_n_val_chunks,
        c4_val_seed=args.c4_val_seed,
        c4_train_file=args.c4_train_file,
        c4_val_file=args.c4_val_file,
    )

    model = load_compressed_llama_eigenattn(
        model_base=args.model_base,
        compressed_model=args.compressed_model,
        cache_dir=args.cache_dir,
        attn_implementation=args.attn_implementation,
        low_rank_config_path=args.low_rank_config_path,
        fix_generation_config_json=not args.no_fix_generation_config_json,
    )
    dev = torch.device(args.device)
    dt = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    model = model.to(device=dev, dtype=dt)

    ppl, info = eval_ppl_c4_eigenattn_base_llama(model, testenc, dev, args.seqlen, args.limit)
    print(f"C4 PPL (EigenAttn_base protocol, Llama): {ppl:.6f}  info={info}")


if __name__ == "__main__":
    main()
