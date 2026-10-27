#!/usr/bin/env python3
"""EigenAttn LongBench prediction (real generate) for Dense / Uniform / FDP Llama-3."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
EIGEN = ROOT / "EigenAttn"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EIGEN))
from common import atomic_write_json, now_iso  # noqa: E402

# Focused English subset — full 38-task sweep can follow once this path is validated.
DEFAULT_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "triviaqa",
    "passage_retrieval_en",
]


def _load_model(model_dir: Path, tok_dir: Path, dtype):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=False, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    lr_json = model_dir / "low_rank_config.json"
    if lr_json.exists():
        from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn

        low = json.loads(lr_json.read_text())
        if isinstance(low, dict) and "low_rank_config" in low:
            low = low["low_rank_config"]
        lr_pt = model_dir / "low_rank_config.pt"
        if lr_pt.exists():
            import torch as _t
            packed = _t.load(str(lr_pt), map_location="cpu")
            if isinstance(packed, dict) and "low_rank_config" in packed:
                low = packed["low_rank_config"]
        cfg = AutoConfig.from_pretrained(str(tok_dir), attn_implementation="eager", local_files_only=True)
        model = LlamaForCausalLM_EigenAttn.from_pretrained(
            str(model_dir),
            config=cfg,
            low_rank_config=low,
            torch_dtype=dtype,
            device_map="cuda:0",
            local_files_only=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=dtype,
            device_map="cuda:0",
            local_files_only=True,
            attn_implementation="eager",
        )
    model.eval()
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--limit_per_task", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    args = ap.parse_args()

    import torch
    from tqdm import tqdm

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_map = json.loads((ROOT / "slim" / "configs" / "dataset2prompt.json").read_text())
    maxlen_map = json.loads((ROOT / "slim" / "configs" / "dataset2maxlen.json").read_text())

    dtype = torch.bfloat16
    model, tok = _load_model(Path(args.model), Path(args.tok), dtype)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    summary = {"method": args.method, "generated_at": now_iso(), "tasks": {}, "eval_executed": True}

    for task in tasks:
        path = data_dir / f"{task}.jsonl"
        if not path.exists():
            summary["tasks"][task] = {"skipped": True, "reason": "missing_jsonl"}
            continue
        prompt_fmt = prompt_map.get(task)
        if not prompt_fmt:
            summary["tasks"][task] = {"skipped": True, "reason": "no_prompt"}
            continue
        max_gen = int(maxlen_map.get(task, args.max_new_tokens))
        preds = []
        with path.open("r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        lines = lines[: args.limit_per_task]
        for i, ln in enumerate(tqdm(lines, desc=f"{args.method}:{task}")):
            obj = json.loads(ln)
            prompt = prompt_fmt.format(**obj)
            ids = tok(prompt, truncation=False, return_tensors="pt").input_ids
            if ids.shape[1] > args.max_len:
                half = args.max_len // 2
                # keep head+tail
                ids = torch.cat([ids[:, :half], ids[:, -half:]], dim=1)
            ids = ids.to(model.device)
            with torch.no_grad():
                out = model.generate(
                    ids,
                    max_new_tokens=max_gen,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tok.eos_token_id,
                )
            gen = out[0][ids.shape[1] :]
            pred = tok.decode(gen, skip_special_tokens=True)
            preds.append({
                "pred": pred,
                "answers": obj.get("answers"),
                "all_classes": obj.get("all_classes"),
                "length": obj.get("length"),
                "_id": obj.get("_id", i),
                "dataset": task,
                "language": obj.get("language", "en"),
            })
        pred_path = out_dir / f"{task}.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        summary["tasks"][task] = {"n": len(preds), "pred_path": str(pred_path), "max_gen": max_gen}

    atomic_write_json(out_dir / "PRED_SUMMARY.json", summary)
    print(json.dumps({"ok": True, "method": args.method, "out": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
