#!/usr/bin/env python3
"""Diagnostic Wiki/C4 PPL for frozen U011 Uniform vs FDP.

The resource gap stays +4.42%. This does not retune epsilon or ranks and is
not a matched-resource formal comparison. Both arms share the frozen MPT
tokenizer windows in MPT7B_EVAL_INPUT_BINDING_V1.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


BINDING = Path("/diskE/jiaqizhang/MPT7B_FDP_EB_CAMPAIGN_V1/input_bindings/MPT7B_EVAL_INPUT_BINDING_V1.json")
MODEL = Path("/diskF/jiaqizhang/models/新建文件夹")
CAMPAIGN_E = Path("/diskE/jiaqizhang/MPT7B_FDP_EB_CAMPAIGN_V1")
CAMPAIGN_F = Path("/diskF/jiaqizhang/MPT7B_FDP_EB_CAMPAIGN_V1/resource_stages")
TIER_PATHS = {
    "U011": {
        "uniform_dir": CAMPAIGN_F / "EB0011_UNIFORM_RESOURCE_ATTEMPT2",
        "uniform_schedule": "MPT7B_UNIFORM_EB0011_STRICT_RESOURCE_SCHEDULE_V1.json",
        "fdp_lock": CAMPAIGN_E / "U011/04_SCHEDULE/U011_FDP_RANK_LOCK.json",
        "relaxed_lock": CAMPAIGN_E / "U011/10_RELAXED_UNIFORM/RELAXED_UNIFORM_RANK_LOCK.json",
        "fdp_out": CAMPAIGN_E / "U011/09_DIAGNOSTIC_QUALITY/fdp",
        "uniform_out": CAMPAIGN_E / "U011/09_DIAGNOSTIC_QUALITY/uniform",
    },
    "U015": {
        "uniform_dir": CAMPAIGN_F / "EB0015_UNIFORM_RESOURCE_ATTEMPT1",
        "uniform_schedule": "MPT7B_UNIFORM_EB0015_STRICT_RESOURCE_SCHEDULE_V1.json",
        "fdp_lock": CAMPAIGN_E / "U015/04_SCHEDULE/U015_FDP_RANK_LOCK.json",
        "relaxed_lock": CAMPAIGN_E / "U015/10_RELAXED_UNIFORM/RELAXED_UNIFORM_RANK_LOCK.json",
        "fdp_out": CAMPAIGN_E / "U015/09_DIAGNOSTIC_QUALITY/fdp",
        "uniform_out": CAMPAIGN_E / "U015/09_DIAGNOSTIC_QUALITY/uniform",
    },
    "U019": {
        "uniform_dir": CAMPAIGN_F / "EB0019_UNIFORM_RESOURCE_ATTEMPT1",
        "uniform_schedule": "MPT7B_UNIFORM_EB0019_STRICT_RESOURCE_SCHEDULE_V1.json",
        "fdp_lock": CAMPAIGN_E / "U019/04_SCHEDULE/U019_FDP_RANK_LOCK.json",
        "relaxed_lock": CAMPAIGN_E / "U019/10_RELAXED_UNIFORM/RELAXED_UNIFORM_RANK_LOCK.json",
        "fdp_out": CAMPAIGN_E / "U019/09_DIAGNOSTIC_QUALITY/fdp",
        "uniform_out": CAMPAIGN_E / "U019/09_DIAGNOSTIC_QUALITY/uniform",
    },
}
WIKI = Path("/home/bai-project/jiaqizhang/wikitext2")
BASE_DATAUTILS = Path("/home/bai-project/jiaqizhang/EigenAttn/EigenAttn_base/datautils.py")
PPL_RUNNER = Path("/home/bai-project/jiaqizhang/EigenAttn/rebuttal_legacy_reproduction/runners/ppl_instrumented.py")
DECOMP = Path("/home/bai-project/jiaqizhang/EigenAttn/EigenAttn_base/models/decompose_modules.py")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tensor_sha(tensor: torch.Tensor) -> str:
    data = tensor.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def load_bound_windows(binding: dict) -> dict[str, list[dict]]:
    from transformers import AutoTokenizer

    os.environ.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, use_fast=True, trust_remote_code=False)
    if type(tokenizer).__name__ != binding["tokenizer"]["class"] or tokenizer.eos_token_id != binding["tokenizer"]["eos_token_id"]:
        raise SystemExit("tokenizer does not match the frozen eval binding")
    datautils = load_module("mpt_base_datautils_ppl", BASE_DATAUTILS)
    datautils._load_tokenizer = lambda _model: tokenizer
    _, testenc = datautils.get_wikitext2(128, 2, 2048, str(MODEL), data_dir=str(WIKI))
    test_ids = testenc.input_ids if hasattr(testenc, "input_ids") else testenc
    protocol = binding["ppl_protocol"]
    wiki_meta = protocol["wiki_windows"]
    n_wiki = int(test_ids.numel() // 2048)
    if n_wiki != protocol["wiki_window_count"] or n_wiki != len(wiki_meta):
        raise SystemExit(f"wiki window count {n_wiki} != binding {protocol['wiki_window_count']}")
    wiki = []
    for i, meta in enumerate(wiki_meta):
        batch = test_ids[:, i * 2048 : (i + 1) * 2048].contiguous()
        digest = tensor_sha(batch)
        if digest != meta["input_ids_sha256"] or meta["window_id"] != f"wiki_{i:05d}":
            raise SystemExit(f"wiki window {i} does not match the frozen binding")
        wiki.append({
            "window_id": meta["window_id"],
            "source_record_id": meta["source_record_id"],
            "input_ids": batch,
            "dataset": "wikitext2",
            "split": "test",
        })
    ppl = load_module("mpt_ppl_window_loader", PPL_RUNNER)
    c4_raw = ppl._load_c4_windows(Path(protocol["c4_source"]), tokenizer, 2048)
    c4_meta = protocol["c4_windows"]
    if len(c4_raw) != protocol["c4_window_count"] or len(c4_raw) != len(c4_meta):
        raise SystemExit("c4 window count does not match the frozen binding")
    c4 = []
    for raw, meta in zip(c4_raw, c4_meta):
        digest = tensor_sha(raw["input_ids"])
        if digest != meta["input_ids_sha256"] or raw["window_id"] != meta["window_id"]:
            raise SystemExit(f"c4 window {raw['window_id']} does not match the frozen binding")
        c4.append(raw)
    return {"wikitext2": wiki, "c4": c4}


def realize(arm: str, device: torch.device, paths: dict) -> tuple[torch.nn.Module, str]:
    uniform_dir = paths["uniform_dir"]
    if arm == "uniform":
        report = json.loads((uniform_dir / paths["uniform_schedule"]).read_text())
        schedule = report["rank_schedule"]
        rank_sha = report["rank_schedule_sha256"]
    elif arm == "fdp":
        report = json.loads(paths["fdp_lock"].read_text())
        schedule = report["rank_schedule"]
        rank_sha = report["rank_schedule_sha256"]
        if report.get("quality_used_to_select") is not False:
            raise SystemExit("FDP lock is not quality-blind")
    elif arm == "relaxed_uniform":
        report = json.loads(paths["relaxed_lock"].read_text())
        schedule = report["rank_schedule"]
        rank_sha = report["rank_schedule_sha256"]
        if report.get("quality_used_to_select") is not False:
            raise SystemExit("relaxed Uniform lock is not quality-blind")
        if int(report["predicted_kv_cache_bytes"]) <= int(report["frozen_fdp_b_live"]):
            raise SystemExit("relaxed Uniform is not above FDP")
    else:
        raise SystemExit(arm)
    if len(schedule) != 32:
        raise SystemExit("schedule is not 32 layers")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    decomp = load_module("mpt_decomp_ppl", DECOMP)
    for li, (rkq, rv) in enumerate(schedule):
        basis_path = uniform_dir / f"layer_{li:02d}_basis_and_uniform_rank.pt"
        basis = torch.load(basis_path, map_location="cpu", weights_only=True)
        layer_report = json.loads((uniform_dir / f"layer_{li:02d}_resource_search.json").read_text())
        if hashlib.sha256(basis_path.read_bytes()).hexdigest() != layer_report["basis_tensor_file_sha256"]:
            raise SystemExit(f"layer {li} basis file hash changed")
        if arm == "uniform" and [int(basis["selected_rank_kq"]), int(basis["selected_rank_v"])] != [int(rkq), int(rv)]:
            raise SystemExit(f"layer {li} stored uniform rank disagrees with the schedule")
        old = model.transformer.blocks[li]
        new = decomp.MptBlockEigenAttn(
            old, SimpleNamespace(), basis["basis_kq"].to(dtype=old.attn.Wqkv.weight.dtype), int(rkq),
            basis["basis_v"].to(dtype=old.attn.Wqkv.weight.dtype), int(rv), model.config,
        ).to(dtype=torch.bfloat16).eval()
        if tuple(new.attn.Wqkv.weight.shape) != (2 * int(rkq) + int(rv), 4096):
            raise SystemExit(f"layer {li} factor shape disagrees with the frozen rank")
        object.__setattr__(new.attn, "Wqkv_low", new.attn.Wqkv)
        object.__setattr__(new.attn, "out_proj_up", new.attn.out_proj)
        model.transformer.blocks[li] = new
        del basis, old, new
    model.eval().to(device)
    return model, rank_sha


def score_dataset(model, windows: list[dict], dataset: str, device: torch.device, out_dir: Path, rank_sha: str) -> dict:
    loss_fct = torch.nn.CrossEntropyLoss()
    seqlen = 2048
    total = 0.0
    used = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, window in enumerate(windows):
        rec_path = out_dir / f"{window['window_id']}.json"
        if rec_path.exists():
            rec = json.loads(rec_path.read_text())
            if rec.get("rank_schedule_sha256") == rank_sha and rec.get("skipped") is False:
                total += float(rec["total_nll"])
                used += 1
                continue
        batch = window["input_ids"].to(device)
        if batch.shape != (1, seqlen):
            raise SystemExit(f"bad window shape {batch.shape}")
        with torch.inference_mode():
            logits = model(
                input_ids=batch,
                attention_mask=torch.ones_like(batch),
                use_cache=False,
                return_dict=True,
            ).logits
            shift_logits = logits[:, :-1, :]
            shift_labels = batch[:, 1:]
            flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
            if dataset == "wikitext2":
                flat_logits = flat_logits.float()
            loss = loss_fct(flat_logits, shift_labels.reshape(-1))
            loss_f = loss.float()
            if not torch.isfinite(loss_f).all():
                raise SystemExit(f"non-finite loss at {window['window_id']}")
            total_nll = float((loss_f * seqlen).item())
            mean_nll = float(loss_f.item())
        rec = {
            "window_id": window["window_id"],
            "dataset": dataset,
            "input_ids_sha256": tensor_sha(window["input_ids"]),
            "token_count": seqlen,
            "total_nll": total_nll,
            "mean_nll": mean_nll,
            "skipped": False,
            "rank_schedule_sha256": rank_sha,
            "cross_entropy": "float32 logits" if dataset == "wikitext2" else "native logits dtype",
        }
        tmp = rec_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, rec_path)
        total += total_nll
        used += 1
        if (i + 1) % 10 == 0 or i + 1 == len(windows):
            ppl = torch.exp(torch.tensor(total / (used * seqlen))).item()
            print(f"[{dataset}] {i+1}/{len(windows)} running_ppl={ppl:.6f}", flush=True)
        del logits, loss
    if used != len(windows):
        raise SystemExit(f"used {used} of {len(windows)}")
    ppl = float(torch.exp(torch.tensor(total / (used * seqlen))).item())
    return {
        "dataset": dataset,
        "ppl": ppl,
        "total_nll_sum": total,
        "token_count_sum": used * seqlen,
        "used_windows": used,
        "log_ppl": total / (used * seqlen),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="U011", choices=["U011", "U015", "U019"])
    ap.add_argument("--arm", required=True, choices=["uniform", "fdp", "relaxed_uniform"])
    ap.add_argument("--memory-fraction", type=float, default=0.40)
    args = ap.parse_args()
    paths = TIER_PATHS[args.tier]
    binding = json.loads(BINDING.read_text())
    windows = load_bound_windows(binding)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device("cuda:0")
    free_b, total_b = torch.cuda.mem_get_info(device)
    if free_b < 36 * 1024**3:
        raise SystemExit(f"GPU free {free_b} < 36 GiB")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(2)
    model, rank_sha = realize(args.arm, device, paths)
    if args.arm == "relaxed_uniform":
        arm_dir = paths["relaxed_lock"].parent
    elif args.arm == "fdp":
        arm_dir = paths["fdp_out"]
    else:
        arm_dir = paths["uniform_out"]
    wiki = score_dataset(model, windows["wikitext2"], "wikitext2", device, arm_dir / "wiki", rank_sha)
    c4 = score_dataset(model, windows["c4"], "c4", device, arm_dir / "c4", rank_sha)
    summary = {
        "schema": f"MPT7B_{args.tier}_DIAGNOSTIC_PPL_V1",
        "tier": args.tier,
        "arm": args.arm,
        "status": "PPL_COMPLETE",
        "experiment": "RELAXED_UNIFORM_ABOVE_FDP" if args.arm == "relaxed_uniform" else "UNMATCHED_RESOURCE_DIAGNOSTIC",
        "formal_matched_quality": False,
        "quality_used_to_select_schedule": False,
        "resource_note": "frozen FDP versus its original Uniform; relaxed Uniform is a separate higher-resource schedule",
        "rank_schedule_sha256": rank_sha,
        "wiki": wiki,
        "c4": c4,
        "J": 0.5 * (wiki["log_ppl"] + c4["log_ppl"]),
        "gpu_free_at_start_bytes": int(free_b),
        "gpu_total_bytes": int(total_b),
    }
    out = arm_dir / "PPL_SUMMARY.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"tier": args.tier, "arm": args.arm, "wiki_ppl": wiki["ppl"], "c4_ppl": c4["ppl"], "J": summary["J"], "rank_sha": rank_sha}, indent=2), flush=True)


if __name__ == "__main__":
    main()
