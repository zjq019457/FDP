#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


BRANCH = Path("/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1")
OLD = Path("/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1")
REPO = Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/Palu")
MODEL = Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/models/Llama-2-7b-hf")
DATA = Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/data")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def current_schedule_shape(path: Path) -> dict:
    if not path.exists():
        return {"status": "MISSING", "path": str(path)}
    doc = json.loads(path.read_text())
    sel = doc.get("selection", {})
    lengths = sorted({len(v) for v in sel.values()})
    first = next(iter(sel.items())) if sel else None
    return {
        "status": "PRESENT",
        "path": str(path),
        "modules": len(sel),
        "rank_slots_per_module": lengths,
        "first_selection": first,
        "realized_rank": sum(sum(int(x) for x in v) for v in sel.values()),
        "schedule_sha256": sha(path),
    }


def main() -> None:
    BRANCH.mkdir(parents=True, exist_ok=True)
    old_uniform = OLD / "schedules/llama2_ratio_0.5_anchor_schedule.json"
    old_fdp = OLD / "exchange/calibration/ratio_0.5/fdp_native/final_schedule.json"
    source_files = {
        "official_repo_commit": "bb22666e2ef96707e8dd21d93fc00146c2e0d615",
        "official_compress": str(REPO / "compress.py"),
        "official_rank_search": str(REPO / "palu/rank_search.py"),
        "official_decomposition": str(REPO / "palu/decomposition.py"),
        "official_svd_linear": str(REPO / "palu/model/modules/svd_linear.py"),
        "official_ppl_eval": str(REPO / "run_ppl_eval.py"),
        "wiki_source": str(DATA / "wiki-test.hf_joined.raw"),
        "c4_source": str(DATA / "c4-validation.00000-of-00008.json.gz"),
    }
    source_hashes = {k: (sha(Path(v)) if Path(v).is_file() else v) for k, v in source_files.items() if k != "official_repo_commit"}
    lock = {
        "status": "PALU_OFFICIAL_PROTOCOL_LOCK_FROZEN",
        "campaign": "PALU_OFFICIAL_PROTOCOL_REPRO_V1",
        "official_repo": str(REPO),
        "official_repo_commit": source_files["official_repo_commit"],
        "model": str(MODEL),
        "model_architecture": "Llama-2-7B HF pretrained",
        "grouping": {
            "method": "G-LRD",
            "head_group_size": 4,
            "llama2_7b_num_attention_heads": 32,
            "llama2_7b_num_key_value_heads": 32,
            "official_groups_per_kv_projection": 8,
            "official_group_output_dim": 512,
        },
        "compression_mapping": {
            "repository_param_ratio_target": "retained native rank fraction",
            "0.7": {"retained_fraction": 0.7, "compression_fraction": 0.3},
            "0.5": {"retained_fraction": 0.5, "compression_fraction": 0.5},
            "0.3": {"retained_fraction": 0.3, "compression_fraction": 0.7},
        },
        "calibration": {
            "fisher_dataset": "WikiText-2",
            "fisher_samples": 2048,
            "fisher_sequence_length": 1024,
            "whiten_samples": 256,
            "whiten_sequence_length": 2048,
            "search_method": "official fisher_uniform for Automatic/Fisher reproduction",
        },
        "evaluation": {
            "sequence_length": 2048,
            "wiki_split": "WikiText-2 test",
            "c4_docs_scanned": 1100,
            "c4_chunks": 256,
            "dtype": "fp16",
            "scorer": "official run_ppl_eval CrossEntropyLoss reduction",
        },
        "source_hashes": source_hashes,
        "resource_formula": "sum native group ranks * sequence_length * 2 bytes; K/V aggregate",
        "old_anchor_schema_results_modified": False,
        "frozen_q_sha": "6f3b59993abbffbd932825b20ff036197c5be6ff63523ab46e94281bc24eecb1",
    }
    write_json(BRANCH / "PALU_OFFICIAL_PROTOCOL_LOCK.json", lock)

    diff = {
        "status": "CURRENT_VS_OFFICIAL_PROTOCOL_DIFF_IDENTIFIED",
        "current_branch": str(OLD),
        "official_branch": str(BRANCH),
        "differences": [
            {
                "item": "native group cardinality",
                "current": current_schedule_shape(old_uniform),
                "official": "8 groups per K/V projection for Llama-2-7B G-LRD group size 4",
                "classification": "PROTOCOL_MISMATCH",
            },
            {
                "item": "factorization path",
                "current": "manual reconstruction from precomputed anchor factor cache",
                "official": "official compress_model -> HeadwiseLowRankModule.from_linear_whiten",
                "classification": "PROTOCOL_MISMATCH",
            },
            {
                "item": "Fisher calibration",
                "current": "32-sample anchor calibration in current method card",
                "official": "official rank_search loads 2048 WikiText-2 samples at seqlen 1024",
                "classification": "PROTOCOL_MISMATCH",
            },
            {
                "item": "FDP current schedule shape",
                "current": current_schedule_shape(old_fdp),
                "official": "not yet ported; must use official 8-group native action space",
                "classification": "PENDING_OFFICIAL_NATIVE_PORT",
            },
            {
                "item": "optional CUDA dependency",
                "current": "official import failed: fast_hadamard_transform_cuda unavailable",
                "official": "non-quantized FP16 path does not call Hadamard; runner uses a pure-Python import fallback",
                "classification": "ENGINEERING_REPAIR_ONLY",
            },
        ],
        "old_results": "preserved as CURRENT_ANCHOR_SCHEMA_BRANCH and VALID_INTERNAL_COMPARISON only",
    }
    write_json(BRANCH / "audit/PROTOCOL_DIFF.json", diff)

    md = """# PaLu official-protocol audit\n\n## Authority\n\n- Paper: PaLu, ICLR 2025, Table 1/Table 8 and Appendix G.\n- Repository: pinned commit `bb22666e2ef96707e8dd21d93fc00146c2e0d615`.\n- Official conversion: `compress.py` -> `compress_model` -> `HeadwiseLowRankModule.from_linear_whiten`.\n\n## Resolved protocol\n\n- Model: Llama-2-7B HF pretrained.\n- G-LRD group size: 4.\n- Llama-2-7B has 32 attention/KV heads, therefore 8 native groups per K/V projection, each 512 output dimensions.\n- Repository ratio is retained native rank fraction: 0.7/0.5/0.3 correspond to 30%/50%/70% compression.\n- Fisher search: 2048 WikiText-2 samples, sequence length 1024.\n- Whitening: 256 WikiText-2 samples, sequence length 2048.\n- Quality: WikiText-2 test and C4 validation, sequence length 2048; C4 scans 1100 documents and uses 256 chunks.\n\n## Root cause of the old mismatch\n\nThe old anchor-schema branch uses two rank slots per K/V projection, while official Llama-2 G-LRD uses eight groups per K/V projection. It also bypasses the official `compress_model` factorization path and uses a 32-sample anchor calibration for its schedule. Uniform therefore diverges even before FDP is applied.\n\nThe old branch is preserved and is not overwritten. Its values remain valid only as an internal anchor-schema comparison.\n\n## Engineering repair\n\nThe official runner initially failed because the optional `fast_hadamard_transform_cuda` extension was unavailable. A pure-Python import fallback is used only for the non-quantized FP16 path, where the Hadamard routine is not invoked. Quantized results are outside this branch.\n\n## Gate order\n\n1. Full-rank identity.\n2. Official Uniform at retained fractions 0.7/0.5/0.3.\n3. Official Fisher/Automatic at the same fractions.\n4. Reproduction classification.\n5. Official-native FDP exchange calibration and formal matrix.\n"""
    (BRANCH / "PALU_OFFICIAL_PROTOCOL_AUDIT.md").write_text(md, encoding="utf-8")
    diff_md = """# Current anchor-schema vs official PaLu protocol\n\n| Item | Current branch | Official branch | Status |\n|---|---|---|---|\n| Llama-2 G-LRD groups per K/V projection | 2 slots | 8 groups (group size 4) | MISMATCH |\n| Factorization | precomputed anchor factor cache + manual module build | official whiten decomposition | MISMATCH |\n| Fisher calibration | 32 samples | 2048 samples, seqlen 1024 | MISMATCH |\n| Native action space for FDP | 2-slot schedule | official 8-group schedule | PENDING PORT |\n| Quantized Hadamard extension | unavailable | unused by FP16 path | ENGINEERING FALLBACK |\n\nNo current result is overwritten.\n"""
    (BRANCH / "PALU_PROTOCOL_DIFF_CURRENT_VS_OFFICIAL.md").write_text(diff_md, encoding="utf-8")
    state = {
        "status": "FULLRANK_IDENTITY_RUNNING",
        "campaign": "PALU_OFFICIAL_PROTOCOL_REPRO_V1",
        "official_protocol_lock": str(BRANCH / "PALU_OFFICIAL_PROTOCOL_LOCK.json"),
        "old_branch_preserved": True,
        "old_branch": str(OLD),
        "identity": "RUNNING",
        "official_uniform": {"0.7": "READY", "0.5": "READY", "0.3": "READY"},
        "official_fisher": {"0.7": "WAIT_IDENTITY", "0.5": "WAIT_IDENTITY", "0.3": "WAIT_IDENTITY"},
        "official_native_fdp": "WAIT_REPRODUCTION_GATE",
        "zero_shot": "WAIT_REPRODUCTION_GATE",
        "foreign_jobs_interrupted": 0,
        "gpu1_used": False,
        "source_dependency_repair": "fast_hadamard_transform import fallback only",
    }
    write_json(BRANCH / "PALU_PIPELINE_STATE.json", state)


if __name__ == "__main__":
    main()
