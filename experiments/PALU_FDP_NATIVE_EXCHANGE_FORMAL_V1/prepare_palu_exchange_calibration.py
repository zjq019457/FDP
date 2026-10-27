import gzip
import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


BRANCH = Path("/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1")
DATA = Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/data")
MODEL = Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/models/Llama-2-7b-hf")
C4 = BRANCH / "exchange_calibration" / "input" / "c4-train.00000-of-01024.json.gz"
OUT = BRANCH / "exchange_calibration" / "manifest.json"
SEED = 20260919
SEQ = 2048


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if not C4.exists():
        raise FileNotFoundError(C4)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, use_fast=True)
    wiki_path = DATA / "wiki-train.hf_joined.raw"
    wiki_ids = tokenizer(wiki_path.read_text(encoding="utf-8"), return_tensors="pt", add_special_tokens=False).input_ids[0]
    rng = random.Random(SEED)
    candidates = list(range(8192, len(wiki_ids) - SEQ - 1, SEQ))
    starts = sorted(rng.sample(candidates, 16))
    wiki_chunks = [
        {"chunk_id": f"wiki_train_{i:02d}", "token_start": s, "token_end": s + SEQ, "sequence_length": SEQ}
        for i, s in enumerate(starts)
    ]

    wanted = []
    next_stride = 503
    target_idx = 37
    with gzip.open(C4, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx != target_idx:
                continue
            doc = json.loads(line)
            ids = tokenizer(doc["text"], return_tensors="pt", add_special_tokens=False).input_ids[0]
            if ids.numel() >= SEQ:
                wanted.append({
                    "chunk_id": f"c4_train_shard0_doc_{idx:06d}",
                    "doc_index": idx,
                    "token_start": 0,
                    "token_end": SEQ,
                    "sequence_length": SEQ,
                    "text_sha256": hashlib.sha256(doc["text"].encode("utf-8")).hexdigest(),
                    "token_count": int(ids.numel()),
                    "url": doc.get("url"),
                })
                if len(wanted) == 16:
                    break
            target_idx += next_stride
    if len(wanted) != 16:
        raise RuntimeError(f"only found {len(wanted)} valid C4 train documents")

    doc = {
        "task": "PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1",
        "status": "FROZEN_EXCHANGE_CALIBRATION_MANIFEST",
        "seed": SEED,
        "sequence_length": SEQ,
        "split": {"wikitext": "train", "c4": "train"},
        "disjointness": {
            "native_phi_manifest": "wiki-test + C4 validation from PALU_FISHER_VS_FUNCTIONAL_MARGINAL_V1; disjoint by split/source",
            "formal_ppl": "WikiText-2 test + C4 validation; disjoint by split/source",
            "zero_shot": "task datasets; disjoint source",
            "longbench": "task datasets; disjoint source",
        },
        "j_definition": "0.5 * mean_Wiki_NLL + 0.5 * mean_C4_NLL",
        "scorer": {
            "cross_entropy": "torch.nn.functional.cross_entropy",
            "logits_reduction_dtype": "float32",
            "masking": "full 2048-token chunk; next-token loss excludes first target position",
            "padding": "none",
            "finite_value_policy": "reject non-finite loss",
        },
        "wikitext": {
            "source": str(wiki_path),
            "source_sha256": sha(wiki_path),
            "tokenizer_path": str(MODEL),
            "tokenizer_sha256": sha(MODEL / "tokenizer.json"),
            "chunks": wiki_chunks,
            "token_count": len(wiki_chunks) * (SEQ - 1),
        },
        "c4": {
            "source": str(C4),
            "source_sha256": sha(C4),
            "shard": "c4-train.00000-of-01024.json.gz",
            "chunks": wanted,
            "token_count": len(wanted) * (SEQ - 1),
        },
        "quality_used_to_select": False,
        "formal_quality_used_to_select": False,
    }
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    audit = {
        "status": "C4_TRAIN_SOURCE_VALID; MANIFEST_FROZEN",
        "source": str(C4),
        "sha256": sha(C4),
        "bytes": C4.stat().st_size,
        "gzip_jsonl_checked": True,
        "selected_c4_chunks": len(wanted),
        "selected_wiki_chunks": len(wiki_chunks),
        "manifest": str(OUT),
        "manifest_sha256": sha(OUT),
        "selection_quality_independent": True,
    }
    (BRANCH / "audit" / "C4_TRAIN_SOURCE_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    lock_path = BRANCH / "locks" / "PALU_LLAMA2_OFFICIAL_REPRO_LOCK.json"
    lock = json.loads(lock_path.read_text())
    lock["calibration_data_gate"] = {
        "status": "PASS",
        "wikitext_source": str(wiki_path),
        "c4_source": str(C4),
        "c4_source_sha256": sha(C4),
        "manifest": str(OUT),
        "manifest_sha256": sha(OUT),
        "manifest_frozen_before_exchange_quality": True,
    }
    lock["status"] = "CALIBRATION_MANIFEST_FROZEN_PRE_EXCHANGE"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    status_path = BRANCH / "audit" / "CAMPAIGN_STATUS.json"
    status = json.loads(status_path.read_text())
    status.update({"status": "CALIBRATION_MANIFEST_FROZEN_PRE_EXCHANGE", "exchange_calibration_manifest_frozen": True, "blocker": None})
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
