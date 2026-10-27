#!/usr/bin/env python3
"""Document-disjoint C4 tuning windows: same construction as the frozen tuning windows (Llama-2 tokenizer, no special tokens,
tokens [2048,4096) of a C4 validation document), but drawn from documents that are NOT among the 256 evaluation documents.
Gate: the frozen tuning windows must be reproducible from the same shard/row indexing (proves row indices refer to this shard)."""
import gzip, hashlib, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import l2x_lib as X

SHARD = pathlib.Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/data/c4-validation.00000-of-00008.json.gz")
EVAL = pathlib.Path("/diskD/jiaqizhang/llama3_8b_eb045_eval_minimal-20260502T170035Z-3-001/llama3_8b_eb045_eval_minimal/data/c4_ppl_selected_texts.jsonl")
STD = pathlib.Path("/diskF/jiaqizhang/fdp125_tuning_stage2_llama2_20261001/LLAMA2_FDP_STAGE3_TUNING_WINDOWS_V2_UNIQUE_IDS.jsonl")
N, SEED = 29, 20261015


def ids_sha(ids): return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/models/Llama-2-7b-hf", use_fast=False, local_files_only=True)
    docs = [json.loads(l)["text"] for l in gzip.open(SHARD, "rt", encoding="utf-8")]
    eval_rows = {int(json.loads(l)["row"]) for l in open(EVAL) if l.strip()}
    std = [json.loads(l) for l in open(STD) if l.strip()]
    std_c4 = [w for w in std if w["dataset"] == "c4"]
    gate = []
    for w in std_c4:
        r = int(w["source_record_id"]); t = docs[r]
        ok_text = hashlib.sha256(t.encode("utf-8")).hexdigest() == w["source_text_sha256"]
        ids = tok(t, add_special_tokens=False).input_ids[w["token_start"]:w["token_end"]]
        gate.append(ok_text and ids_sha(ids) == w["input_ids_sha256_int64le"])
    if not all(gate): raise SystemExit(f"GATE FAIL: frozen tuning windows not reproducible from shard rows ({sum(gate)}/{len(gate)})")
    std_rows = {int(w["source_record_id"]) for w in std_c4}
    if not std_rows <= eval_rows: print("note: some frozen tuning rows are not eval rows")
    order = np.random.Generator(np.random.PCG64(SEED)).permutation(len(docs))
    out = []
    for r in order:
        r = int(r)
        if r in eval_rows or r in std_rows: continue
        ids = tok(docs[r], add_special_tokens=False).input_ids
        if len(ids) < 4096: continue
        w = ids[2048:4096]
        out.append({"dataset": "c4", "window_id": f"c4_val_docdisjoint_row_{r}_tokens_2048_4096", "source_split": "validation_shard0_not_in_eval",
                    "source_record_id": str(r), "source_text_sha256": hashlib.sha256(docs[r].encode("utf-8")).hexdigest(),
                    "token_start": 2048, "token_end": 4096, "token_count": 2048, "input_ids": [int(x) for x in w], "input_ids_sha256_int64le": ids_sha(w),
                    "disjointness": "document-disjoint from all 256 evaluation documents and from the frozen tuning documents"})
        if len(out) == N: break
    p = X.ROOT / "data/C4_DOC_DISJOINT_TUNING_WINDOWS.jsonl"; p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists(): raise FileExistsError(p)
    p.write_text("".join(json.dumps(w) + "\n" for w in out))
    X.wjson(X.ROOT / "data/C4_DOC_DISJOINT_TUNING_WINDOWS.meta.json", {"shard": str(SHARD), "shard_sha256": X.sha_file(SHARD), "seed": SEED, "n": len(out),
            "eval_rows": len(eval_rows), "gate_frozen_windows_reproduced": f"{sum(gate)}/{len(gate)}", "file_sha256": X.sha_file(p), "at": X.now()})
    print("ok", len(out), f"gate {sum(gate)}/{len(gate)}")


if __name__ == "__main__":
    main()
