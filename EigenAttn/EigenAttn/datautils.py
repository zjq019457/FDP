import os
import glob
import pdb
import random
import numpy as np
import torch

from transformers import AutoTokenizer
from datasets import load_dataset


def set_seed(seed: int):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# -------------------------
# Local-file strict loader
# -------------------------
def _collect_local_files(data_dir: str):
    patterns = [
        "**/*.jsonl.gz", "**/*.json.gz",
        "**/*.jsonl", "**/*.json",
        "**/*.parquet",
        "**/*.txt.gz", "**/*.txt",
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pat), recursive=True))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"[LOCAL_DATA] No files found under: {data_dir}")
    return files


def _load_local_streaming(data_dir: str):
    files = _collect_local_files(data_dir)
    pick = files[0].lower()
    print(f"[LOCAL_DATA] loader from {data_dir} files={len(files)} example={files[0]}")

    if pick.endswith(".parquet"):
        return load_dataset("parquet", data_files=files, split="train", streaming=True)
    if any(pick.endswith(suf) for suf in [".json", ".jsonl", ".json.gz", ".jsonl.gz"]):
        return load_dataset("json", data_files=files, split="train", streaming=True)
    return load_dataset("text", data_files=files, split="train", streaming=True)


def _load_local_nonstream(data_dir: str):
    files = _collect_local_files(data_dir)
    pick = files[0].lower()
    print(f"[LOCAL_DATA] (nonstream) loader from {data_dir} files={len(files)} example={files[0]}")

    if pick.endswith(".parquet"):
        return load_dataset("parquet", data_files=files, split="train", streaming=False)
    if any(pick.endswith(suf) for suf in [".json", ".jsonl", ".json.gz", ".jsonl.gz"]):
        return load_dataset("json", data_files=files, split="train", streaming=False)
    return load_dataset("text", data_files=files, split="train", streaming=False)


def _get_tokenizer(model: str):
    # keep your original behavior
    if "mpt" in model.lower():
        return AutoTokenizer.from_pretrained(model, use_fast=True)
    return AutoTokenizer.from_pretrained(model, use_fast=False)


# -------------------------
# Existing loaders
# -------------------------
def get_pile(nsamples, seed, seqlen, model):
    print("get_pile")
    traindata = load_dataset(
        "json",
        data_files="/cpfs01/user/chenmengzhao/prompt_quantization/val.jsonl.zst",
        split="train"
    )
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata["text"][:1000]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, None


def get_alpaca(nsamples, seed, seqlen, model):
    print("get alpaca")
    ds = load_dataset("tatsu-lab/alpaca")
    ds = ds.remove_columns(["input", "output", "instruction"])
    traindata = ds["train"]

    tokenizer = _get_tokenizer(model)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def _find_wikitext_raw_files(local_root: str):
    """Resolve wiki.{train,test}.raw under WIKITEXT2_DATA_DIR (wikitext-2-raw-v1 unzip layout)."""
    root = os.path.abspath(local_root)
    candidates_train = [
        os.path.join(root, "wiki.train.raw"),
        os.path.join(root, "wikitext-2-raw-v1", "wiki.train.raw"),
    ]
    candidates_test = [
        os.path.join(root, "wiki.test.raw"),
        os.path.join(root, "wikitext-2-raw-v1", "wiki.test.raw"),
    ]
    train_path = next((p for p in candidates_train if os.path.isfile(p)), None)
    test_path = next((p for p in candidates_test if os.path.isfile(p)), None)
    if train_path and test_path:
        return train_path, test_path
    hits = glob.glob(os.path.join(root, "**", "wiki.train.raw"), recursive=True)
    t_train = hits[0] if hits else None
    hits_t = glob.glob(os.path.join(root, "**", "wiki.test.raw"), recursive=True)
    t_test = hits_t[0] if hits_t else None
    if t_train and t_test:
        return t_train, t_test
    raise FileNotFoundError(
        f"[wikitext2] WIKITEXT2_DATA_DIR={local_root!r}: could not find wiki.train.raw / wiki.test.raw. "
        f"Download https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip "
        f"and set WIKITEXT2_DATA_DIR to the folder that contains those two files."
    )


def _wikitext_concat_parquet_text(parquet_paths: list):
    """HF-export WikiText-2 shards: rows have column 'text'; join like load_dataset + '\\n\\n'.join."""
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "[wikitext2] reading *.parquet requires pandas and pyarrow; "
            "e.g. pip install pandas pyarrow"
        ) from e
    parts = []
    for fp in parquet_paths:
        df = pd.read_parquet(fp)
        if "text" not in df.columns:
            raise ValueError(f"[wikitext2] {fp!r} has no 'text' column: {list(df.columns)}")
        parts.append("\n\n".join(df["text"].astype(str).tolist()))
    return "\n\n".join(parts)


def _find_wikitext_parquet_splits(local_root: str):
    """HF cache / save_to_disk layout: train-*.parquet, test-*.parquet (or validation-*.parquet)."""
    root = os.path.abspath(local_root)
    train_files = sorted(glob.glob(os.path.join(root, "*train*.parquet")))
    test_files = sorted(glob.glob(os.path.join(root, "*test*.parquet")))
    if not test_files:
        test_files = sorted(glob.glob(os.path.join(root, "*validation*.parquet")))
    if train_files and test_files:
        return train_files, test_files
    return None, None


def get_wikitext2(nsamples, seed, seqlen, model):
    print("get_wikitext2")
    tokenizer = _get_tokenizer(model)

    local_root = os.environ.get("WIKITEXT2_DATA_DIR", "").strip()
    if local_root:
        train_blob = None
        test_blob = None
        try:
            train_path, test_path = _find_wikitext_raw_files(local_root)
            print(f"[wikitext2] OFFLINE/local raw: train={train_path} test={test_path}")
            with open(train_path, "r", encoding="utf-8") as f:
                train_blob = f.read()
            with open(test_path, "r", encoding="utf-8") as f:
                test_blob = f.read()
        except FileNotFoundError:
            train_pq, test_pq = _find_wikitext_parquet_splits(local_root)
            if train_pq and test_pq:
                print(f"[wikitext2] OFFLINE/local parquet: train={train_pq} test={test_pq}")
                train_blob = _wikitext_concat_parquet_text(train_pq)
                test_blob = _wikitext_concat_parquet_text(test_pq)
            else:
                raise FileNotFoundError(
                    f"[wikitext2] WIKITEXT2_DATA_DIR={local_root!r}: need one of:\n"
                    f"  (1) wiki.train.raw + wiki.test.raw (wikitext-2-raw-v1 unzip), or\n"
                    f"  (2) train-*.parquet + test-*.parquet (HF export; test may be validation-*.parquet)."
                ) from None
        trainenc = tokenizer(train_blob, return_tensors="pt")
        testenc = tokenizer(test_blob, return_tensors="pt")
    else:
        offline = str(os.environ.get("HF_DATASETS_OFFLINE", "")).lower() in ("1", "true", "yes")
        if offline:
            raise RuntimeError(
                "WikiText-2: HF_DATASETS_OFFLINE is set but WIKITEXT2_DATA_DIR is empty. "
                "Unzip wikitext-2-raw-v1 (wiki.train.raw + wiki.test.raw) and either:\n"
                "  export WIKITEXT2_DATA_DIR=/path/to/that/folder\n"
                "or pass --wikitext2_dir to sanity_ppl_dual_protocol.py."
            )
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, model):
    print("get_ptb")
    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")

    tokenizer = _get_tokenizer(model)
    trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


# -------------------------
# C4 loaders (offline-capable)
# -------------------------
def get_c4(nsamples, seed, seqlen, model):
    """
    Offline-first C4 loader.

    If env C4_DATA_DIR is set, load local shards from that directory (json/jsonl/parquet/text).
    Otherwise fallback to HF hub 'allenai/c4' (will fail if offline).
    """
    print("get_c4")
    tokenizer = _get_tokenizer(model)

    c4_local = os.environ.get("C4_DATA_DIR", "").strip()
    val_blocks = int(os.environ.get("C4_VAL_BLOCKS", "64"))  # default smaller than 256 to avoid OOM

    if c4_local:
        # NOTE: local C4 usually doesn't have official train/validation split in one folder.
        # We'll sample train/val from the same stream for calibration purposes.
        print(f"[C4] Using LOCAL shards: {c4_local}")
        traindata_stream = _load_local_streaming(c4_local)
        valdata_stream = _load_local_streaming(c4_local)
        traindata = traindata_stream
        valdata = valdata_stream
        # streaming: cannot len()/random index. We'll iterate to get random long-enough samples.
        def _sample_one_long_text(it):
            while True:
                ex = next(it)
                if isinstance(ex, dict):
                    txt = ex.get("text", None)
                    if txt is None:
                        # fallback: pick first string field
                        for v in ex.values():
                            if isinstance(v, str):
                                txt = v
                                break
                else:
                    txt = ex
                if not txt:
                    continue
                enc = tokenizer(txt, return_tensors="pt")
                if enc.input_ids.shape[1] > seqlen:
                    return enc
        train_it = iter(traindata)
        val_it = iter(valdata)
    else:
        print("[C4] Using HF hub: allenai/c4 (offline will fail)")
        traindata = load_dataset(
            "allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train"
        )
        valdata = load_dataset(
            "allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation"
        )
        def _sample_one_long_text(_):
            while True:
                i = random.randint(0, len(traindata) - 1)
                enc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if enc.input_ids.shape[1] > seqlen:
                    return enc
        train_it = None
        val_it = None

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        if c4_local:
            trainenc = _sample_one_long_text(train_it)
        else:
            trainenc = _sample_one_long_text(None)

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Build valenc = [1, val_blocks*seqlen]
    random.seed(0)
    valenc_chunks = []
    for _ in range(val_blocks):
        if c4_local:
            tmp = _sample_one_long_text(val_it)
        else:
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
                if tmp.input_ids.shape[1] > seqlen:
                    break

        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc_chunks.append(tmp.input_ids[:, i:j])

    valenc = torch.hstack(valenc_chunks)
    return trainloader, valenc


def get_ptb_new(nsamples, seed, seqlen, model):
    print("get_ptb_new")
    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model):
    """
    New C4 loader: offline-first.
    For local shards, builds valenc by concatenating many blocks like original.
    valenc size can be controlled by env C4_VAL_BLOCKS (default 64).
    """
    print("get_c4_new")
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    c4_local = os.environ.get("C4_DATA_DIR", "").strip()
    val_blocks = int(os.environ.get("C4_VAL_BLOCKS", "64"))

    if c4_local:
        print(f"[C4_NEW] Using LOCAL shards: {c4_local}")
        stream = _load_local_streaming(c4_local)
        it = iter(stream)

        def _next_long():
            while True:
                ex = next(it)
                if isinstance(ex, dict):
                    txt = ex.get("text", None)
                    if txt is None:
                        for v in ex.values():
                            if isinstance(v, str):
                                txt = v
                                break
                else:
                    txt = ex
                if not txt:
                    continue
                enc = tokenizer(txt, return_tensors="pt")
                if enc.input_ids.shape[1] >= seqlen:
                    return enc

        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            trainenc = _next_long()
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        # valenc: concatenate blocks
        random.seed(0)
        valenc_chunks = []
        for _ in range(val_blocks):
            tmp = _next_long()
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            valenc_chunks.append(tmp.input_ids[:, i:j])
        valenc = torch.hstack(valenc_chunks)
        return trainloader, valenc

    # fallback to hub (offline will fail)
    traindata = load_dataset(
        "allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train"
    )
    valdata = load_dataset(
        "allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation"
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (val_blocks * seqlen)]
    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=""):
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if "pile" in name:
        return get_pile(nsamples, seed, seqlen, model)
    if "ptb" in name:
        if "new" in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if "c4" in name:
        if "new" in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)
    if "mix" in name:
        wiki_train, wiki_val = get_wikitext2(nsamples // 2, seed, seqlen, model)
        c4_train, c4_val = get_c4(nsamples // 2, seed, seqlen, model)
        train = wiki_train + c4_train
        val = None
        return train, val
    if "alpaca" in name:
        train = get_alpaca(nsamples, seed, seqlen, model)
        return train, None
    raise ValueError(f"Unknown dataset name: {name}")
