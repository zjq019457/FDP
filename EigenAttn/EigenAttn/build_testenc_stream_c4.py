#!/usr/bin/env python3
import os, json, glob, gzip, argparse, random
import torch
from transformers import AutoTokenizer

def _iter_json_texts_gz(fp, text_key="text", read_size=1 << 20):
    dec = json.JSONDecoder()
    buf = ""
    def skip_seps(s, i, n):
        while i < n and s[i] in " \t\r\n,\ufeff\x00":
            i += 1
        if i < n and s[i] == "\x1e":
            i += 1
        return i

    with gzip.open(fp, "rt", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(read_size)
            if not chunk:
                break
            buf += chunk
            i = 0
            while True:
                n = len(buf)
                i = skip_seps(buf, i, n)
                if i >= n:
                    break
                if buf[i] not in "{[":
                    nxt = buf.find("{", i)
                    if nxt == -1:
                        buf = ""
                        i = 0
                        break
                    i = nxt
                try:
                    obj, j = dec.raw_decode(buf, i)
                except json.JSONDecodeError as e:
                    if e.pos >= n - 32:
                        break
                    nxt = buf.find("{", i + 1)
                    if nxt == -1:
                        break
                    i = nxt
                    continue
                i = j
                if isinstance(obj, dict):
                    v = obj.get(text_key, None)
                    if isinstance(v, str):
                        t = v.strip()
                        if t:
                            yield t
            buf = buf[i:]

def build_doc_pool(data_dir, glob_pat, scan_docs, seed):
    files = sorted(glob.glob(os.path.join(data_dir, glob_pat)))
    if not files:
        raise FileNotFoundError(f"No files matched {glob_pat} under {data_dir}")

    rng = random.Random(seed)
    texts = []
    seen = 0
    for fp in files:
        for t in _iter_json_texts_gz(fp, text_key="text"):
            texts.append(t)
            seen += 1
            if scan_docs > 0 and seen >= scan_docs:
                break
        if scan_docs > 0 and seen >= scan_docs:
            break

    # shuffle for reproducibility (optional but helpful)
    rng.shuffle(texts)
    return texts, {"seen_docs": seen, "nfiles": len(files), "file0": files[0]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--glob", required=True)
    ap.add_argument("--scan_docs", type=int, default=0, help="0 means read all docs from matched files")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--target_tokens", type=int, default=4194304)  # 2048*2048
    ap.add_argument("--add_special_tokens", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tok_dir, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos = tok.eos_token_id

    pool, info = build_doc_pool(args.data_dir, args.glob, args.scan_docs, args.seed)
    print("[Pool]", info, "docs_in_pool=", len(pool))

    ids = []
    for t in pool:
        enc = tok(t, add_special_tokens=bool(args.add_special_tokens)).input_ids
        if not enc:
            continue
        ids.extend(enc)
        if eos is not None:
            ids.append(eos)
        if len(ids) >= args.target_tokens + 1:
            break

    if len(ids) < args.target_tokens + 1:
        raise RuntimeError(f"Not enough tokens: got {len(ids)} < target_tokens+1={args.target_tokens+1}. "
                           f"Need more docs/files.")

    testenc = torch.tensor(ids[:args.target_tokens], dtype=torch.long).unsqueeze(0)  # [1, T]
    torch.save(testenc, args.out)
    print("[Saved]", args.out, "shape=", tuple(testenc.shape), "tokens=", testenc.numel())

if __name__ == "__main__":
    main()
