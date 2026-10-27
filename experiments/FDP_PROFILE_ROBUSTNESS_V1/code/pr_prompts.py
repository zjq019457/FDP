"""Build 8 WikiText-2 *train* passages, each >= 600 Llama-3 tokens (so 512 tokens with no padding), from the start of
consecutive article boundaries. Train split only (calibration source); never test/validation."""
import json
import pandas as pd
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/diskD/jiaqizhang/Llama-3", local_files_only=True, use_fast=False)
df = pd.read_parquet("/home/bai-project/jiaqizhang/wikitext2/train-00000-of-00001.parquet")
lines = [t for t in df["text"].tolist()]
# articles start with " = Title = " headings
arts, cur = [], []
for t in lines:
    if t.startswith(" = ") and not t.startswith(" = = ") and cur:
        arts.append("".join(cur)); cur = []
    cur.append(t)
arts.append("".join(cur))
out, meta = [], []
step = max(1, len(arts) // 40)
for k in range(0, len(arts), step):
    a = " ".join(arts[k].split())
    n = len(tok(a).input_ids)
    if n >= 600:
        out.append(a[:20000]); meta.append({"article_index": k, "tokens": n})
    if len(out) == 8: break
assert len(out) == 8
open("/diskF/jiaqizhang/FDP_PROFILE_ROBUSTNESS_V1/wikitext2_train_prompts.txt", "w").write("\n".join(out) + "\n")
json.dump(meta, open("/diskF/jiaqizhang/FDP_PROFILE_ROBUSTNESS_V1/wikitext2_train_prompts_meta.json", "w"), indent=1)
print(meta)
