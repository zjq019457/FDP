"""MPT-7B dense (uncompressed) five-task accuracy under the frozen MPT formal-accuracy manifest.
Same tasks, metric contract (ARC acc_norm; others acc), lm_eval settings, sample-identity/order checks and
correctness parity as mpt_formal_acc_runner.py; only the model is the dense base model (no cell/resource lock)."""
import os, sys, json, hashlib, datetime
from pathlib import Path
E = Path("/diskE/jiaqizhang/MPT7B_FDP_EB_CAMPAIGN_V1/formal_accuracy_20261007T202413Z")
OUT = Path("/diskF/jiaqizhang/DENSE_BASELINES_V1/mpt_dense_acc")
def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()
def canonical_sha(x): return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
mp = E / "FORMAL_ACCURACY_MANIFEST.json"; manifest = json.loads(mp.read_text()); mh = digest(mp)
assert mh == (E / "FORMAL_ACCURACY_MANIFEST.sha256").read_text().strip()
for path, sha in {**manifest["runtime_sources"], **manifest["dataset_files"]}.items():
    if digest(path) != sha: raise RuntimeError("scorer/data changed " + path)
OUT.mkdir(parents=True, exist_ok=True); (OUT / "cache").mkdir(exist_ok=True)
os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_CACHE=str(OUT / "cache"), TOKENIZERS_PARALLELISM="false")
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import evaluator
from lm_eval.tasks import TaskManager, get_task_dict
from lm_eval.models.huggingface import HFLM
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.manual_seed(2)
model = AutoModelForCausalLM.from_pretrained(manifest["model_path"], local_files_only=True, trust_remote_code=False,
                                             torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().to("cuda:0")
tok = AutoTokenizer.from_pretrained(manifest["model_path"], local_files_only=True, use_fast=True, trust_remote_code=False)
lm = HFLM(pretrained=model, tokenizer=tok, batch_size=1, max_length=2048, logits_cache=False, add_bos_token=False)
sys.path.insert(0, str(E / "tasks")); tm = TaskManager(include_path=str(E / "tasks")); task_dict = get_task_dict(list(manifest["tasks"]), tm)
scores = {}
for name, contract in manifest["tasks"].items():
    metric = contract["metric"]
    result = evaluator.simple_evaluate(model=lm, tasks=[name], task_manager=tm, num_fewshot=0, batch_size=1, device="cuda:0", log_samples=True,
                                       limit=None, cache_requests=False, bootstrap_iters=0, random_seed=2, numpy_random_seed=2, torch_random_seed=2, fewshot_random_seed=2)
    samples = result["samples"][name]; task = task_dict[name]; ordered = []
    for row in samples:
        choices = task.doc_to_choice(row["doc"]); ll = np.array([r[0] for r in row["filtered_resps"]], dtype=float)
        gold = task.doc_to_text(row["doc"]) if task.multiple_input else task.doc_to_target(row["doc"])
        if isinstance(gold, str): gold = choices.index(gold) if gold in choices else -100
        pred = int(np.argmax(ll)); lengths = np.array([len(x) for x in choices], dtype=float); pred_norm = int(np.argmax(ll / lengths))
        chosen = pred_norm if metric == "acc_norm" else pred
        if float(chosen in gold if isinstance(gold, list) else chosen == gold) != float(row[metric]): raise RuntimeError("correctness parity failed")
        ordered.append({"doc_id": row["doc_id"], "doc_sha256": canonical_sha(row["doc"])})
    if len(samples) != contract["n"] or canonical_sha(ordered) != contract["ordered_document_sha256"]: raise RuntimeError("sample identity/order mismatch")
    score = float(result["results"][name][metric + ",none"])
    rec = {"task": name, "metric": metric, "score": score, "n": len(samples), "samples": samples, "manifest_sha256": mh,
           "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (OUT / (name + ".json")).write_text(json.dumps(rec, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
    scores[name] = score; print("TASK_DONE", name, score, flush=True)
(OUT / "ACC_SUMMARY.json").write_text(json.dumps({"arm": "dense", "manifest_sha256": mh, "per_task": scores, "macro": sum(scores.values()) / len(scores)}, indent=2) + "\n")
print("DONE", scores, flush=True)
