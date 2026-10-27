#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
finetune_ppl_c4_peft_dora.py

- OPT-30B 用 C4 streaming 做继续训练
- PEFT: DoRA (LoRA with use_dora=True)
- 训练前先评估一次 PPL
- 训练中每 20 step 评估一次 PPL（写入 ppl_log.csv）
- ✅ 支持断点续训（自动找最新 checkpoint-xxxxx）
- ✅ 解决 DDP + streaming IterableDataset 下 checkpoint 不更新的问题：
    1) 对 streaming dataset 做 rank shard（每个 rank 读不同数据流）
    2) 加 on_save callback 打印，确认 Trainer 的 save 是否触发
- 完全离线 + DDP torchrun
"""

import os, math, glob, json, shutil, time
from typing import List, Dict, Any, Optional, Tuple
from transformers.trainer_callback import DefaultFlowCallback
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
)

# =========================================================
# 0) DDP rank utils (更稳：优先 torch.distributed，其次环境变量)
# =========================================================
def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, None)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default

def dist_is_init() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def get_rank() -> int:
    if dist_is_init():
        return torch.distributed.get_rank()
    # torchrun 常见环境变量
    return _env_int("RANK", 0)

def get_world_size() -> int:
    if dist_is_init():
        return torch.distributed.get_world_size()
    return _env_int("WORLD_SIZE", 1)

def get_local_rank() -> int:
    # torchrun 常用
    return _env_int("LOCAL_RANK", 0)

def is_rank0() -> bool:
    return get_rank() == 0

def log0(msg: str):
    if is_rank0():
        print(msg, flush=True)

# =========================================================
# 1) ENV：离线缓存 + TMPDIR 兜底 + allocator 配置
# =========================================================
os.environ.setdefault("HF_HOME", "/diskF/hf")
os.environ.setdefault("HF_DATASETS_CACHE", "/diskF/jiaqi_hf_cache/datasets")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/diskF/hf/hub")
os.environ.setdefault("HF_HUB_CACHE", "/diskF/hf/hub")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# TMPDIR 兜底：避免 No usable temporary directory
os.environ.setdefault("TMPDIR", "/diskF/tmp")
os.environ.setdefault("TEMP", "/diskF/tmp")
os.environ.setdefault("TMP", "/diskF/tmp")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

# 避免 expandable_segments 内部断言（强制覆盖）
alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" in alloc_conf:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

# 绑定 GPU
LOCAL_RANK = get_local_rank()
if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)
device = torch.device("cuda", LOCAL_RANK) if torch.cuda.is_available() else torch.device("cpu")

# =========================================================
# 2) CONFIG：路径与超参（按你的环境写死）
# =========================================================
DATA_DIR  = "/diskE/jiaqizhang/c4-en/en/"
TASKS_DIR = "/diskF/a1804298693/offline_datasets"

# ✅ 输出全部放 F 盘
OUTPUT_DIR = "/diskF/jiaqizhang/OptFinetune_opt30b_peft_dora"

MODEL_PATH     = "/diskD/jiaqizhang/eigen_outputs_opt30b_r448/final_model"
# MODEL_PATH  = "/diskF/jiaqizhang/eigen_outputs_opt30b_r7168_gpu3_2026-02-09_140915/final_model"
TOKENIZER_PATH = "/diskE/opt-30b"
#ADAPTER_PATH   = "/diskD/peft_opt30b_dora_r8"
ADAPTER_PATH = "/diskF/jiaqizhang/OptFinetune_opt30b_peft_dora/best_model"

BLOCK_SIZE = 1024

# ✅ 每 20 step 评估一次 PPL
PPL_EVAL_STEPS = 100
PPL_BATCHES    = 8

# ACC 低频
ACC_EVAL_STEPS    = 2000
MAX_ACC_EXAMPLES  = 200
OPTION_BATCH_SIZE = 2

C4_EVAL_FILES = 1
EXTRA_STEPS   = 50000

PER_DEVICE_BS = 2
GRAD_ACCUM    = 16
LR            = 5e-5

# DoRA 超参
PEFT_R       = 8
PEFT_ALPHA   = 16
PEFT_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]

# =========================================================
# 3) DISK CHECK
# =========================================================
def check_disk(path="/diskF", min_gb=50):
    free = shutil.disk_usage(path).free / (1024**3)
    if is_rank0():
        print(f"[Disk] Free at {path}: {free:.1f} GB", flush=True)
    if free < min_gb:
        raise RuntimeError(f"Disk almost full at {path}: {free:.1f} GB < {min_gb} GB")

check_disk("/diskF", 20)

if is_rank0():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

log0(f"[DDP] rank={get_rank()} local_rank={LOCAL_RANK} world_size={get_world_size()}")
log0(f"[Paths] MODEL_PATH={MODEL_PATH}")
log0(f"[Paths] TOKENIZER_PATH={TOKENIZER_PATH}")
log0(f"[Paths] ADAPTER_PATH={ADAPTER_PATH}")
log0(f"[Paths] OUTPUT_DIR={OUTPUT_DIR}")
log0(f"[PEFT]  MODE=DORA r={PEFT_R} alpha={PEFT_ALPHA} dropout={PEFT_DROPOUT}")
log0(f"[PPL]   pre-eval once + every {PPL_EVAL_STEPS} steps (batches={PPL_BATCHES})")

LOG_FILE = os.path.join(OUTPUT_DIR, "train_log_rank0.txt")
def log0_file(msg: str):
    if not is_rank0():
        return
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")
    print(msg, flush=True)

PPL_LOG = os.path.join(OUTPUT_DIR, "ppl_log.csv")
ACC_LOG = os.path.join(OUTPUT_DIR, "acc_log.jsonl")

# =========================================================
# 4) TOKENIZER / MODEL：base + DoRA adapter
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
model_type = getattr(cfg, "model_type", "")
hidden_size = getattr(cfg, "hidden_size", None)
log0_file(f"[Config] model_type={model_type} hidden_size={hidden_size}")

base_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
if model_type == "opt":
    base_kwargs["attn_implementation"] = "eager" \
    ""

base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **base_kwargs)

def has_adapter(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_config.json"))

if has_adapter(ADAPTER_PATH):
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=True)
    log0_file(f"[PEFT] Loaded existing adapter from: {ADAPTER_PATH}")
else:
    os.makedirs(ADAPTER_PATH, exist_ok=True)
    dora_cfg = LoraConfig(
        r=PEFT_R,
        lora_alpha=PEFT_ALPHA,
        lora_dropout=PEFT_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
        use_dora=True,
        init_lora_weights=True,
    )
    model = get_peft_model(base_model, dora_cfg)
    if is_rank0():
        model.save_pretrained(ADAPTER_PATH)
        tokenizer.save_pretrained(ADAPTER_PATH)
    log0_file(f"[PEFT] DoRA adapter initialized and saved to {ADAPTER_PATH}")

model.config.use_cache = False
model.enable_input_require_grads()

# ✅ 显式 use_reentrant=False（你要的 warning 也会消失/减少）
try:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    log0_file("[GC] gradient_checkpointing_enable(use_reentrant=False)")
except TypeError:
    model.gradient_checkpointing_enable()
    log0_file("[GC] gradient_checkpointing_enable() fallback (no kwargs supported)")

model.to(device)

if is_rank0():
    model.print_trainable_parameters()
    log0_file(f"[Model] attn_impl={base_kwargs.get('attn_implementation','(default)')}")

# =========================================================
# 5) C4 STREAMING：train/eval 切分 + 固定长度样本
# =========================================================
all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json.gz")))
if len(all_files) == 0:
    raise RuntimeError(f"No C4 json.gz found in: {DATA_DIR}")

eval_files  = all_files[:C4_EVAL_FILES]
train_files = all_files[C4_EVAL_FILES:] if len(all_files) > C4_EVAL_FILES else all_files
log0_file(f"[C4] train_files={len(train_files)}, eval_files={len(eval_files)}")

train_stream = load_dataset(
    "json",
    data_files=train_files,
    split="train",
    streaming=True,
    cache_dir=os.environ["HF_DATASETS_CACHE"],
)
eval_stream = load_dataset(
    "json",
    data_files=eval_files,
    split="train",
    streaming=True,
    cache_dir=os.environ["HF_DATASETS_CACHE"],
)
def tokenize_c4(batch):
    return tokenizer(batch["text"], return_attention_mask=False)

# 先做 tokenize（保持你原来的逻辑）
train_stream = train_stream.map(tokenize_c4, batched=True)
eval_stream  = eval_stream.map(tokenize_c4,  batched=True)

# ✅ 关键修复：DDP + streaming 下做“手动 sharding”（兼容没有 .shard() 的 iterable）
WS = get_world_size()
RK = get_rank()

def shard_iterable(ds, num_shards: int, index: int):
    """
    对任何 iterable 都生效的分片方式：
    rank=index 只取第 index, index+num_shards, index+2*num_shards, ... 个样本。
    返回一个 torch IterableDataset，供后续 ConstantLengthDataset 包装使用。
    """
    if num_shards <= 1:
        return ds

    class _Wrapped(torch.utils.data.IterableDataset):
        def __iter__(self):
            for i, x in enumerate(ds):
                if i % num_shards == index:
                    yield x

    return _Wrapped()

if WS > 1:
    train_stream = shard_iterable(train_stream, WS, RK)
    eval_stream  = shard_iterable(eval_stream,  WS, RK)
    log0_file(f"[C4] streaming sharded (manual): num_shards={WS}, index={RK}")

class ConstantLengthDataset(IterableDataset):
    def __init__(self, dataset, block_size=1024):
        self.dataset = dataset
        self.block_size = block_size

    def __iter__(self):
        buf = []
        for sample in self.dataset:
            ids = sample.get("input_ids", None)
            if ids is None:
                continue
            buf.extend(ids)
            while len(buf) >= self.block_size:
                chunk = buf[:self.block_size]
                buf = buf[self.block_size:]
                yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}

train_dataset    = ConstantLengthDataset(train_stream, BLOCK_SIZE)
ppl_eval_dataset = ConstantLengthDataset(eval_stream,  BLOCK_SIZE)

def collate_fn(batch):
    ids = torch.stack([x["input_ids"] for x in batch])
    return {"input_ids": ids, "labels": ids.clone()}

# =========================================================
# 6) PPL（主指标）
# =========================================================
@torch.no_grad()
def compute_ppl(model, dataset, max_batches=8, batch_size=1):
    model.eval()
    total_loss, total_tokens = 0.0, 0

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(**batch)
            loss = out.loss
        if loss is None:
            raise RuntimeError("compute_ppl: model did not return loss. Check labels.")
        n = batch["input_ids"].numel()
        total_loss += loss.item() * n
        total_tokens += n

    ppl = math.exp(total_loss / max(total_tokens, 1))
    model.train()
    return ppl

def write_ppl(step: int, ppl: float):
    if not is_rank0():
        return
    header_needed = not os.path.exists(PPL_LOG)
    with open(PPL_LOG, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("step,ppl\n")
        f.write(f"{step},{ppl}\n")

# =========================================================
# 7) ACC（可选，低频）
# =========================================================
def _safe_read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _read_labels_lst(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f]

def truncate_prompt(prompt_ids: List[int], option_ids: List[int], max_len: int) -> List[int]:
    keep = max_len - len(option_ids)
    if keep <= 0:
        return prompt_ids[-1:]
    return prompt_ids[-keep:] if len(prompt_ids) > keep else prompt_ids

@torch.no_grad()
def score_options_batch_ce(model, prompt: str, options: List[str], max_seq_len: int = 2048, opt_batch_size: int = 2) -> List[float]:
    model.eval()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    option_ids_list = [tokenizer.encode(opt, add_special_tokens=False) for opt in options]
    scores_all = []

    for start in range(0, len(option_ids_list), opt_batch_size):
        chunk_opts = option_ids_list[start:start + opt_batch_size]
        seqs, labs = [], []
        for opt_ids in chunk_opts:
            p_ids = truncate_prompt(prompt_ids, opt_ids, max_seq_len)
            ids = p_ids + opt_ids
            lab = [-100] * len(p_ids) + opt_ids
            seqs.append(ids)
            labs.append(lab)

        max_len = max(len(s) for s in seqs)
        pad_id = tokenizer.pad_token_id

        input_ids, label_ids = [], []
        for ids, lab in zip(seqs, labs):
            pad_n = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_n)
            label_ids.append(lab + [-100] * pad_n)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        label_ids = torch.tensor(label_ids, dtype=torch.long, device=device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(input_ids=input_ids)
            logits = out.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = label_ids[:, 1:].contiguous()
        B, Tm1, V = shift_logits.shape

        loss_flat = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100
        ).view(B, Tm1)

        valid = (shift_labels != -100)
        nll = (loss_flat * valid).sum(dim=1)
        scores = (-nll).detach().to("cpu").tolist()
        scores_all.extend(scores)

        del input_ids, label_ids, logits, shift_logits, shift_labels, loss_flat, valid, nll
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.train()
    return scores_all

def eval_arc(dev_path_jsonl: str, max_examples: int) -> Tuple[int, int]:
    rows = _safe_read_jsonl(dev_path_jsonl)[:max_examples]
    correct, total = 0, 0
    for ex in rows:
        q = ex.get("question", {})
        if isinstance(q, str):
            stem = q
            choices = ex.get("choices", [])
        else:
            stem = q.get("stem", ex.get("query", ""))
            choices = q.get("choices", ex.get("choices", []))

        opts, labels = [], []
        if isinstance(choices, dict):
            lbls = choices.get("label", [])
            txts = choices.get("text", [])
            for l, t in zip(lbls, txts):
                labels.append(str(l)); opts.append(str(t))
        else:
            for c in choices:
                labels.append(str(c.get("label", "")))
                opts.append(str(c.get("text", c.get("choice", ""))))

        answer = str(ex.get("answerKey", ex.get("answer", ""))).strip()

        prompt_lines = [f"Question: {stem}", ""]
        for l, o in zip(labels, opts):
            prompt_lines.append(f"{l}) {o}")
        prompt_lines.append("Answer:")
        prompt = "\n".join(prompt_lines)

        option_completions = [f" {l}" for l in labels]
        scores = score_options_batch_ce(model, prompt, option_completions, 2048, OPTION_BATCH_SIZE)
        pred_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        pred = labels[pred_idx]
        correct += int(pred == answer)
        total += 1
    return correct, total

def eval_hellaswag(dev_path_jsonl: str, max_examples: int) -> Tuple[int, int]:
    rows = _safe_read_jsonl(dev_path_jsonl)[:max_examples]
    correct, total = 0, 0
    for ex in rows:
        ctx = ex.get("ctx", None)
        if ctx is None:
            ctx = (ex.get("ctx_a", "") + " " + ex.get("ctx_b", "")).strip()
        endings = ex.get("endings", ex.get("choices", ex.get("options", [])))

        label = ex.get("label", ex.get("answer", ex.get("gold", 0)))
        try:
            gold = int(label)
        except:
            mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
            gold = mapping.get(str(label).strip(), 0)

        prompt = f"Context: {ctx}\nAnswer:"
        option_completions = [f" {e}" for e in endings]
        scores = score_options_batch_ce(model, prompt, option_completions, 2048, OPTION_BATCH_SIZE)
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        correct += int(pred == gold)
        total += 1
    return correct, total

def eval_winogrande(dev_jsonl: str, dev_labels_lst: Optional[str], max_examples: int) -> Tuple[int, int]:
    rows = _safe_read_jsonl(dev_jsonl)
    labels_from_lst = None
    if dev_labels_lst and os.path.exists(dev_labels_lst):
        labels_from_lst = _read_labels_lst(dev_labels_lst)

    rows = rows[:max_examples]
    if labels_from_lst is not None:
        labels_from_lst = labels_from_lst[:max_examples]

    correct, total = 0, 0
    for i, ex in enumerate(rows):
        sent = ex.get("sentence", "")
        opt1 = ex.get("option1", ex.get("choice1", ""))
        opt2 = ex.get("option2", ex.get("choice2", ""))

        ans = ex.get("answer", None)
        if ans is None and labels_from_lst is not None and i < len(labels_from_lst):
            ans = labels_from_lst[i]
        ans = str(ans).strip() if ans is not None else "1"

        if "_" in sent:
            s1 = sent.replace("_", opt1)
            s2 = sent.replace("_", opt2)
        else:
            s1 = (sent + " " + opt1).strip()
            s2 = (sent + " " + opt2).strip()

        prompt = "Choose the more plausible sentence.\nAnswer:"
        option_completions = [f" {s1}", f" {s2}"]
        scores = score_options_batch_ce(model, prompt, option_completions, 2048, OPTION_BATCH_SIZE)
        pred = 1 if scores[0] >= scores[1] else 2
        gold = int(ans) if ans.isdigit() else (1 if ans.upper() == "A" else 2)
        correct += int(pred == gold)
        total += 1
    return correct, total

@torch.no_grad()
def compute_accuracy_suite(max_examples: int = 200) -> Dict[str, float]:
    results = {}
    arc_easy_dev = os.path.join(TASKS_DIR, "ai2_arc", "ARC-Easy", "ARC-Easy-Dev.jsonl")
    arc_chal_dev = os.path.join(TASKS_DIR, "ai2_arc", "ARC-Challenge", "ARC-Challenge-Dev.jsonl")
    if os.path.exists(arc_easy_dev):
        c, t = eval_arc(arc_easy_dev, max_examples); results["arc_easy_acc"] = c / max(t, 1)
    if os.path.exists(arc_chal_dev):
        c, t = eval_arc(arc_chal_dev, max_examples); results["arc_challenge_acc"] = c / max(t, 1)

    hs_dev = os.path.join(TASKS_DIR, "hellaswag-train-dev", "valid.jsonl")
    if os.path.exists(hs_dev):
        c, t = eval_hellaswag(hs_dev, max_examples); results["hellaswag_acc"] = c / max(t, 1)

    wg_dev = os.path.join(TASKS_DIR, "winogrande", "dev.jsonl")
    wg_dev_labels = os.path.join(TASKS_DIR, "winogrande", "dev-labels.lst")
    if os.path.exists(wg_dev):
        c, t = eval_winogrande(wg_dev, wg_dev_labels if os.path.exists(wg_dev_labels) else None, max_examples)
        results["winogrande_acc"] = c / max(t, 1)
    return results

# =========================================================
# 8) CALLBACK：PPL/ACC + 保存调试（关键）
# =========================================================
class PPLAndAccCallback(TrainerCallback):
    def __init__(self):
        self.best_ppl = 1e18

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if state.global_step == 0:
            return

        m = kwargs["model"]

        if state.global_step % PPL_EVAL_STEPS == 0:
            t0 = time.time()
            ppl = compute_ppl(m, ppl_eval_dataset, max_batches=PPL_BATCHES, batch_size=1)
            dt = time.time() - t0
            log0_file(f"[PPL] step={state.global_step} ppl={ppl:.4f} (batches={PPL_BATCHES}, {dt:.1f}s)")
            write_ppl(state.global_step, ppl)

            if ppl < self.best_ppl:
                self.best_ppl = ppl
                best_dir = os.path.join(args.output_dir, "best_model")
                m.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                log0_file(f"[PPL] New best saved to {best_dir}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if state.global_step % ACC_EVAL_STEPS == 0:
            log0_file(f"[ACC] Evaluating suite at step={state.global_step} (max_examples={MAX_ACC_EXAMPLES}) ...")
            try:
                accs = compute_accuracy_suite(MAX_ACC_EXAMPLES)
                row = {"step": state.global_step, **accs}
                if accs:
                    log0_file("[ACC] " + " ".join([f"{k}={v:.4f}" for k, v in accs.items()]))
                else:
                    log0_file("[ACC] (no tasks found / empty results)")
                with open(ACC_LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            except Exception as e:
                log0_file(f"[ACC] skipped due to error: {repr(e)}")
                with open(ACC_LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": state.global_step, "error": repr(e)}) + "\n")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class SaveDebugCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            log0_file(f"[SAVE] Trainer saved checkpoint at step={state.global_step} -> {args.output_dir}")
        return control

# =========================================================
# 9) 自动恢复 checkpoint + max_steps
# =========================================================
def get_last_ckpt(d: str) -> Optional[str]:
    if not os.path.isdir(d):
        return None
    cks = [x for x in os.listdir(d) if x.startswith("checkpoint-")]
    if not cks:
        return None
    cks = sorted(cks, key=lambda x: int(x.split("-")[-1]))
    return os.path.join(d, cks[-1])

def parse_ckpt_step(ckpt_path: Optional[str]) -> int:
    if not ckpt_path:
        return 0
    name = os.path.basename(ckpt_path)
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-")[-1])
        except:
            return 0
    return 0

ckpt = get_last_ckpt(OUTPUT_DIR)
resume_step = parse_ckpt_step(ckpt)
MAX_STEPS = max(resume_step + EXTRA_STEPS, EXTRA_STEPS)

log0_file(f"[Resume] last_ckpt = {ckpt}")
log0_file(f"[Resume] resume_step = {resume_step}")
log0_file(f"[Train]  EXTRA_STEPS = {EXTRA_STEPS}")
log0_file(f"[Train]    MAX_STEPS = {MAX_STEPS}")

# =========================================================
# 10) TRAIN ARGS
# =========================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,

    save_strategy="steps",
    save_steps=1,
    # ✅ 建议先用 3，确认 checkpoint 在滚动；确认后你再改回 1
    save_total_limit=1,
    save_safetensors=True,
    save_on_each_node=False,

    logging_steps=10,
    report_to="none",
    log_level="info",

    bf16=True,

    per_device_train_batch_size=PER_DEVICE_BS,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,

    ddp_find_unused_parameters=False,

    ignore_data_skip=True,
    gradient_checkpointing=True,

    evaluation_strategy="no",
    optim="adamw_torch_fused",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    callbacks=[DefaultFlowCallback(), PPLAndAccCallback()],
)


# =========================================================
# 11) 训练前先评估一次 PPL（step=resume_step）
# =========================================================
if is_rank0():
    log0_file("[PPL] Pre-evaluating before training ...")
    t0 = time.time()
    ppl0 = compute_ppl(model, ppl_eval_dataset, max_batches=PPL_BATCHES, batch_size=1)
    dt = time.time() - t0
    log0_file(f"[PPL] step={resume_step} ppl={ppl0:.4f} (batches={PPL_BATCHES}, {dt:.1f}s)")
    write_ppl(resume_step, ppl0)

    best_dir0 = os.path.join(OUTPUT_DIR, "best_model")
    model.save_pretrained(best_dir0)
    tokenizer.save_pretrained(best_dir0)
    log0_file(f"[PPL] Saved initial best_model to {best_dir0}")

# =========================================================
# 12) TRAIN（支持断点续训）
# =========================================================
#trainer.train(resume_from_checkpoint=ckpt)
trainer.train()

# =========================================================
# 13) SAVE final
# =========================================================
final_dir = os.path.join(OUTPUT_DIR, "final_model")
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)
log0_file(f"[Done] final_model saved to: {final_dir}")
