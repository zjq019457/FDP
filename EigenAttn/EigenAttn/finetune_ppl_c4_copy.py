#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_ppl_c4.py
- 目标：以 C4 streaming 数据进行 PPL 为主的继续训练（PEFT/DoRA/LoRA adapter），并周期性评估下游多选题准确率（ARC / HellaSwag / WinoGrande）
- 特点：
  1) 完全离线（HF_DATASETS_OFFLINE=1）
  2) DDP 多卡（4×A100）用 torchrun 启动
  3) 支持断点续训（自动找最新 checkpoint）
  4) PPL 为主指标：按 PPL 保存 best_model
  5) ACC 辅助指标：使用 CE 累加方式算选项 log-likelihood，尽量省显存
  6) 规避 PyTorch allocator expandable_segments 内部断言：强制移除 expandable_segments

推荐启动方式：
  (finetune) CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 \
  torchrun --nproc_per_node=4 finetune_ppl_c4_copy.py
"""

import os, math, glob, json, shutil
from typing import List, Dict, Any, Optional, Tuple

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
from peft import PeftModel

# =========================================================
# 0) ENV：离线缓存 + 多卡
# =========================================================
# 你服务器的离线 cache 路径
os.environ.setdefault("HF_HOME", "/diskF/hf")
os.environ.setdefault("HF_DATASETS_CACHE", "/diskF/jiaqi_hf_cache/datasets")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_CACHE", "/diskF/hf/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/diskF/hf/hub")
os.environ.setdefault("HF_HUB_CACHE", "/diskF/hf/hub")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 多卡：建议命令行设 CUDA_VISIBLE_DEVICES；这里给默认值
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

# -----------------------------
# 强烈建议：不要使用 expandable_segments
# 你之前遇到的 INTERNAL ASSERT FAILED 大概率来自 expandable_segments
# 这里即使你 shell 里设了，也会强制覆盖掉
# -----------------------------
alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" in alloc_conf:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

# LOCAL_RANK 用于 DDP 每个进程绑定一张卡
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)
device = torch.device("cuda", LOCAL_RANK) if torch.cuda.is_available() else torch.device("cpu")

# =========================================================
# 1) CONFIG：路径与超参
# =========================================================
# C4 数据（json.gz）
DATA_DIR  = "/diskE/c4-en/en/"

# 你的离线多选题任务目录（ARC/HellaSwag/WinoGrande）
TASKS_DIR = "/diskF/a1804298693/offline_datasets"

# 训练输出目录（checkpoint、日志、best_model、final_model）
OUTPUT_DIR = "/home/bai-project/jiaqizhang/OptFinetune"

# base model（你 eigen 输出的 final_model）
MODEL_PATH     = "./eigen_outputs_1.0x/final_model"
# tokenizer
TOKENIZER_PATH = "/home/bai-project/jiaqizhang/opt-125m"
# PEFT adapter（DoRA/LoRA）
ADAPTER_PATH   = "./dora_extreme_ppl_model"

# 训练序列长度（对显存影响很大，OOM/不稳定可先降到 512）
BLOCK_SIZE = 512

# 评估频率（按 step）
PPL_EVAL_STEPS = 2000
ACC_EVAL_STEPS = 2000

# PPL：评估多少个 batch（越大越稳，越慢）
PPL_BATCHES = 80

# ACC：每个任务最多评估多少条 dev（先用 500~2000 比较实用）
MAX_ACC_EXAMPLES = 1000

# C4 eval 用多少个文件（1~4 通常够）
C4_EVAL_FILES = 1

# 继续训练的步数（在 checkpoint 基础上 + EXTRA_STEPS）
EXTRA_STEPS = 50000

# -----------------------------
# 训练 batch 配置
# 说明：总 batch = PER_DEVICE_BS * world_size * GRAD_ACCUM
# PER_DEVICE_BS 对显存最敏感，先保守；稳定后再上调
# -----------------------------
PER_DEVICE_BS = 1
GRAD_ACCUM = 32
LR = 5e-5

# ACC scoring 时的 option 小 batch（越大越快但越吃显存）
OPTION_BATCH_SIZE = 2

# =========================================================
# 2) DISK CHECK：避免写爆盘
# =========================================================
def check_disk(path="/diskF", min_gb=50):
    free = shutil.disk_usage(path).free / (1024**3)
    if LOCAL_RANK == 0:
        print(f"[Disk] Free at {path}: {free:.1f} GB")
    if free < min_gb:
        raise RuntimeError(f"Disk almost full at {path}: {free:.1f} GB < {min_gb} GB")

check_disk("/diskF", 50)

# =========================================================
# 3) TOKENIZER / MODEL：加载 base + adapter，设置训练相关开关
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 读取 config：判断是否 OPT（OPT 不支持 sdpa，需要 attn_implementation="eager"）
cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
is_opt = (getattr(cfg, "model_type", "") == "opt")

base_kwargs = dict(
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
if is_opt:
    base_kwargs["attn_implementation"] = "eager"

base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **base_kwargs)

# 加载 PEFT adapter，并设置为可训练
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=True)
model.to(device)

# 训练必须关闭 cache（否则会额外占用显存且与 checkpointing 冲突）
model.config.use_cache = False

# 关键：让输入可求导，否则 gradient checkpointing 会出现 “None of the inputs require grad”
# 这个在 PEFT/冻结 base 权重时尤其重要
model.enable_input_require_grads()

# 显式设置 gradient checkpointing 的 reentrant 行为（更稳）
try:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
except TypeError:
    model.gradient_checkpointing_enable()

if LOCAL_RANK == 0:
    model.print_trainable_parameters()
    print(f"[Model] model_type={getattr(cfg,'model_type',None)} attn_impl={base_kwargs.get('attn_implementation','(default)')}")

# =========================================================
# 4) C4 STREAMING LOADER：按文件切 train/eval
# =========================================================
all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json.gz")))
if len(all_files) == 0:
    raise RuntimeError(f"No C4 json.gz found in: {DATA_DIR}")

eval_files  = all_files[:C4_EVAL_FILES]
train_files = all_files[C4_EVAL_FILES:] if len(all_files) > C4_EVAL_FILES else all_files

if LOCAL_RANK == 0:
    print(f"[C4] train_files={len(train_files)}, eval_files={len(eval_files)}")

# streaming=True：不会把全量数据加载进内存
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
    # 不返回 attention_mask 以减少数据体积；训练时用 causal LM 自回归
    return tokenizer(batch["text"], return_attention_mask=False)

train_stream = train_stream.map(tokenize_c4, batched=True)
eval_stream  = eval_stream.map(tokenize_c4,  batched=True)

class ConstantLengthDataset(IterableDataset):
    """
    将 streaming 的 token 序列拼接成固定长度 block，作为 language modeling 训练样本
    注意：IterableDataset 不支持 __len__，恢复 checkpoint 时要用 ignore_data_skip=True
    """
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
            # 每凑够 block_size 就 yield 一条训练样本
            while len(buf) >= self.block_size:
                chunk = buf[:self.block_size]
                buf = buf[self.block_size:]
                yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}

train_dataset    = ConstantLengthDataset(train_stream, BLOCK_SIZE)
ppl_eval_dataset = ConstantLengthDataset(eval_stream,  BLOCK_SIZE)

def collate_fn(batch):
    """
    causal LM：labels 与 input_ids 相同（内部 shift）
    """
    ids = torch.stack([x["input_ids"] for x in batch])
    return {"input_ids": ids, "labels": ids.clone()}

# =========================================================
# 5) PPL 计算（主指标）
# =========================================================
@torch.no_grad()
def compute_ppl(model, dataset, max_batches=80, batch_size=1):
    """
    在 eval_stream 上取 max_batches 个 batch，计算 token-level 平均 loss，再 exp 得到 PPL
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        # bf16 autocast：省显存
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

# =========================================================
# 6) ACC（辅助指标）：多选题 log-likelihood（CE 累加版，省显存）
# =========================================================
def _safe_read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def _read_labels_lst(path: str) -> List[str]:
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            labels.append(line.strip())
    return labels

def truncate_prompt(prompt_ids: List[int], option_ids: List[int], max_len: int) -> List[int]:
    """
    为 prompt 预留 option 的长度，超长就截断 prompt（保留尾部更合理）
    """
    keep = max_len - len(option_ids)
    if keep <= 0:
        return prompt_ids[-1:]
    return prompt_ids[-keep:] if len(prompt_ids) > keep else prompt_ids

@torch.no_grad()
def score_options_batch_ce(
    model,
    prompt: str,
    options: List[str],
    max_seq_len: int = 2048,
    opt_batch_size: int = 2,
) -> List[float]:
    """
    对每个 option 计算 log-likelihood 分数（越大越好）
    做法：
      - 输入序列 = prompt + option
      - labels 只在 option 部分有效，prompt 部分为 -100（ignore）
      - 计算 CE loss 的逐 token 和（NLL），分数= -NLL
    为了省显存：将 options 分成小 batch 计算
    """
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
            logits = out.logits  # [B, T, V]

        # shift：预测下一个 token
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = label_ids[:, 1:].contiguous()

        B, Tm1, V = shift_logits.shape

        # 逐 token 的 cross entropy（ignore -100）
        loss_flat = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100
        ).view(B, Tm1)

        valid = (shift_labels != -100)
        nll = (loss_flat * valid).sum(dim=1)  # [B]
        scores = (-nll).detach().to("cpu").tolist()
        scores_all.extend(scores)

        # 清理显存碎片
        del input_ids, label_ids, logits, shift_logits, shift_labels, loss_flat, valid, nll
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.train()
    return scores_all

def eval_arc(dev_path_jsonl: str, max_examples: int) -> Tuple[int, int]:
    """
    ARC jsonl 常见结构：
      - question: {stem: str, choices: [{label, text}, ...]}
      - answerKey: "A"/"B"/"C"/"D"
    """
    rows = _safe_read_jsonl(dev_path_jsonl)
    if max_examples > 0:
        rows = rows[:max_examples]

    correct, total = 0, 0
    for ex in rows:
        q = ex.get("question", {})
        stem = q.get("stem", ex.get("question", ex.get("query", "")))
        choices = q.get("choices", ex.get("choices", []))

        opts, labels = [], []
        if isinstance(choices, dict):
            lbls = choices.get("label", [])
            txts = choices.get("text", [])
            for l, t in zip(lbls, txts):
                labels.append(str(l))
                opts.append(str(t))
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

        # completion 用 " A" / " B" ... 形式让模型选择
        option_completions = [f" {l}" for l in labels]

        scores = score_options_batch_ce(model, prompt, option_completions, 2048, OPTION_BATCH_SIZE)
        pred_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        pred = labels[pred_idx]

        correct += int(pred == answer)
        total += 1

    return correct, total

def eval_hellaswag(dev_path_jsonl: str, max_examples: int) -> Tuple[int, int]:
    """
    HellaSwag jsonl 常见字段：
      - ctx 或 ctx_a/ctx_b
      - endings: list[str]
      - label: int (0..3)
    """
    rows = _safe_read_jsonl(dev_path_jsonl)
    if max_examples > 0:
        rows = rows[:max_examples]

    correct, total = 0, 0
    for ex in rows:
        ctx = ex.get("ctx", None)
        if ctx is None:
            ctx_a = ex.get("ctx_a", "")
            ctx_b = ex.get("ctx_b", "")
            ctx = (ctx_a + " " + ctx_b).strip()

        endings = ex.get("endings", None)
        if endings is None:
            endings = ex.get("choices", ex.get("options", []))

        label = ex.get("label", ex.get("answer", None))
        if label is None:
            label = ex.get("gold", 0)
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
    """
    WinoGrande 常见字段：
      - sentence: 包含 "_" 空位
      - option1, option2
      - answer: "1"/"2"（dev 通常有；也可能用 dev-labels.lst）
    """
    rows = _safe_read_jsonl(dev_jsonl)
    labels_from_lst = None
    if dev_labels_lst and os.path.exists(dev_labels_lst):
        labels_from_lst = _read_labels_lst(dev_labels_lst)

    if max_examples > 0:
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

        # 将 "_" 替换成选项
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
def compute_accuracy_suite(max_examples: int = 1000) -> Dict[str, float]:
    """
    只评估：
      - ARC-Easy
      - ARC-Challenge
      - HellaSwag
      - WinoGrande
    不包含 PIQA（你明确说 PIQA 有问题先不用）
    """
    results = {}

    arc_easy_dev = os.path.join(TASKS_DIR, "ai2_arc", "ARC-Easy", "ARC-Easy-Dev.jsonl")
    arc_chal_dev = os.path.join(TASKS_DIR, "ai2_arc", "ARC-Challenge", "ARC-Challenge-Dev.jsonl")
    if os.path.exists(arc_easy_dev):
        c, t = eval_arc(arc_easy_dev, max_examples)
        results["arc_easy_acc"] = c / max(t, 1)
    if os.path.exists(arc_chal_dev):
        c, t = eval_arc(arc_chal_dev, max_examples)
        results["arc_challenge_acc"] = c / max(t, 1)

    hs_dev = os.path.join(TASKS_DIR, "hellaswag-train-dev", "valid.jsonl")
    if os.path.exists(hs_dev):
        c, t = eval_hellaswag(hs_dev, max_examples)
        results["hellaswag_acc"] = c / max(t, 1)

    wg_dev = os.path.join(TASKS_DIR, "winogrande", "dev.jsonl")
    wg_dev_labels = os.path.join(TASKS_DIR, "winogrande", "dev-labels.lst")
    if os.path.exists(wg_dev):
        c, t = eval_winogrande(wg_dev, wg_dev_labels if os.path.exists(wg_dev_labels) else None, max_examples)
        results["winogrande_acc"] = c / max(t, 1)

    return results

# =========================================================
# 7) CALLBACK：PPL 主 + ACC 辅，best_model 按 PPL 保存
# =========================================================
class PPLAndAccCallback(TrainerCallback):
    def __init__(self):
        self.best_ppl = 1e18

    def on_step_end(self, args, state, control, **kwargs):
        # 只在 rank0 做评估/写日志/保存 best，避免多进程重复
        if not state.is_world_process_zero:
            return
        if state.global_step == 0:
            return

        m = kwargs["model"]

        # -------- PPL（主指标）--------
        if state.global_step % PPL_EVAL_STEPS == 0:
            print(f"\n[PPL] Evaluating at step {state.global_step} ...")
            ppl = compute_ppl(m, ppl_eval_dataset, max_batches=PPL_BATCHES, batch_size=1)
            print(f"[PPL] step={state.global_step} ppl={ppl:.4f}")

            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "ppl_log.txt"), "a") as f:
                f.write(f"{state.global_step},{ppl}\n")

            # 按 PPL 保存 best_model
            if ppl < self.best_ppl:
                self.best_ppl = ppl
                best_dir = os.path.join(args.output_dir, "best_model")
                m.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                print(f"[PPL] New best saved to {best_dir}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # -------- ACC（辅助指标）--------
        if state.global_step % ACC_EVAL_STEPS == 0:
            print(f"\n[ACC] Evaluating (ARC/HellaSwag/WinoGrande) at step {state.global_step} ...")
            accs = compute_accuracy_suite(MAX_ACC_EXAMPLES)

            for k, v in accs.items():
                print(f"[ACC] {k}={v:.4f}")

            with open(os.path.join(args.output_dir, "acc_log.txt"), "a") as f:
                row = {"step": state.global_step, **accs}
                f.write(json.dumps(row) + "\n")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# =========================================================
# 8) 自动恢复 checkpoint + 自动设置 max_steps
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

# 总训练步数：在已有 checkpoint 基础上，再训练 EXTRA_STEPS
MAX_STEPS = max(resume_step + EXTRA_STEPS, EXTRA_STEPS)

if LOCAL_RANK == 0:
    print(f"[Resume] last_ckpt = {ckpt}")
    print(f"[Resume] resume_step = {resume_step}")
    print(f"[Train]  EXTRA_STEPS = {EXTRA_STEPS}")
    print(f"[Train]    MAX_STEPS = {MAX_STEPS}")

# =========================================================
# 9) TRAIN ARGS：关键点在 ignore_data_skip + checkpointing
# =========================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,

    # --- 断点保存 ---
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=5,

    # --- 日志 ---
    logging_steps=100,
    report_to="none",

    # --- 精度 ---
    bf16=True,

    # --- batch（显存优先）---
    per_device_train_batch_size=PER_DEVICE_BS,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,

    # --- DDP ---
    ddp_find_unused_parameters=False,

    # 关键：Iterable/streaming 恢复 checkpoint 时避免卡住（否则会尝试跳过数据导致超慢/卡 0%）
    ignore_data_skip=True,

    # 关键：显存优化（Trainer 层面也打开）
    gradient_checkpointing=True,

    # 关键：训练期间不做 Trainer 自带 eval（减少峰值）
    evaluation_strategy="no",

    # 关键：A100 上更稳/更省的优化器（需要 torch>=2 且支持 fused）
    optim="adamw_torch_fused",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    callbacks=[PPLAndAccCallback()],
)

# =========================================================
# 10) 开始训练（支持自动 resume）
# =========================================================
trainer.train(resume_from_checkpoint=ckpt)

# =========================================================
# 11) 保存最终模型（final_model）
# =========================================================
final_dir = os.path.join(OUTPUT_DIR, "final_model")
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)
if LOCAL_RANK == 0:
    print(f"[Done] final_model saved to: {final_dir}")
