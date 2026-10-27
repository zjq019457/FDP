import os, json, glob, gzip, math, random, argparse, time, shutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn
from models.low_rank_balance import apply_low_rank_balance_to_opt_model, collect_orth_balance_regularizers

try:
    from tqdm import tqdm
except Exception:
    # Fallback if tqdm is not installed.
    def tqdm(x, **kwargs):
        return x

# ---- make attention path stable (match sanity_AB_opt30b.py) ----
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# -------------------------
# dataset reader (txt/jsonl/json/gz) -> token stream (CPU)
# -------------------------
def iter_texts_from_dir(data_dir, max_files=80, max_lines=300000):
    exts = ["*.txt", "*.jsonl", "*.json", "*.gz"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(data_dir, "**", e), recursive=True))
    files = sorted(files)[:max_files]
    if not files:
        return
    n = 0
    for fp in files:
        opener = gzip.open if fp.endswith(".gz") else open
        try:
            with opener(fp, "rt", encoding="utf-8", errors="ignore") as f:
                if fp.endswith(".json") and (not fp.endswith(".gz")):
                    try:
                        obj = json.load(f)
                        if isinstance(obj, list):
                            for it in obj:
                                t = None
                                if isinstance(it, dict):
                                    t = it.get("text") or it.get("content")
                                elif isinstance(it, str):
                                    t = it
                                if isinstance(t, str) and t.strip():
                                    yield t.strip()
                                    n += 1
                                    if n >= max_lines: return
                        elif isinstance(obj, dict):
                            t = obj.get("text") or obj.get("content")
                            if isinstance(t, str) and t.strip():
                                yield t.strip()
                    except Exception:
                        pass
                    continue

                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    t = None
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            j = json.loads(line)
                            if isinstance(j, dict):
                                t = j.get("text") or j.get("content")
                        except Exception:
                            t = None
                    if t is None:
                        t = line
                    if isinstance(t, str) and t.strip():
                        yield t.strip()
                        n += 1
                        if n >= max_lines: return
        except Exception:
            continue

def build_token_stream(tok, texts_iter, target_tokens=500000, seed=2, add_special_tokens: bool = True):
    random.seed(seed)
    ids = []
    total = 0
    for t in texts_iter:
        enc = tok(t, add_special_tokens=add_special_tokens)["input_ids"]
        if not enc:
            continue
        if add_special_tokens:
            # tokenizer 默认（sanity_AB_opt30b.py 里用法）会把特殊符号一起拼进去
            ids.extend(enc)
            total += len(enc)
        else:
            # 兼容旧逻辑：手动追加 eos 作为分隔符
            ids.extend(enc + [tok.eos_token_id])
            total += len(enc) + 1
        if total >= target_tokens:
            break
    if total == 0:
        return None
    return torch.tensor(ids, dtype=torch.long)  # CPU

def batchify_stream_cpu(stream_cpu, seqlen, batch_size):
    N = stream_cpu.numel()
    assert N > seqlen + 1
    while True:
        starts = torch.randint(low=0, high=N - seqlen - 1, size=(batch_size,))
        x = torch.stack([stream_cpu[s:s+seqlen] for s in starts], dim=0)
        attn = torch.ones_like(x, dtype=torch.long)
        yield x, attn


# -------------------------
# Utkarsh-aligned C4 helpers (align with sanity_AB_opt30b.py)
# -------------------------
def _iter_jsonl_texts_gz(fp):
    with gzip.open(fp, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
                if isinstance(j, dict) and isinstance(j.get("text", None), str):
                    t = j["text"].strip()
                    if t:
                        yield t
            except Exception:
                continue


def build_doc_pool_from_c4_train_dir(
    data_dir: str,
    glob_pat: str = "c4-train.*.json.gz",
    pool_docs: int = 20000,
    scan_lines: int = 200000,
    seed: int = 2,
):
    files = sorted(glob.glob(os.path.join(data_dir, glob_pat)))
    if not files:
        raise FileNotFoundError(f"No files matched {glob_pat} under {data_dir}")

    rng = random.Random(seed)
    reservoir = []
    seen = 0

    for fp in files:
        for t in _iter_jsonl_texts_gz(fp):
            seen += 1
            if len(reservoir) < pool_docs:
                reservoir.append(t)
            else:
                j = rng.randint(1, seen)
                if j <= pool_docs:
                    reservoir[j - 1] = t
            if seen >= scan_lines:
                break
        if seen >= scan_lines:
            break

    if not reservoir:
        raise RuntimeError(f"Failed to read any text from {files[0]}")
    return reservoir, {"seen_docs": seen, "pool_docs": len(reservoir), "file": files[0]}


@torch.no_grad()
def build_utkarsh_like_valenc_from_pool(
    tok,
    doc_pool,
    seqlen: int,
    n_chunks: int = 256,
    val_seed: int = 0,
    max_tries_per_chunk: int = 2000,
):
    rng = random.Random(val_seed)
    valenc = []
    for _k in range(n_chunks):
        ok = False
        for _ in range(max_tries_per_chunk):
            text = doc_pool[rng.randint(0, len(doc_pool) - 1)]
            tmp = tok(text, return_tensors="pt")
            L = tmp.input_ids.shape[1]
            if L > seqlen:
                i = rng.randint(0, L - seqlen - 1)
                j = i + seqlen
                valenc.append(tmp.input_ids[:, i:j])
                ok = True
                break
        if not ok:
            raise RuntimeError(
                f"Unable to sample a doc with token_len > seqlen after {max_tries_per_chunk} tries. "
                f"Try increasing scan_lines/pool_docs."
            )
    return torch.hstack(valenc)  # [1, n_chunks*seqlen]

# -------------------------
# trainable selector
# -------------------------
def set_trainable(model, mode="vscale_only", train_last_n=0, unfreeze_balance: bool = False):
    """
    mode:
      - vscale_only: only vscale_log
      - up_vscale: out_proj_up + vscale_log
      - qkv_up_vscale: q/k/v low + out_proj_up + vscale_log
      - output_adapter_only: 仅方案 C（output_adapter_down/up + output_adapter_scale），需构建时 EIGEN_OUTPUT_ADAPTER=1
      - o_residual_only: 仅 o 侧低秩残差，需 EIGEN_O_RESIDUAL=1
      - output_adapter_o_residual: 同时训练 C 与 o_residual
      - all: everything

    train_last_n:
      - 0 => all layers
      - >0 => only last N decoder layers are unfrozen (for modes except 'all')

    unfreeze_balance: 解冻 apply_low_rank_balance_to_opt_model 挂载的 balance_* 子模块（与 qkv 同层范围）
    """
    for p in model.parameters():
        p.requires_grad_(False)

    def mark(module):
        for p in module.parameters():
            p.requires_grad_(True)

    if mode == "all":
        for p in model.parameters():
            p.requires_grad_(True)
        return

    layers = list(model.model.decoder.layers)
    if train_last_n and train_last_n > 0:
        layers = layers[-train_last_n:]

    for layer in layers:
        attn = layer.self_attn

        if hasattr(attn, "vscale_log"):
            try:
                attn.vscale_log.requires_grad_(True)
            except Exception:
                pass

        if mode in ["up_vscale", "qkv_up_vscale"]:
            if hasattr(attn, "out_proj_up"):
                mark(attn.out_proj_up)

        if mode == "qkv_up_vscale":
            for name in ["q_proj_low", "k_proj_low", "v_proj_low"]:
                if hasattr(attn, name):
                    mark(getattr(attn, name))

        if mode in ("output_adapter_only", "output_adapter_o_residual"):
            if getattr(attn, "output_adapter_down", None) is not None:
                mark(attn.output_adapter_down)
                mark(attn.output_adapter_up)
                if attn.output_adapter_scale is not None:
                    attn.output_adapter_scale.requires_grad_(True)

        if mode in ("o_residual_only", "output_adapter_o_residual"):
            if getattr(attn, "o_residual_down", None) is not None:
                mark(attn.o_residual_down)
                mark(attn.o_residual_up)
                if attn.o_residual_scale is not None:
                    attn.o_residual_scale.requires_grad_(True)

        if unfreeze_balance:
            for br in ("q", "k", "v"):
                core = getattr(attn, f"balance_{br}_core", None)
                if core is not None and getattr(core, "trainable", True):
                    mark(core)
                col = getattr(attn, f"balance_{br}_col", None)
                if col is not None and getattr(col, "mode", "trainable") == "trainable":
                    mark(col)

# -------------------------
# eval ppl (correct; labels=input_ids)
# -------------------------
@torch.no_grad()
def eval_ppl(model, stream_cpu, device, seqlen=2048, stride=2048, dtype=torch.bfloat16):
    """
    Strictly follow sanity_AB_opt30b.py eval_ppl_utkarsh_opt().
    - build [1, T] testenc
    - non-overlapping chunks of length seqlen
    - decoder(batch) -> lm_head(hidden)
    - shift logits/labels and use CrossEntropyLoss(mean)
    - nll_i = loss * seqlen, ppl = exp(sum(nll_i) / (nsamples * seqlen))
    """
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    if stream_cpu is None:
        return float("inf")

    if stream_cpu.dim() == 1:
        testenc = stream_cpu.unsqueeze(0)  # [1, T]
    elif stream_cpu.dim() == 2:
        testenc = stream_cpu  # [1, T]
    else:
        raise ValueError(f"eval_ppl expects [T] or [1,T], got shape={tuple(stream_cpu.shape)}")
    nsamples = testenc.numel() // seqlen
    if nsamples <= 0:
        return float("inf")

    loss_fct = torch.nn.CrossEntropyLoss()
    nlls = []

    for i in range(nsamples):
        batch = testenc[:, i * seqlen : (i + 1) * seqlen].to(device, non_blocking=True)

        outputs = model.model.decoder(batch)
        hidden_states = outputs[0]
        logits = model.lm_head(hidden_states)

        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:].to(model.lm_head.weight.device)

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.float() * seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    return ppl.item()


def _parse_layers(spec, n_layers):
    s = (spec or "all").strip().lower()
    if s in ("", "all"):
        return list(range(n_layers))
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    out = sorted({i for i in out if 0 <= i < n_layers})
    if not out:
        raise ValueError(f"invalid layers: {spec}")
    return out


def _basis_key(layer_idx, branch):
    return f"base_model.model.model.decoder.layers.{layer_idx}.self_attn.{branch}_proj_basis"


@torch.no_grad()
def iterative_refine_basis_inplace(student, tok, args, refine_round):
    from tools.build_activation_eigen_basis_opt import (
        collect_activation_basis_one_layer,
        iter_texts_from_dir as iter_texts_for_basis,
        build_token_stream as build_token_stream_for_basis,
    )

    n_layers = int(student.config.num_hidden_layers)
    num_heads = int(student.config.num_attention_heads)
    hidden = int(student.config.hidden_size)
    head_dim = hidden // num_heads
    layer_ids = _parse_layers(args.iter_refine_layers, n_layers)
    input_device = student.get_input_embeddings().weight.device

    cur_sd = student.state_dict()
    rank_kq = int(cur_sd[_basis_key(0, "q")].shape[1])
    rank_v = int(cur_sd[_basis_key(0, "v")].shape[1])
    rank_kq_per_head = rank_kq // num_heads
    rank_v_per_head = rank_v // num_heads

    basis_seed = int(args.seed + 1000 * max(refine_round, 1))
    texts = iter_texts_for_basis(
        args.data_dir,
        max_files=int(args.iter_refine_max_data_files),
        max_lines=int(args.iter_refine_max_data_lines),
    )
    stream = build_token_stream_for_basis(
        tok,
        texts,
        target_tokens=int(args.iter_refine_token_budget),
        seed=basis_seed,
    )
    if stream is None:
        raise RuntimeError("iter refine: no text read from data_dir")

    g = torch.Generator()
    g.manual_seed(basis_seed)
    N = stream.numel()
    batch_list = []
    for _ in range(int(args.iter_refine_max_batches)):
        starts = torch.randint(
            low=0,
            high=N - int(args.iter_refine_seqlen) - 1,
            size=(int(args.iter_refine_batch_size),),
            generator=g,
        )
        x = torch.stack([stream[s : s + int(args.iter_refine_seqlen)] for s in starts], dim=0)
        batch_list.append(x)

    print(
        f"[IterRefine] round={refine_round} layers={args.iter_refine_layers} "
        f"mode={args.iter_refine_weight_mode} token_budget={args.iter_refine_token_budget}"
    )
    out_state = {}
    was_training = student.training
    student.eval()
    for li in layer_ids:
        u_q, u_k, u_v, n_tok = collect_activation_basis_one_layer(
            student,
            layer_idx=li,
            num_heads=num_heads,
            head_dim=head_dim,
            rank_kq_per_head=rank_kq_per_head,
            rank_v_per_head=rank_v_per_head,
            repair_keqk=int(args.iter_refine_repair_keqk),
            data_iter=batch_list,
            input_device=input_device,
            weight_mode=args.iter_refine_weight_mode,
            nll_power=float(args.iter_refine_nll_power),
            nll_clip_percentile=float(args.iter_refine_nll_clip_percentile),
            nll_mix_lambda=float(args.iter_refine_nll_mix_lambda),
            nll_power_qk=args.iter_refine_nll_power_qk,
            nll_power_v=args.iter_refine_nll_power_v,
            nll_clip_percentile_qk=args.iter_refine_nll_clip_percentile_qk,
            nll_clip_percentile_v=args.iter_refine_nll_clip_percentile_v,
            nll_mix_lambda_qk=args.iter_refine_nll_mix_lambda_qk,
            nll_mix_lambda_v=args.iter_refine_nll_mix_lambda_v,
            fisher_batch_norm=1,
            teacher_model=None,
            teacher_distinct=False,
        )
        out_state[_basis_key(li, "q")] = u_q
        out_state[_basis_key(li, "k")] = u_k
        out_state[_basis_key(li, "v")] = u_v
        print(f"[IterRefine] layer={li} tokens={n_tok}")

    for li in range(n_layers):
        for br in ("q", "k", "v"):
            key = _basis_key(li, br)
            if key not in out_state:
                out_state[key] = cur_sd[key].float().cpu().clone()

    miss, unexp = student.load_state_dict(out_state, strict=False)
    print(f"[IterRefine] hot-swap done: missing={len(miss)} unexpected={len(unexp)}")

    if args.iter_refine_save_basis_dir:
        os.makedirs(args.iter_refine_save_basis_dir, exist_ok=True)
        save_p = os.path.join(args.iter_refine_save_basis_dir, f"iter_refine_round{refine_round}.pt")
        torch.save(out_state, save_p)
        print(f"[IterRefine] basis saved: {save_p}")

    if was_training:
        student.train()

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="", help="base full model path (用于 PPL 对比). 不填时默认用 --tok 作为 base 权重路径。")
    ap.add_argument("--cmp", required=True)
    ap.add_argument("--low_json", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--data_dir", required=True)

    ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16"])
    ap.add_argument("--device", default="cuda:0")  # under CUDA_VISIBLE_DEVICES=2,3 => cuda:0=phys2, cuda:1=phys3

    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--token_budget", type=int, default=600000)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1)

    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--grad_accum", type=int, default=1)

    ap.add_argument(
        "--mode",
        default="vscale_only",
        choices=[
            "vscale_only",
            "up_vscale",
            "qkv_up_vscale",
            "output_adapter_only",
            "o_residual_only",
            "output_adapter_o_residual",
            "all",
        ],
    )
    ap.add_argument("--train_last_n", type=int, default=0,
                    help="only unfreeze last N decoder layers (0=all layers)")

    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_tokens", type=int, default=200000)
    ap.add_argument(
        "--no_eval",
        action="store_true",
        help="跳过所有 PPL 评估（包含初始 base/cmp 与训练过程 eval_every），只训练并保存模型。",
    )
    ap.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="保存 finetuned 模型到的目录；默认自动在 /diskF/jiaqizhang 下创建时间戳新目录。",
    )
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument(
        "--loss_jsonl",
        type=str,
        default="finetune_loss.jsonl",
        help="write per-step loss records to this jsonl file",
    )
    ap.add_argument("--iter_refine_every", type=int, default=0, help=">0: 每 N 步做一次在线 basis 重估")
    ap.add_argument("--iter_refine_start", type=int, default=0, help="从第几步开始允许在线重估")
    ap.add_argument("--iter_refine_layers", type=str, default="all", help='在线重估层范围，如 "all" 或 "32-47"')
    ap.add_argument("--iter_refine_weight_mode", type=str, default="nll", choices=["uniform", "nll"])
    ap.add_argument("--iter_refine_repair_keqk", type=int, default=1, choices=[0, 1])
    ap.add_argument("--iter_refine_token_budget", type=int, default=800000)
    ap.add_argument("--iter_refine_max_batches", type=int, default=64)
    ap.add_argument("--iter_refine_batch_size", type=int, default=1)
    ap.add_argument("--iter_refine_seqlen", type=int, default=512)
    ap.add_argument("--iter_refine_max_data_files", type=int, default=120)
    ap.add_argument("--iter_refine_max_data_lines", type=int, default=800000)
    ap.add_argument("--iter_refine_nll_power", type=float, default=1.0)
    ap.add_argument("--iter_refine_nll_clip_percentile", type=float, default=99.5)
    ap.add_argument("--iter_refine_nll_mix_lambda", type=float, default=0.9)
    ap.add_argument("--iter_refine_nll_mix_lambda_qk", type=float, default=0.95)
    ap.add_argument("--iter_refine_nll_mix_lambda_v", type=float, default=0.25)
    ap.add_argument("--iter_refine_nll_clip_percentile_qk", type=float, default=99.5)
    ap.add_argument("--iter_refine_nll_clip_percentile_v", type=float, default=100.0)
    ap.add_argument("--iter_refine_nll_power_qk", type=float, default=1.08)
    ap.add_argument("--iter_refine_nll_power_v", type=float, default=0.95)
    ap.add_argument("--iter_refine_save_basis_dir", type=str, default="", help="可选：每轮在线重估的 basis 落盘目录")

    # Low-rank 几何稳定：orthogonal + column balance（见 models/low_rank_balance.py）
    ap.add_argument(
        "--use_core_rebalance",
        type=int,
        default=0,
        choices=[0, 1],
        help="在低秩 W 左侧挂 headwise Cayley 正交重参数（不改 rank）",
    )
    ap.add_argument(
        "--use_column_balance",
        type=int,
        default=0,
        choices=[0, 1],
        help="按输入维做列尺度平衡（不改 rank）",
    )
    ap.add_argument("--core_rebalance_trainable", type=int, default=1, choices=[0, 1])
    ap.add_argument("--core_rebalance_orth_rank", type=int, default=8)
    ap.add_argument("--balance_strength", type=float, default=0.25)
    ap.add_argument("--balance_mode", type=str, default="trainable", choices=["static", "trainable"])
    ap.add_argument(
        "--balance_target",
        type=str,
        default="qkv",
        help="qkv | qk | qv | q | k | v | all",
    )
    ap.add_argument("--balance_lr_mult_k", type=float, default=0.5)
    ap.add_argument("--balance_lr_mult_v", type=float, default=1.0)
    ap.add_argument(
        "--ft_orth_reg_weight",
        type=float,
        default=0.0,
        help="正交正则 ||H^T H - I||^2 权重（加到 CE 上）",
    )
    ap.add_argument(
        "--ft_balance_reg_weight",
        type=float,
        default=0.0,
        help="列平衡正则 ||log_gamma||^2 权重",
    )

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if not args.base:
        # OPT/T5 等场景下 tokenizer 目录通常与 base 权重目录同源
        args.base = args.tok

    low_rank_config = json.load(open(args.low_json))

    tok = AutoTokenizer.from_pretrained(args.tok, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count() =", torch.cuda.device_count())
    print("[Args]", vars(args))

    # 须在 from_pretrained 构造层之前设置，才会创建对应子模块
    if args.mode in ("output_adapter_only", "output_adapter_o_residual"):
        os.environ["EIGEN_OUTPUT_ADAPTER"] = "1"
    if args.mode in ("o_residual_only", "output_adapter_o_residual"):
        os.environ["EIGEN_O_RESIDUAL"] = "1"

    print("[Load] student CMP...")
    student = OPTForCausalLM_EigenAttn.from_pretrained(
        args.cmp,
        local_files_only=True,
        torch_dtype=dtype,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
        low_rank_config=low_rank_config,
    )
    student.config.use_cache = False
    if args.grad_ckpt == 1 and hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
        print("[GradCkpt] enabled")
    student.eval()

    # 与 main_eigen_attn 一致：ft_train_last_n 控制 balance 挂载层范围
    setattr(args, "ft_train_last_n", int(args.train_last_n))
    if int(args.use_core_rebalance) or int(args.use_column_balance):
        apply_low_rank_balance_to_opt_model(student, args, logger=print)
        print(
            "[low_rank_balance] finetune: 建议与 mode=qkv_up_vscale 或 all 联用以训练低秩投影；"
            "balance 子模块已按 unfreeze_balance 解冻。"
        )

    set_trainable(
        student,
        args.mode,
        train_last_n=args.train_last_n,
        unfreeze_balance=bool(int(args.use_core_rebalance) or int(args.use_column_balance)),
    )
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in trainable_params)
    total = sum(p.numel() for p in student.parameters())
    print(f"[Trainable] {trainable:,} / {total:,} ({100.0*trainable/total:.6f}%) mode={args.mode} lastN={args.train_last_n}")

    # Data stream (CPU)
    texts = iter_texts_from_dir(args.data_dir)
    stream_cpu = build_token_stream(
        tok,
        texts,
        target_tokens=args.token_budget,
        seed=args.seed,
        add_special_tokens=True,
    )
    assert stream_cpu is not None, "No data read from data_dir"
    print(f"[Data] token_stream={stream_cpu.numel()} seqlen={args.seqlen} bs={args.batch_size}")

    loader = batchify_stream_cpu(stream_cpu, args.seqlen, args.batch_size)

    # Eval stream
    texts2 = iter_texts_from_dir(args.data_dir)
    eval_stream_cpu = build_token_stream(
        tok,
        texts2,
        target_tokens=args.eval_tokens,
        seed=args.seed + 1,
        add_special_tokens=True,
    )
    assert eval_stream_cpu is not None

    eval_seqlen = 2048

    if args.no_eval:
        print("[Eval] --no_eval enabled: skip PPL eval (BASE/CMP).")
    else:
        print("[Eval] initial PPL ... (BASE vs CMP)")

    # Build Utkarsh-aligned testenc (only needed for eval)
    testenc = None
    base_ppl = None
    if not args.no_eval:
        pool, pool_info = build_doc_pool_from_c4_train_dir(
            args.data_dir,
            glob_pat=getattr(args, "c4_glob", "c4-train.*.json.gz"),
            pool_docs=getattr(args, "c4_pool_docs", 20000),
            scan_lines=getattr(args, "c4_scan_lines", 200000),
            seed=args.seed,
        )
        print(f"[C4-Pool] {pool_info}")
        testenc = build_utkarsh_like_valenc_from_pool(
            tok,
            pool,
            seqlen=eval_seqlen,
            n_chunks=getattr(args, "c4_val_chunks", 256),
            val_seed=getattr(args, "c4_val_seed", 0),
        )
        assert testenc.shape[1] == getattr(args, "c4_val_chunks", 256) * eval_seqlen, (
            tuple(testenc.shape),
            getattr(args, "c4_val_chunks", 256),
            eval_seqlen,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            args.base,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        base_model.config.use_cache = False
        base_ppl = eval_ppl(
            base_model,
            testenc,
            device=args.device,
            seqlen=eval_seqlen,
            stride=2048,
            dtype=dtype,
        )
        del base_model
        torch.cuda.empty_cache()
        print(f"BASE PPL init: {base_ppl:.6f}")

        cmp_ppl = eval_ppl(
            student,
            testenc,
            device=args.device,
            seqlen=eval_seqlen,
            stride=2048,
            dtype=dtype,
        )
        print(f"CMP  PPL init: {cmp_ppl:.6f}  ratio={cmp_ppl/(base_ppl+1e-12):.6f}")

    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    student.train()
    opt.zero_grad(set_to_none=True)

    # Write per-step loss for later plotting/debugging.
    loss_f = open(args.loss_jsonl, "w", encoding="utf-8")
    try:
        pbar = tqdm(range(1, args.steps + 1), desc=f"finetune({args.mode})", miniters=10)
        for step in pbar:
            x_cpu, attn_cpu = next(loader)
            x = x_cpu.to(args.device, non_blocking=True)
            attn = attn_cpu.to(args.device, non_blocking=True)
            labels = x  # ✅ PPL 对应的监督

            with torch.autocast(device_type="cuda", dtype=dtype):
                out = student(input_ids=x, attention_mask=attn, labels=labels)
                # out.loss is already mean over tokens; log raw scalar before grad_accum scaling
                raw_loss = float(out.loss.detach().float().item())
                loss = out.loss / max(args.grad_accum, 1)
                ow = float(getattr(args, "ft_orth_reg_weight", 0.0) or 0.0)
                bw = float(getattr(args, "ft_balance_reg_weight", 0.0) or 0.0)
                if ow > 0 or bw > 0:
                    orth_sum, bal_sum = collect_orth_balance_regularizers(student)
                    reg = ow * orth_sum.to(loss.dtype).to(loss.device) + bw * bal_sum.to(
                        loss.dtype
                    ).to(loss.device)
                    loss = loss + reg

            loss.backward()

            if step % args.grad_accum == 0:
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if step % 20 == 0:
                # 这里打印的是 (per-token) NLL（raw_loss）
                print(f"[Step {step:6d}] nll={raw_loss:.6f}")
                try:
                    pbar.set_postfix({"nll": f"{raw_loss:.4f}"})
                except Exception:
                    pass

            # Always log per-step.
            loss_f.write(json.dumps({"step": step, "loss": raw_loss, "t": time.time()}) + "\n")

            if (
                args.iter_refine_every > 0
                and step >= args.iter_refine_start
                and step % args.iter_refine_every == 0
            ):
                student.eval()
                torch.cuda.empty_cache()
                refine_round = step // args.iter_refine_every
                iterative_refine_basis_inplace(student, tok, args, refine_round=refine_round)
                torch.cuda.empty_cache()
                student.train()

            if (not args.no_eval) and args.eval_every > 0 and step % args.eval_every == 0:
                student.eval()
                cmp_ppl = eval_ppl(student, testenc, device=args.device, seqlen=eval_seqlen, stride=2048, dtype=dtype)
                print(
                    f"[Eval step {step}] BASE PPL: {base_ppl:.6f}  CMP PPL: {cmp_ppl:.6f}  ratio={cmp_ppl/(base_ppl+1e-12):.6f}"
                )
                student.train()
    finally:
        loss_f.close()

    # Save finetuned model
    if args.save_dir:
        save_dir = args.save_dir
    else:
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        save_dir = f"/diskF/jiaqizhang/finetune_opt30b_eigen_{args.mode}_{ts}"

    os.makedirs(save_dir, exist_ok=True)
    print(f"[Save] writing model to: {save_dir}")
    # Save HF weights/config
    student.save_pretrained(save_dir)
    # Save tokenizer (so future eval scripts can load consistently)
    try:
        tok.save_pretrained(save_dir)
    except Exception:
        pass
    # Save low-rank config for compatibility with your loader (LMClass / other scripts)
    try:
        shutil.copy2(args.low_json, os.path.join(save_dir, "low_rank_config.json"))
    except Exception:
        pass

    print("[Done] training finished and model saved.")

if __name__ == "__main__":
    main()