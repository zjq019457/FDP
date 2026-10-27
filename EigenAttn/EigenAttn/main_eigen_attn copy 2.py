# main_eigen_attn.py
# ============================================================
# Key fixes (for your OPT-30B / A100-80G multi-GPU + OOM issues):
# 1) DO NOT force fp32 in evaluate()  (your old code did lm.model.float()).
# 2) If args.multigpu: use layer-wise placement across ALL visible GPUs
#    via map_layers_to_multi_gpus(...) BEFORE eigenattn() and evaluate().
# 3) If launched with torchrun (WORLD_SIZE>1) + args.multigpu:
#    only rank0 runs and uses all GPUs (single-process multi-GPU).
#    Other ranks exit to avoid "4 processes all on GPU0".
# 4) Provide --dtype {bf16,fp16,fp32} and keep model in bf16 by default on A100.
# 5) Avoid overriding HF_HOME/HF_DATASETS_CACHE if user already exported them.
# ============================================================

import os
import sys
import time
import random
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoConfig

# Your project imports
from models.LMClass import LMClass
from datautils import get_loaders
from decompose.eigen_attn import eigenattn
import utils
from categories import subcategories, categories

from parallel_utils import map_layers_to_multi_gpus  # keep mapping util

# (Optional) these are imported in your original file; keep them if other modules use them
from models.modelling_mpt_eigen_attn import MptEigenAttention
from models.modelling_llama_eigen_attn import LlamaEigenAttention
from transformers.models.mpt.modeling_mpt import MptAttention

torch.serialization.add_safe_globals([BatchEncoding])
torch.backends.cudnn.benchmark = True

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "mpt-7b",
    "mpt-30b",
    "Llama-3-8b",
    "Llama-3-80b",
]


def _dist_info():
    """Return (rank, local_rank, world_size) from torchrun env vars if present."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_dtype(dtype_str: str):
    s = (dtype_str or "").lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_str}")


def get_model_size(model):
    num_elements = 0
    for key in model.model.state_dict().keys():
        if "lm_head" in key:
            continue
        param = model.model.state_dict()[key]
        num_elements += param.numel()
    # (assuming 16-bit storage in this reporting)
    model_size = num_elements * 16 / (8 * 1024 * 1024 * 1024)
    return num_elements, model_size


def get_kvcache_size_opt(lm):
    num_elements = 0
    seq_len = lm.seqlen
    for key in lm.model.state_dict().keys():
        if (
            "k_proj.weight" in key
            or "v_proj.weight" in key
            or "v_proj_low.weight" in key
            or "k_proj_low.weight" in key
        ):
            num_elements += seq_len * lm.model.state_dict()[key].shape[0]
    cache_size = num_elements * 16 / (8 * 1024 * 1024 * 1024)
    return cache_size


def get_kvcache_size_mpt(lm, args):
    num_elements = 0
    seq_len = lm.seqlen
    total_rankkq = 0
    total_rankv = 0
    i = 0
    for name, m in lm.model.named_modules():
        if (".attn" in name and isinstance(m, MptAttention)):
            num_elements += seq_len * m.Wqkv.weight.shape[0] * 2 / 3
        elif (".attn" in name and isinstance(m, MptEigenAttention)):
            num_elements += seq_len * (m.rank_kq + m.rank_v)
            total_rankkq += m.rank_kq
            total_rankv += m.rank_v
            i += 1
    cache_size = (num_elements * 16) / (8 * 1024 * 1024 * 1024)
    return cache_size


def get_kvcache_size_llama(lm):
    num_elements = 0
    seq_len = lm.seqlen
    for key in lm.model.state_dict().keys():
        if (
            "k_proj.weight" in key
            or "v_proj.weight" in key
            or "v_proj_low.weight" in key
            or "k_proj_low.weight" in key
        ):
            num_elements += seq_len * lm.model.state_dict()[key].shape[0]
    cache_size = num_elements * 16 / (8 * 1024 * 1024 * 1024)
    return cache_size


def _place_model_multigpu(lm, args, logger):
    """
    Layer-wise pipeline placement across all visible GPUs.
    This is the "one layer group per GPU" style mapping (not DDP).
    """
    if not args.multigpu:
        return

    if not torch.cuda.is_available():
        logger.warning("CUDA not available; multigpu mapping skipped.")
        return

    if "opt" in args.net.lower():
        map_layers_to_multi_gpus(lm.model.model.decoder.layers)
        input_device = lm.model.model.decoder.layers[0].device
        output_device = lm.model.model.decoder.layers[-1].device
        lm._device = input_device
        # Your original code asserts equal; keep it
        assert input_device == output_device
        lm.model.model.decoder.embed_positions.to(input_device)
        lm.model.model.decoder.embed_tokens.to(input_device)
        lm.model.model.decoder.final_layer_norm.to(output_device)
        lm.model.lm_head.to(output_device)

    elif "llama" in args.net.lower() or "mistral" in args.net.lower() or "qwen2" in args.net.lower():
        map_layers_to_multi_gpus(lm.model.model.layers)
        input_device = lm.model.model.layers[0].device
        output_device = lm.model.model.layers[-1].device
        assert input_device == output_device
        lm._device = input_device
        lm.model.model.embed_tokens.to(input_device)
        lm.model.model.norm.to(output_device)
        lm.model.lm_head.to(output_device)

    elif "falcon" in args.net.lower():
        map_layers_to_multi_gpus(lm.model.transformer.h)
        input_device = lm.model.transformer.h[0].device
        output_device = lm.model.transformer.h[-1].device
        assert input_device == output_device
        lm._device = input_device
        lm.model.transformer.word_embeddings.to(input_device)
        lm.model.transformer.ln_f.to(output_device)
        lm.model.lm_head.to(output_device)

    elif "mpt" in args.net.lower():
        map_layers_to_multi_gpus(lm.model.transformer.blocks)
        input_device = lm.model.transformer.blocks[0].device
        output_device = lm.model.transformer.blocks[-1].device
        assert input_device == output_device
        lm._device = input_device
        lm.model.transformer.wte.to(input_device)
        lm.model.transformer.norm_f.to(output_device)
        lm.model.lm_head.to(output_device)

    else:
        logger.warning(f"multigpu mapping not implemented for net={args.net}")


@torch.no_grad()
def evaluate(lm, args, logger):
    """
    IMPORTANT: Do NOT convert the whole model to fp32 here.
    That was the biggest VRAM killer in your old code.
    """
    results = {}

    # Ensure correct placement for evaluation too (safe even if already placed)
    if args.multigpu:
        _place_model_multigpu(lm, args, logger)
    else:
        # single GPU / single device
        if "opt" in args.net.lower():
            lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
        elif "llama" in args.net.lower() or "mistral" in args.net.lower() or "qwen2" in args.net.lower():
            lm.model = lm.model.to(lm.device)
        elif "mpt" in args.net.lower():
            lm.model.transformer = lm.model.transformer.to(lm.device)

    if args.eval_ppl:
        for dataset in ["wikitext2", "c4"]:
            cache_testloader = f"{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache"
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                logger.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=lm.seqlen,
                )
                torch.save(testloader, cache_testloader)

            testenc = testloader if "c4" in dataset else testloader.input_ids
            nsamples = testenc.numel() // lm.seqlen

            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()

            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen): ((i + 1) * lm.seqlen)].to(lm.device)

                if "opt" in args.net.lower():
                    outputs = lm.model.model.decoder(batch)
                elif "llama" in args.net.lower() or "mistral" in args.net.lower() or "qwen2" in args.net.lower():
                    outputs = lm.model.model(batch)
                elif "falcon" in args.model:
                    outputs = lm.model.transformer(batch)
                elif "mpt" in args.net.lower():
                    outputs = lm.model.transformer(batch)
                else:
                    raise RuntimeError(f"Unsupported net for eval_ppl: {args.net}")

                hidden_states = outputs[0]
                logits = lm.model.lm_head(hidden_states)

                # numeric guard
                logits = torch.clamp(logits, -20, 20)

                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen): ((i + 1) * lm.seqlen)][:, 1:].to(lm.device)

                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("⚠ NaN/Inf detected, skipping batch")
                    continue

                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)

                if i == args.limit:
                    break

            if len(nlls) == 0:
                logger.error("❌ All batches produced NaN. PPL invalid for this model.")
                results[dataset] = float("nan")
            else:
                total_tokens = len(nlls) * lm.seqlen
                ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
                logger.info(f"{dataset} : {ppl.item()}")
                results[dataset] = ppl.item()

            lm.model.config.use_cache = use_cache

    if args.tasks != "":
        import lm_eval
        task_manager = lm_eval.tasks.TaskManager()
        t_results = lm_eval.simple_evaluate(
            model=lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            task_manager=task_manager,
        )

        results.update(t_results)
        logger.info(results)
        pprint(results)

        if "mmlu" in args.tasks:
            all_cors = []
            all_cors_norm = []
            subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
            cat_cors = {cat: [] for cat in categories}
            cat_cors_norm = {cat: [] for cat in categories}

            for key in t_results["results"].keys():
                if "hendrycksTest" not in key:
                    continue
                subject = key.split("-")[-1]
                cors = t_results["results"][key]["acc"]
                cors_norm = t_results["results"][key]["acc_norm"]
                subcats = subcategories[subject]

                for subcat in subcats:
                    subcat_cors[subcat].append(cors)
                    for cat in categories.keys():
                        if subcat in categories[cat]:
                            cat_cors[cat].append(cors)
                            cat_cors_norm[cat].append(cors_norm)
                    all_cors.append(cors)
                    all_cors_norm.append(cors_norm)

            for cat in cat_cors:
                cat_acc = np.mean(cat_cors[cat])
                logger.info("Average accuracy {:.4f} - {}".format(cat_acc, cat))
            weighted_acc = np.mean(all_cors)
            logger.info("Average accuracy: {:.4f}".format(weighted_acc))

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="model name or model path")
    parser.add_argument("--cache_dir", default="./HF_cache", type=str, help="cache dir for datasets / hf")
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix", "pile", "alpaca"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")

    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)

    # IMPORTANT: this is "single-process multi-GPU layer mapping"
    parser.add_argument("--multigpu", action="store_true", help="map model layers to multiple gpus (single-process)")

    parser.add_argument(
        "--attn_implementation",
        type=str,
        required=False,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument("--net", type=str, default=None, choices=net_choices)

    parser.add_argument("--load_low_rank", action="store_true", help="load already compressed model")
    parser.add_argument("--load_peft_model", action="store_true", help="load the peft model")
    parser.add_argument("--peft_model_path", default=None, help="path to finetuned model")
    parser.add_argument("--save_dir", default=None, help="path to save compressed/finetuned model")
    parser.add_argument("--output_dir", default=None, help="path to save log file")

    parser.add_argument("--evaluate_baseline", action="store_true")
    parser.add_argument("--avg_dim", type=int)
    parser.add_argument("--error_budget", type=float, default=0.025)
    parser.add_argument("--fine_tune", action="store_true", help="finetune model after compressing")

    # NEW: dtype control (VRAM!)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                        help="model compute dtype (A100 recommended: bf16)")

    # NEW: seqlen control
    parser.add_argument("--seqlen", type=int, default=2048, help="sequence length used for calibration/eval")

    args = parser.parse_args()

    # ----- torchrun safety for multigpu layer mapping -----
    rank, local_rank, world_size = _dist_info()
    if world_size > 1 and args.multigpu:
        # multigpu here means "one process uses all GPUs"; torchrun would spawn many.
        # Only keep rank0 to avoid 4 processes fighting for GPU0.
        if rank != 0:
            # Exit quietly
            return

    _set_seed(args.seed)

    # Respect user-exported HF env; only set if missing.
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)
        os.environ.setdefault("HF_HOME", args.cache_dir)

    if args.net is None:
        args.net = args.model.split("/")[-1]

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir) if args.output_dir else Path("./outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = utils.create_logger(output_dir)

    logger.info(args)
    logger.info(f"[dist] world_size={world_size} rank={rank} local_rank={local_rank}")

    # ----- load model -----
    args.model_family = args.net.split("-")[0]
    lm = LMClass(args)

    lm.seqlen = int(args.seqlen)
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    # enforce dtype (VRAM)
    dtype = _resolve_dtype(args.dtype)
    try:
        lm.model = lm.model.to(dtype=dtype)
    except Exception as e:
        logger.warning(f"Could not cast model to dtype={args.dtype} via .to(dtype=..): {e}")

    logger.info(f"error_budget : {args.error_budget}")
    logger.info(str(lm.model.config))

    # If baseline-only, do it early
    if args.evaluate_baseline:
        logger.info("=== Evaluating baselines ===")
        model_params, _ = get_model_size(lm)

        if "opt" in args.net.lower():
            cache_size = get_kvcache_size_opt(lm)
        elif "mpt" in args.net.lower():
            cache_size = get_kvcache_size_mpt(lm, args)
        elif "llama" in args.net.lower() or "mistral" in args.net.lower() or "qwen2" in args.net.lower():
            cache_size = get_kvcache_size_llama(lm)
        else:
            cache_size = float("nan")

        logger.info(f"Baseline model parameters : {model_params/1e9} Billion")
        logger.info(f"Baseline model KV Cache Size : {cache_size} GB for batch size of 1")

        # Place model across GPUs for eval if requested
        if args.multigpu:
            _place_model_multigpu(lm, args, logger)

        evaluate(lm, args, logger)
        return

    # ----- MULTI-GPU LAYER PLACEMENT BEFORE eigenattn() -----
    if args.multigpu:
        _place_model_multigpu(lm, args, logger)

    logger.info("=== start low rank decomposition ===")
    tick = time.time()

    # ----- load calibration dataset -----
    cache_dataloader = f"{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache"
    if os.path.exists(cache_dataloader):
        dataloader = torch.load(cache_dataloader)
        logger.info(f"load calibration from {cache_dataloader}")
    else:
        dataloader, _ = get_loaders(
            args.calib_dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=lm.seqlen,
        )
        torch.save(dataloader, cache_dataloader)

    args.eigen_attn_params = {
        "threshold": 1.0,
        "avg_dim_features": args.avg_dim,
        "error_budget": args.error_budget,
    }

    if not args.load_low_rank:
        eigenattn(lm, args, dataloader, logger)
    else:
        logger.info("You are already loading a low rank compressed model.")

    # ----- build a clean low-rank model (your original behavior) -----
    if "opt" in args.net.lower():
        layers = lm.model.model.decoder.layers
        low_rank_config = torch.zeros(len(layers), 2)
        for i in range(len(layers)):
            low_rank_config[i][0] = layers[i].self_attn.rank_kq
            low_rank_config[i][1] = layers[i].self_attn.rank_v

        config = AutoConfig.from_pretrained(
            args.model,
            attn_implementation=args.attn_implementation,
            cache_dir=args.cache_dir,
        )
        from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

        model2 = OPTForCausalLM_EigenAttn.from_pretrained(
            args.model,
            config=config,
            low_rank_config=low_rank_config,
            device_map="cpu",
            torch_dtype=torch.float16,   # store weights as fp16
            cache_dir=args.cache_dir,
        )

        layers2 = model2.model.decoder.layers
        for i in range(len(layers2)):
            layers2[i].self_attn.k_proj_low.weight.data = layers[i].self_attn.k_proj.weight.data
            layers2[i].self_attn.k_proj_low.bias.data = layers[i].self_attn.k_proj.bias.data

            layers2[i].self_attn.q_proj_low.weight.data = layers[i].self_attn.q_proj.weight.data
            layers2[i].self_attn.q_proj_low.bias.data = layers[i].self_attn.q_proj.bias.data

            layers2[i].self_attn.v_proj_low.weight.data = layers[i].self_attn.v_proj.weight.data
            layers2[i].self_attn.v_proj_low.bias.data = layers[i].self_attn.v_proj.bias.data

            layers2[i].self_attn.out_proj_up.weight.data = layers[i].self_attn.out_proj_up.weight.data
            layers2[i].self_attn.out_proj_up.bias.data = layers[i].self_attn.out_proj_up.bias.data

        lm.model = model2

        if args.save_dir:
            lm.model.save_pretrained(args.save_dir)
            lm.tokenizer.save_pretrained(args.save_dir)
            torch.save(low_rank_config, os.path.join(args.save_dir, "low_rank_config.pt"))

    elif "mpt" in args.net.lower():
        layers = lm.model.transformer.blocks
        low_rank_config = torch.zeros(len(layers), 2)
        for i in range(len(layers)):
            low_rank_config[i][0] = layers[i].attn.rank_kq
            low_rank_config[i][1] = layers[i].attn.rank_v

        from models.modelling_mpt_eigen_attn import MptForCausalLM_EigenAttn
        config = AutoConfig.from_pretrained(args.model, attn_implementation=args.attn_implementation, cache_dir=args.cache_dir)
        model2 = MptForCausalLM_EigenAttn.from_pretrained(
            args.model,
            config=config,
            low_rank_config=low_rank_config,
            device_map="cpu",
            torch_dtype=torch.float16,
            cache_dir=args.cache_dir,
        )
        layers2 = model2.transformer.blocks
        for i in range(len(layers2)):
            layers2[i].attn.Wqkv_low.weight.data = layers[i].attn.Wqkv.weight.data
            layers2[i].attn.out_proj_up.weight.data = layers[i].attn.out_proj.weight.data

        lm.model = model2

        if args.save_dir:
            lm.model.save_pretrained(args.save_dir)
            lm.tokenizer.save_pretrained(args.save_dir)
            torch.save(low_rank_config, os.path.join(args.save_dir, "low_rank_config.pt"))

    elif "llama" in args.net.lower():
        layers = lm.model.model.layers
        low_rank_config = []
        for i in range(len(layers)):
            low_rank_config.append([int(layers[i].self_attn.rank_kq), int(layers[i].self_attn.rank_v)])

        from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn
        config = AutoConfig.from_pretrained(args.model, attn_implementation=args.attn_implementation, cache_dir=args.cache_dir)

        model2 = LlamaForCausalLM_EigenAttn.from_pretrained(
            args.model,
            config=config,
            low_rank_config=low_rank_config,
            device_map="cpu",
            torch_dtype=torch.float16,
            cache_dir=args.cache_dir,
        )

        layers2 = model2.model.layers
        bias = config.attention_bias
        for i in range(len(layers2)):
            layers2[i].self_attn.k_proj_low.weight.data = layers[i].self_attn.k_proj_low.weight.data
            if bias:
                layers2[i].self_attn.k_proj_low.bias.data = layers[i].self_attn.k_proj_low.bias.data
            layers2[i].self_attn.k_proj_up.weight.data = layers[i].self_attn.k_proj_up.weight.data

            layers2[i].self_attn.v_proj_low.weight.data = layers[i].self_attn.v_proj_low.weight.data
            if bias:
                layers2[i].self_attn.v_proj_low.bias.data = layers[i].self_attn.v_proj_low.bias.data

            layers2[i].self_attn.o_proj_up.weight.data = layers[i].self_attn.o_proj_up.weight.data
            if bias:
                layers2[i].self_attn.o_proj_up.bias.data = layers[i].self_attn.o_proj_up.bias.data

        lm.model = model2

        if args.save_dir:
            lm.model.save_pretrained(args.save_dir)
            lm.tokenizer.save_pretrained(args.save_dir)
            torch.save(low_rank_config, os.path.join(args.save_dir, "low_rank_config.pt"))

    logger.info(f"Decomposition time: {time.time() - tick:.2f}s")

    # ----- Evaluate compressed model -----
    logger.info("=== Evaluating compressed model ===")
    model_params, _ = get_model_size(lm)
    logger.info(f"Compressed model parameters : {model_params/1e9} Billion")

    if "opt" in args.net.lower():
        cache_size = get_kvcache_size_opt(lm)
    elif "mpt" in args.net.lower():
        cache_size = get_kvcache_size_mpt(lm, args)
    elif "llama" in args.net.lower():
        cache_size = get_kvcache_size_llama(lm)
    else:
        cache_size = float("nan")

    logger.info(f"Compressed model KV Cache Size : {cache_size} GB for batch size of 1")

    # Put compressed model on requested devices for eval
    if args.multigpu:
        _place_model_multigpu(lm, args, logger)

    evaluate(lm, args, logger)

    # ----- Save final model -----
    save_dir = Path(args.output_dir) / "final_model"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving compressed model to: {save_dir}")

    lm.model.save_pretrained(save_dir, safe_serialization=True)
    lm.tokenizer.save_pretrained(save_dir)
    logger.info("✅ Model saved successfully.")

    # Optional fine-tune (kept as-is, but note: this section can OOM easily on 30B)
    if args.fine_tune:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        import transformers
        from time import perf_counter

        ds = load_dataset("tatsu-lab/alpaca")
        ds = ds.remove_columns(["input", "output", "instruction"])
        traindata = ds["train"]

        model = lm.model
        model.config.use_cache = False

        tokenizer = lm.tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        lora_alpha = 64
        lora_dropout = 0.05
        lora_r = 64

        if "mpt" in args.net.lower():
            peft_config = LoraConfig(
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["Wqkv_low", "out_proj_up"],
                r=lora_r,
                bias="none",
                task_type="CAUSAL_LM",
            )
        elif "llama" in args.net.lower():
            peft_config = LoraConfig(
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["k_proj_low", "k_proj_up", "v_proj_low", "o_proj_up"],
                r=lora_r,
                bias="none",
                task_type="CAUSAL_LM",
            )
        else:
            raise RuntimeError("fine_tune only implemented for mpt/llama in this script.")

        output_dir = os.path.join(args.save_dir, "peft_model") if args.save_dir else str(Path(args.output_dir) / "peft_model")

        training_arguments = transformers.TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            optim="adamw_torch",
            learning_rate=2e-4,
            lr_scheduler_type="linear",
            save_strategy="steps",
            save_steps=500,
            logging_steps=10,
            num_train_epochs=1,
            max_steps=2000,
            fp16=True,
            push_to_hub=False,
        )

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)

        from trl import SFTTrainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=traindata,
            peft_config=peft_config,
            dataset_text_field="text",
            args=training_arguments,
            tokenizer=tokenizer,
            packing=False,
            max_seq_length=args.seqlen,
        )

        start_time = perf_counter()
        trainer.train()
        end_time = perf_counter()
        print(f"Time taken for training: {end_time - start_time} seconds")


if __name__ == "__main__":
    print(sys.argv)
    main()
