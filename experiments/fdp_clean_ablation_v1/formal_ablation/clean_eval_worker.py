#!/usr/bin/env python3
"""
CLEAN SVD PROJECTION & EVALUATION PIPELINE
=========================================
Accurately slices the basis according to each config's target ranks,
projects weights, and evaluates WikiText-2, C4, and 5 Zero-Shot tasks.
"""
import os, sys, json, math, shutil, subprocess, argparse
from pathlib import Path
from datetime import datetime, timezone
import torch
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

WORKSPACE   = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1")
FORMAL_DIR  = WORKSPACE / "formal_ablation"
PER_WIN_DIR = FORMAL_DIR / "CORE_FORMAL_PER_WINDOW"
MODEL_BASE  = "/diskD/jiaqizhang/Llama-3"
REPO        = Path("/home/bai-project/jiaqizhang/EigenAttn")
PPL_SCRIPT  = REPO / "rebuttal_legacy_reproduction/runners/ppl_instrumented.py"
ZS_SCRIPT   = REPO / "rebuttal_legacy_reproduction/runners/zero_shot_instrumented.py"
WIKI_DIR    = "/home/bai-project/jiaqizhang/wikitext2"
C4_TEXTS    = "/diskF/jiaqizhang/fdp_eb_orth_paper_v1_results/offline_calib/c4/c4_ppl_selected_texts.jsonl"
ZS_YAML_DIR = "/diskF/jiaqizhang/a1804298693/offline_datasets"
CACHE_DIR   = REPO / "EigenAttn/HF_cache"

_orig_rope = LlamaRotaryEmbedding.forward
def _compat_rope(self, x, position_ids):
    if hasattr(self, "inv_freq") and torch.is_tensor(self.inv_freq):
        if self.inv_freq.device != position_ids.device:
            self.inv_freq = self.inv_freq.to(position_ids.device)
    return _orig_rope(self, x, position_ids)
LlamaRotaryEmbedding.forward = _compat_rope

UPSTREAM = Path("/diskF/jiaqizhang/fdp_lowrank_operator_zoo_v1/repos/eigenattention_upstream")
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))
from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn

def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def mkenv(gpu_id, extra=""):
    e = os.environ.copy()
    e["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    parts = [str(REPO / "EigenAttn")]
    if extra: parts.append(str(REPO / extra))
    parts.append(e.get("PYTHONPATH", ""))
    e["PYTHONPATH"] = ":".join(parts)
    return e

def project_and_save_model(model_dir, ranks, basis_pt, device="cpu"):
    """Accurately slice basis for ranks and save to model_dir."""
    model_dir.mkdir(parents=True, exist_ok=True)
    basis_dict = torch.load(basis_pt, map_location="cpu")
    config = AutoConfig.from_pretrained(MODEL_BASE)
    config.use_cache = False
    
    dense_model = LlamaForCausalLM.from_pretrained(MODEL_BASE, torch_dtype=torch.bfloat16, device_map=device)
    dense_model.eval()
    
    cmp_model = LlamaForCausalLM_EigenAttn.from_pretrained(
        MODEL_BASE, config=config, low_rank_config=ranks, device_map=device, torch_dtype=torch.bfloat16
    )
    cmp_model.eval()
    
    for li in range(32):
        cmp_attn = cmp_model.model.layers[li].self_attn
        dense_layer = dense_model.model.layers[li].self_attn
        rkq, rv = ranks[li]
        k_col = rkq // 8
        v_col = rv // 8
        
        q_key = f"base_model.model.model.layers.{li}.self_attn.q_proj_basis"
        v_key = f"base_model.model.model.layers.{li}.self_attn.v_proj_basis"
        U_kq = basis_dict[q_key].to(device=device, dtype=torch.bfloat16)
        U_v = basis_dict[v_key].to(device=device, dtype=torch.bfloat16)
        
        W_k = dense_layer.k_proj.weight.data
        W_v = dense_layer.v_proj.weight.data
        W_o = dense_layer.o_proj.weight.data
        
        B_k = U_kq[:k_col, :].contiguous()
        cmp_attn.k_proj.weight.data.copy_(B_k @ W_k)
        
        B_v = U_v[:v_col, :].contiguous()
        cmp_attn.v_proj.weight.data.copy_(B_v @ W_v)
        
        cmp_attn.o_proj.weight.data.copy_(W_o)

    del dense_model
    torch.cuda.empty_cache()
    
    # Save model and config
    cmp_model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
    tokenizer.save_pretrained(model_dir)
    with open(model_dir / "low_rank_config.json", "w") as f:
        json.dump(ranks, f, indent=2)
    torch.save(torch.tensor(ranks, dtype=torch.long), model_dir / "low_rank_config.pt")
    del cmp_model
    torch.cuda.empty_cache()

def evaluate_one_state(sha, info, gpu_id, scratch_base):
    state_dir = PER_WIN_DIR / sha
    state_dir.mkdir(parents=True, exist_ok=True)
    done_file = state_dir / "STATE_EVAL_DONE.json"
    if done_file.exists():
        print(f"[{utc()}] {sha[:12]} already DONE, skipping.")
        return True

    model_dir = scratch_base / f"model_{sha[:12]}"
    print(f"[{utc()}][GPU {gpu_id}] Building model for {info['tier']} {info['config_name']} (TotRank={info['tot_rank']})...")
    
    try:
        if not list(model_dir.glob("model-*.safetensors")):
            project_and_save_model(model_dir, info["ranks"], info["basis_pt"], device=f"cuda:{gpu_id}")
            print(f"[{utc()}][GPU {gpu_id}] Model projection complete on disk.")

        # 1. WikiText-2 PPL
        wiki_out = state_dir / "wikitext"
        wiki_out.mkdir(parents=True, exist_ok=True)
        wiki_win = state_dir / "wiki_windows.jsonl"
        if not wiki_win.exists():
            wiki_cmd = [
                sys.executable, "-u", str(PPL_SCRIPT),
                "--dataset", "wikitext2",
                "--model", str(model_dir),
                "--tok", MODEL_BASE,
                "--out_dir", str(wiki_out),
                "--seqlen", "2048",
                "--dtype", "bf16",
                "--device", "cuda:0",
                "--method", info["config_name"],
                "--net", "llama",
                "--seed", "2",
                "--wikitext2_dir", WIKI_DIR,
                "--evaluation_protocol_hash", "da93ade0391d1886941e896092c016d08918defcb8c68eab7e315758209f0644",
            ]
            print(f"[{utc()}][GPU {gpu_id}] Running WikiText-2 PPL...")
            p = subprocess.run(wiki_cmd, cwd=str(REPO), env=mkenv(gpu_id, "rebuttal_legacy_reproduction"),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                print(f"[{utc()}][GPU {gpu_id}] WIKI FAILED:\n{p.stdout.decode('utf-8', errors='ignore')[-2000:]}")
                return False
            out_f = wiki_out / "ppl_windows.jsonl"
            if out_f.exists(): out_f.replace(wiki_win)
            else: return False

        # 2. C4 PPL
        c4_out = state_dir / "c4"
        c4_out.mkdir(parents=True, exist_ok=True)
        c4_win = state_dir / "c4_windows.jsonl"
        if not c4_win.exists():
            c4_cmd = [
                sys.executable, "-u", str(PPL_SCRIPT),
                "--dataset", "c4",
                "--model", str(model_dir),
                "--tok", MODEL_BASE,
                "--c4_texts", C4_TEXTS,
                "--out_dir", str(c4_out),
                "--seqlen", "2048",
                "--dtype", "bf16",
                "--device", "cuda:0",
                "--method", info["config_name"],
                "--net", "llama",
                "--seed", "2",
                "--evaluation_protocol_hash", "da93ade0391d1886941e896092c016d08918defcb8c68eab7e315758209f0644",
            ]
            print(f"[{utc()}][GPU {gpu_id}] Running C4 PPL...")
            p = subprocess.run(c4_cmd, cwd=str(REPO), env=mkenv(gpu_id, "rebuttal_legacy_reproduction"),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                print(f"[{utc()}][GPU {gpu_id}] C4 FAILED:\n{p.stdout.decode('utf-8', errors='ignore')[-2000:]}")
                return False
            out_f = c4_out / "ppl_windows.jsonl"
            if out_f.exists(): out_f.replace(c4_win)
            else: return False

        # 3. 5-Task Zero-Shot
        zs_out = state_dir / "zeroshot"
        zs_out.mkdir(parents=True, exist_ok=True)
        zs_agg = zs_out / "zero_shot_aggregate.json"
        if not zs_agg.exists():
            zs_cmd = [
                sys.executable, "-u", str(ZS_SCRIPT),
                "--model", str(model_dir),
                "--tokenizer_model", MODEL_BASE,
                "--task_yaml_dir", ZS_YAML_DIR,
                "--tasks", "hellaswag_localjsonl,arc_challenge_localjsonl,arc_easy_localjsonl,winogrande_localjsonl,piqa_localjsonl",
                "--out_dir", str(zs_out),
                "--net", "llama-3-8b",
                "--dtype", "bf16",
                "--method", info["config_name"],
                "--cache_dir", str(CACHE_DIR),
                "--attn_implementation", "eager",
            ]
            print(f"[{utc()}][GPU {gpu_id}] Running 5-Task Zero-Shot...")
            p = subprocess.run(zs_cmd, cwd=str(REPO), env=mkenv(gpu_id, "rebuttal_legacy_reproduction"),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                print(f"[{utc()}][GPU {gpu_id}] ZS FAILED:\n{p.stdout.decode('utf-8', errors='ignore')[-2000:]}")
                return False

        # 4. Final Aggregation
        w_lines = [json.loads(l) for l in wiki_win.read_text().splitlines() if l.strip()]
        w_nll = sum(float(l["total_nll"]) for l in w_lines) / sum(int(l["token_count"]) for l in w_lines)
        w_ppl = math.exp(w_nll)

        c_lines = [json.loads(l) for l in c4_win.read_text().splitlines() if l.strip()]
        c_nll = sum(float(l["total_nll"]) for l in c_lines) / sum(int(l["token_count"]) for l in c_lines)
        c_ppl = math.exp(c_nll)

        z_dict = json.loads(zs_agg.read_text())
        tasks_res = {tn: td.get("lm_eval_primary", td.get("accuracy_from_correct"))
                     for tn, td in z_dict.get("tasks", {}).items()}
        z_macro = sum(tasks_res.values()) / len(tasks_res) if tasks_res else None

        rec = {
            "sha": sha,
            "status": "DONE_VALID_EVALUATED",
            "tier": info["tier"],
            "group": info["group"],
            "config_name": info["config_name"],
            "tot_rank": info["tot_rank"],
            "b_live": info["b_live"],
            "gap_pct": info["gap_pct"],
            "wiki_ppl": w_ppl,
            "wiki_nll": w_nll,
            "wiki_windows": len(w_lines),
            "c4_ppl": c_ppl,
            "c4_nll": c_nll,
            "c4_windows": len(c_lines),
            "zero_shot_tasks": tasks_res,
            "zero_shot_macro": z_macro,
            "scorer_authority": "lm_eval_primary",
            "completed_at": utc(),
        }
        with open(done_file, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[{utc()}][GPU {gpu_id}] SUCCESS: {info['tier']} {info['config_name']} => Wiki={w_ppl:.4f} C4={c_ppl:.4f} ZS={z_macro:.4f}")
        return True

    finally:
        shutil.rmtree(model_dir, ignore_errors=True)
        print(f"[{utc()}][GPU {gpu_id}] Cleaned scratch {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sha", type=str, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--scratch", type=str, default="/diskE/jiaqizhang/core_models_scratch")
    args = parser.parse_args()

    with open(FORMAL_DIR / "LOO_CONTROLS_AUTHORITY.json") as f:
        records = json.load(f)

    target_info = next((r for r in records if r["sha"] == args.sha), None)
    if not target_info:
        print(f"Error: SHA {args.sha} not found in LOO_CONTROLS_AUTHORITY.json")
        sys.exit(1)

    ok = evaluate_one_state(args.sha, target_info, args.gpu, Path(args.scratch))
    sys.exit(0 if ok else 1)
