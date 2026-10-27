import os, json, torch
from transformers import AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

BASE="/diskE/opt-30b"
CMP="/diskD/eigen_outputs_opt30b_r7168/final_model"
LOW_JSON="./low_rank_config_layerwise_7168_opt30b.json"
OUT="/diskD/eigen_outputs_opt30b_r7168/final_model_fullrank_fixed"

os.makedirs(OUT, exist_ok=True)

# CPU加载，避免显存
base = AutoModelForCausalLM.from_pretrained(
    BASE, local_files_only=True, torch_dtype=torch.float16, device_map={"":"cpu"}
)
cfg = json.load(open(LOW_JSON))
cmp = OPTForCausalLM_EigenAttn.from_pretrained(
    CMP, local_files_only=True, torch_dtype=torch.float16, device_map={"":"cpu"}, low_rank_config=cfg
)

# copy layerwise Q/K/V + out_proj
n_layers = len(base.model.decoder.layers)
assert n_layers == len(cmp.model.decoder.layers)

def copy_linear(dst, src, name):
    assert dst.weight.shape == src.weight.shape, (name, dst.weight.shape, src.weight.shape)
    dst.weight.data.copy_(src.weight.data)
    if getattr(dst, "bias", None) is not None and getattr(src, "bias", None) is not None:
        dst.bias.data.copy_(src.bias.data)

for i in range(n_layers):
    b_attn = base.model.decoder.layers[i].self_attn
    c_attn = cmp.model.decoder.layers[i].self_attn

    # Q/K/V: baseline q_proj/k_proj/v_proj  -> cmp q_proj_low/k_proj_low/v_proj_low
    copy_linear(c_attn.q_proj_low, b_attn.q_proj, f"layer{i}.q")
    copy_linear(c_attn.k_proj_low, b_attn.k_proj, f"layer{i}.k")
    copy_linear(c_attn.v_proj_low, b_attn.v_proj, f"layer{i}.v")

    # out: baseline out_proj -> cmp out_proj_up
    # (如果你的结构里 out_proj_up 是恢复矩阵，这里 full-rank 直接等于 out_proj)
    copy_linear(c_attn.out_proj_up, b_attn.out_proj, f"layer{i}.out")

print("[OK] copied Q/K/V/out for all layers")

# 保存
cmp.save_pretrained(OUT, safe_serialization=True)
# tokenizer/config 通常从 BASE 拷贝更稳（也可从 CMP）
base.config.save_pretrained(OUT)

print("Saved to:", OUT)
