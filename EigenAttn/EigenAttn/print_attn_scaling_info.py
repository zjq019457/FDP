import json, torch
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

CMP="/diskD/eigen_outputs_opt30b_r5712/final_model_sanitized_KeqQ"
LOW_JSON="./low_rank_config_layerwise_5712_opt30b.json"
low=json.load(open(LOW_JSON,"r"))

m=OPTForCausalLM_EigenAttn.from_pretrained(
    CMP, local_files_only=True, torch_dtype=torch.float16,
    device_map=None, low_rank_config=low, low_cpu_mem_usage=True
).cpu().eval()

attn=m.model.decoder.layers[0].self_attn
print("rank_q (from q_proj_low out):", attn.q_proj_low.out_features)
print("rank_k:", attn.k_proj_low.out_features)
print("rank_v:", attn.v_proj_low.out_features)
print("num_heads:", attn.num_heads)
print("head_dim_low:", attn.q_proj_low.out_features // attn.num_heads)

for name in ["head_dim","scaling","scale","qk_scale"]:
    if hasattr(attn, name):
        print(name, "=", getattr(attn, name))
