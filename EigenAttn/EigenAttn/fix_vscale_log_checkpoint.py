import os, json, math, torch
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

CMP="/diskD/eigen_outputs_opt30b_r7168/final_model_fullrank_fixed"
LOW_JSON="./low_rank_config_layerwise_7168_opt30b.json"
OUT="/diskD/eigen_outputs_opt30b_r7168/final_model_fullrank_fixed_vscale1"

os.makedirs(OUT, exist_ok=True)

cfg = json.load(open(LOW_JSON))

m = OPTForCausalLM_EigenAttn.from_pretrained(
    CMP, local_files_only=True, torch_dtype=torch.float16, device_map={"":"cpu"},
    low_rank_config=cfg
)

target_log = math.log(1.0)   # <- 改成 math.log(0.03) 也行

cnt = 0
for i, layer in enumerate(m.model.decoder.layers):
    attn = layer.self_attn
    if hasattr(attn, "vscale_log"):
        with torch.no_grad():
            attn.vscale_log.data.fill_(target_log)
        cnt += 1

print("reset vscale_log layers:", cnt, "to", target_log)

m.save_pretrained(OUT, safe_serialization=True)
m.config.save_pretrained(OUT)
print("saved:", OUT)
