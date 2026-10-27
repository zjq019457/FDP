import os, json, math, torch
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

low=json.load(open("./low_rank_config_layerwise_5712_opt30b.json","r"))

src="/diskD/eigen_outputs_opt30b_r5712/final_model"
dst="/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed"

m=OPTForCausalLM_EigenAttn.from_pretrained(
    src, local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map=None, low_cpu_mem_usage=True,
    low_rank_config=low
).cuda().eval()

# 选择一个“统一初值”
target = float(os.getenv("EIGEN_VSCALE_RESET", "1.0"))
target = max(target, 1e-6)
target_log = math.log(target)

bad=0
total=0
with torch.no_grad():
    for mod in m.modules():
        if hasattr(mod, "vscale_log"):
            total += 1
            v = mod.vscale_log.data
            if (not torch.isfinite(v).all()) or (v.abs().max().item() > 50):
                bad += 1
            mod.vscale_log.data = torch.tensor([target_log], device=v.device, dtype=torch.float32)
print(f"reset vscale_log: total={total}, bad_before={bad}, set to log({target})={target_log:.6g}")

# 保存（30B 会写很多 shard，耐心等它跑完）
m.save_pretrained(dst, safe_serialization=True)
print("saved to:", dst)
