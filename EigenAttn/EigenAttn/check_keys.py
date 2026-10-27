from safetensors.torch import safe_open

path = "/home/bai-project/jiaqizhang/EigenAttn/EigenAttn/eigen_outputs_1.0x/final_model/model.safetensors"

must_have = [
    "model.decoder.layers.0.self_attn.q_proj.weight",
    "model.decoder.layers.0.self_attn.k_proj.weight",
    "model.decoder.layers.0.self_attn.v_proj.weight",
    "model.decoder.layers.0.self_attn.out_proj.weight",
]

with safe_open(path, framework="pt", device="cpu") as f:
    keys = set(f.keys())

print("Total keys:", len(keys))
for k in must_have:
    print(k, "=>", "OK" if k in keys else "MISSING")

# 额外：粗略看一下包含 self_attn 的 key 有多少
attn_keys = [k for k in keys if "self_attn" in k]
print("self_attn keys:", len(attn_keys))
print("sample attn keys:", attn_keys[:20])
