import torch
from transformers import AutoModelForCausalLM

model_path = "/diskE/opt-30b"   # 例如 /diskE/opt-30b 或你的 final_model
layer_id = 0                        # 改成你要的层号
out_path = "qkv_mats.pt"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,      # 也可 float32
    device_map="cpu"                # 建议先 cpu 导出，稳
)

layer = model.model.decoder.layers[layer_id]  # OPT 系列是这个路径
Wq = layer.self_attn.q_proj.weight.detach().cpu()
Wk = layer.self_attn.k_proj.weight.detach().cpu()
Wv = layer.self_attn.v_proj.weight.detach().cpu()

torch.save({"Wq": Wq, "Wk": Wk, "Wv": Wv}, out_path)
print("saved:", out_path, Wq.shape, Wk.shape, Wv.shape)