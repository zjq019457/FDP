import torch
import os
from tqdm import tqdm

folder = "qkv_outputs"
output_path = os.path.join(folder, "final_qkv.pt")

qkv_dict = {}

for filename in tqdm(sorted(os.listdir(folder))):
    if filename.endswith(".pt") and "final_qkv" not in filename:
        tensor = torch.load(os.path.join(folder, filename), map_location="cpu")
        key = filename.replace(".pt", "")
        qkv_dict[key] = tensor

# 一次性保存合并后的字典
torch.save(qkv_dict, output_path)
print(f"✅ 合并完成，共 {len(qkv_dict)} 个张量，保存至 {output_path}")

