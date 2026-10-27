import torch
import os
import numpy as np
import gc
from tqdm import tqdm

BLOCK_FOLDER = "merged_blocks"
SAVE_PATH = "/media/bai3090/Seagate Portable Drive/expanded_qkv_memmap.dat"
SHAPE_PATH = "/media/bai3090/Seagate Portable Drive/expanded_qkv_shape.txt"

# === 第一步：统计所有 block 的总长度 ===

block_files = sorted([
    os.path.join(BLOCK_FOLDER, f)
    for f in os.listdir(BLOCK_FOLDER)
    if f.startswith("block_") and f.endswith(".pt")
])

print(f"发现 {len(block_files)} 个 block")

total_len = 0
hidden_dim = None

print("\n=== 统计所有 block 的总长度 ===")
for f in tqdm(block_files):
    t = torch.load(f, map_location="cpu")
    if hidden_dim is None:
        hidden_dim = t.shape[2]
    total_len += t.shape[1]
    del t
    gc.collect()

print(f"\n⚡ 总长度 total_len = {total_len}")
print(f"⚡ hidden_dim = {hidden_dim}")

# === 第二步：创建 memmap 文件（不会占内存） ===

print("\n=== 创建 memmap 文件 ===")
mm = np.memmap(
    SAVE_PATH,
    dtype="float32",
    mode="w+",
    shape=(1, total_len, hidden_dim)
)

# === 第三步：逐个 block 写入 memmap ===

print("\n=== 开始写入数据到 memmap（不会 OOM）===\n")
offset = 0

for f in tqdm(block_files):
    t = torch.load(f, map_location="cpu").float().numpy()

    length = t.shape[1]
    mm[0, offset:offset+length, :] = t  # 直接写入硬盘
    offset += length

    del t
    gc.collect()

# 刷盘
mm.flush()

print("\n🎉 memmap 合并完成！")
print(f"📁 数据保存在: {SAVE_PATH}")

# 保存最终形状
with open(SHAPE_PATH, "w") as f:
    f.write(f"{1} {total_len} {hidden_dim}")

print(f"📄 张量形状保存在: {SHAPE_PATH}")

