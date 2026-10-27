import numpy as np
import torch

# ================================
# 配置
# ================================
memmap_path = "expanded_qkv_memmap.dat"

shape = (1, 27022755, 768)
dtype = np.float32

# 🔥 安全 chunk 大小
chunk_size = 100_000   # 10万行 → ~300MB RAM 永不 OOM
r = 768  # 要保留的维度


# ================================
# 第一步：加载 memmap（不会占 RAM）
# ================================
print("Loading memmap...")
mm = np.memmap(memmap_path, dtype=dtype, mode="r", shape=shape)

# 协方差矩阵（仅 2MB）
C = np.zeros((768, 768), dtype=np.float64)


# ================================
# 第二步：分块累积 C = XᵀX
# ================================
print("Computing covariance matrix C = X^T X ...")
total_rows = shape[1]

start = 0
while start < total_rows:
    end = min(start + chunk_size, total_rows)

    print(f"Processing rows {start} → {end} ...")
    X = mm[0, start:end, :].astype(np.float64)   # 不会爆内存

    C += X.T @ X
    del X

    start = end


# ================================
# 第三步：对 C 做 SVD
# ================================
print("Performing SVD on small 768×768 matrix...")
U, S, Vt = np.linalg.svd(C)

U_r = torch.tensor(U[:, :r], dtype=torch.float32)

torch.save(U_r, "eigen_basis.pt")
print("Done! eigen_basis.pt saved.")
print("Shape:", U_r.shape)

