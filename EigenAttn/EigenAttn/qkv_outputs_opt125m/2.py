import torch

# 加载最终保存的 Q/K/V 张量 (shape: [1, num_samples, 768])
qkv = torch.load("final_step.pt")

# 将保存所有的基向量
eigen_basis = {}
r = 60  # 子空间维度，需满足 rank_kq ≥ num_heads 才能正常 reshape

# 遍历 qkv 中所有键值对
for name, tensors in qkv.items():
    if not tensors:
        print(f"⚠️ 跳过空张量: {name}")
        continue

    t = tensors[0]  # 取第一个 tensor
    if t.ndim == 3:
        mat = t.squeeze(0)  # 从 [1, N, 768] → [N, 768]
    elif t.ndim == 2:
        mat = t
    else:
        print(f"❌ 跳过 {name}，维度不匹配: {t.shape}")
        continue

    if mat.shape[1] != 768:
        print(f"❌ 跳过 {name}，不是有效的特征形状: {mat.shape}")
        continue

    try:
        # 进行 SVD 分解
        mat_T = mat.T  # 转置为 [768, N]
        U, _, _ = torch.linalg.svd(mat_T, full_matrices=False)

        # 提取前 r 个特征向量 [768, r]
        #basis = U[:, :r].contiguous().to(torch.float32)
        basis = U[:, :48].contiguous().to(torch.float32)

        if name.endswith((".q_proj", ".k_proj", ".v_proj")):
            new_key = name + "_basis"
            eigen_basis[new_key] = basis.cpu()  # 保存到 CPU
            print(f"✔ 保存成功: {new_key} → {basis.shape}")
        else:
            print(f"⚠️ 跳过无关权重: {name}")
    except Exception as e:
        print(f"❌ 处理 {name} 时出错: {e}")

# 保存所有的基向量到文件
torch.save(eigen_basis, "eigen_basis.pt")
print("✅ 所有 *_proj_basis 成功保存，维度应为 [768, 8]")

