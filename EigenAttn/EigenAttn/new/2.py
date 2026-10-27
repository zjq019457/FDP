import torch

# 加载最终保存的 Q/K/V 张量
qkv = torch.load("final_qkv.pt")

# 保存所有基向量
eigen_basis = {}
r = 648  # 子空间维度（建议与 hidden_dim 相关）

for name, t in qkv.items():
    print(f"[DEBUG] {name} shape: {t.shape}")

    # 维度检查与处理
    if t.ndim == 3:
        mat = t.squeeze(0)  # [1, N, 768] → [N, 768]
    elif t.ndim == 2:
        mat = t
    else:
        print(f"❌ 跳过 {name}，维度不匹配: {t.shape}")
        continue

    if mat.shape[1] != 768:
        print(f"⚠️ 跳过 {name}，不是有效的特征形状: {mat.shape}")
        continue

    try:
        # 进行 SVD 分解
        mat_T = mat.T  # 转置为 [768, N]
        U, _, _ = torch.linalg.svd(mat_T, full_matrices=False)

        # 提取前 r 个特征向量 [768, r]
        basis = U[:, :r].contiguous().to(torch.float32)

        # ✅ 匹配 "_q_proj"、"_k_proj"、"_v_proj"
        if any(x in name for x in ["_q_proj", "_k_proj", "_v_proj"]):
            new_key = name + "_basis"
            eigen_basis[new_key] = basis.cpu()
            print(f"✔ 保存成功: {new_key} → {basis.shape}")
        else:
            print(f"⚠️ 跳过无关权重: {name}")

    except Exception as e:
        print(f"❌ 处理 {name} 时出错: {e}")

# 保存所有的基向量
torch.save(eigen_basis, "eigen_basis.pt")
print("✅ 所有 *_proj_basis 成功保存，维度应为 [768, 648]")

