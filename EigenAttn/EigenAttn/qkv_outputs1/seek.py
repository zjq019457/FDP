import torch
import re

# 加载最终的 Q/K/V 张量
qkv = torch.load("final_qkv1.pt")

eigen_basis = {}
r = 768  # 子空间维度（可与 avg_dim 对齐）

for name, t in qkv.items():
    print(f"[DEBUG] 原始键: {name}  shape: {t.shape}")

    # 1️⃣ 修复 key 路径
    name_fixed = name
    name_fixed = name_fixed.replace("base_model_model_model_decoder_layers", "base_model.model.model.decoder.layers")
    name_fixed = re.sub(r"layers_(\d+)", r"layers.\1", name_fixed)
    name_fixed = name_fixed.replace("_self_attn_", ".self_attn.")
    name_fixed = name_fixed.replace("_q_proj", ".q_proj")
    name_fixed = name_fixed.replace("_k_proj", ".k_proj")
    name_fixed = name_fixed.replace("_v_proj", ".v_proj")
    name_fixed = re.sub(r"_step\d+", "", name_fixed)
    print(f"➡️ 修正后键: {name_fixed}")

    # 2️⃣ 维度处理
    if t.ndim == 3:
        mat = t.squeeze(0)
    elif t.ndim == 2:
        mat = t
    else:
        print(f"❌ 跳过 {name_fixed}，维度不匹配: {t.shape}")
        continue

    if mat.shape[1] != 768:
        print(f"⚠️ 跳过 {name_fixed}，不是有效的特征形状: {mat.shape}")
        continue

    # 3️⃣ SVD 分解
    try:
        mat_T = mat.T
        U, _, _ = torch.linalg.svd(mat_T, full_matrices=False)
        basis = U[:, :r].contiguous().to(torch.float32)

        if any(x in name_fixed for x in [".q_proj", ".k_proj", ".v_proj"]):
            new_key = f"{name_fixed}_basis"
            eigen_basis[new_key] = basis.cpu()  # ✅ 只保存 tensor
            print(f"✔ 保存成功: {new_key} → {basis.shape}")
        else:
            print(f"⚠️ 跳过无关模块: {name_fixed}")

    except Exception as e:
        print(f"❌ 处理 {name_fixed} 时出错: {e}")

# 4️⃣ 保存 eigen_basis.pt
torch.save(eigen_basis, "eigen_basis.pt")
print("\n✅ 修复后的 eigen_basis.pt 保存完毕，可直接用于 main_eigen_attn.py")

