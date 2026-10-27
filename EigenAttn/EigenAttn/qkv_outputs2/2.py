import torch

# ============================================================
# 加载最终保存的 Q/K/V 张量
# ============================================================
qkv = torch.load("final_qkv1.pt", map_location="cpu")

eigen_basis = {}
r = 468  # 目标子空间维度（你可以改 48、156、468、700）
rank_stats = []  # 保存每层的统计信息

# ✅ 强制使用 GPU 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

print("🚀 开始在 GPU0 上执行分解...\n")

for name, tensors in qkv.items():
    # ---------- 跳过空或非法张量 ----------
    if tensors is None or (isinstance(tensors, torch.Tensor) and tensors.numel() == 0):
        print(f"⚠️ 跳过空张量: {name}")
        continue

    t = tensors[0] if isinstance(tensors, (list, tuple)) else tensors

    if t.ndim == 3:
        mat = t.squeeze(0)  # [1, N, 768] → [N, 768]
    elif t.ndim == 2:
        mat = t
    else:
        print(f"❌ 跳过 {name}，维度不匹配: {t.shape}")
        continue

    if mat.shape[1] != 768:
        print(f"❌ 跳过 {name}，不是有效的特征形状: {mat.shape}")
        continue

    try:
        # ---------- 使用 GPU0 执行 SVD ----------
        mat_T = mat.T.to(device)  # 转置到 [768, N]
        U, S, Vh = torch.linalg.svd(mat_T, full_matrices=False)

        # ---------- 自动调整 rank ----------
        actual_r = min(r, U.shape[1])
        basis = U[:, :actual_r].contiguous().to(torch.float32).cpu()

        # ---------- 保存结果 ----------
        if name.endswith((".q_proj", ".k_proj", ".v_proj")):
            new_key = name + "_basis"
            eigen_basis[new_key] = basis
            print(f"✔ 保存成功: {new_key} → {basis.shape}")
        else:
            print(f"⚠️ 跳过无关权重: {name}")

        # ---------- 统计信息 ----------
        rank_stats.append({
            "layer": name,
            "input_shape": list(mat_T.shape),
            "available_rank": U.shape[1],
            "used_rank": actual_r,
            "compress_ratio": round(actual_r / 768, 3)
        })

        # ✅ 主动释放 GPU 内存，防止 OOM
        del mat_T, U, S, Vh
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ 处理 {name} 时出错: {e}")
        torch.cuda.empty_cache()

# ============================================================
# 保存结果
# ============================================================
torch.save(eigen_basis, "eigen_basis1.pt")
print("\n✅ 所有 *_proj_basis 成功保存至 eigen_basis.pt")

# ============================================================
# 打印统计信息
# ============================================================
print("\n📊 Rank 统计结果（每层实际可分解 rank）:")
print("-" * 80)
print(f"{'Layer':45} {'AvailRank':>10} {'UsedRank':>10} {'Ratio':>10}")
print("-" * 80)
for stat in rank_stats:
    print(f"{stat['layer'][:45]:45} {stat['available_rank']:>10} {stat['used_rank']:>10} {stat['compress_ratio']:>10.3f}")
print("-" * 80)
print(f"📦 共处理 {len(rank_stats)} 层，目标维度 r={r}，结果已统计完成。")

