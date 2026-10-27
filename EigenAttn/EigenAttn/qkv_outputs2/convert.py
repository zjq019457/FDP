import torch
import re
import math

# ======================================================
# 配置
# ======================================================
qkv = torch.load("final_qkv1.pt")  # 你保存的结果
HIDDEN_SIZE = 768                  # OPT-125M hidden size = 768
TARGET_RATIO = 0.8125                 # ⭐ 压缩比例 = 0.8 => avg_dim = 0.8 × 768
r = math.ceil(HIDDEN_SIZE * TARGET_RATIO)  # 计算 avg_dim
print(f"✅ 设置 avg_dim = {r}  (ratio={TARGET_RATIO})")

eigen_basis = {}

# ======================================================
# 转换和 SVD
# ======================================================
for name, t in qkv.items():
    print(f"\n[DEBUG] 原始键: {name} shape={t.shape}")

    # 跳过 LoRA 权重矩阵，例如 ...lora_A / lora_B
    if "lora" in name.lower():
        print(f"⚠️ 跳过 LoRA 权重: {name}")
        continue

    # 修复 key (main_eigen_attn.py 需要这种 key 命名方式)
    name_fixed = name
    name_fixed = name_fixed.replace("base_model_model_model_decoder_layers", "base_model.model.model.decoder.layers")
    name_fixed = re.sub(r"layers_(\d+)", r"layers.\1", name_fixed)
    name_fixed = name_fixed.replace("_self_attn_", ".self_attn.")
    name_fixed = name_fixed.replace("_q_proj", ".q_proj")
    name_fixed = name_fixed.replace("_k_proj", ".k_proj")
    name_fixed = name_fixed.replace("_v_proj", ".v_proj")
    name_fixed = re.sub(r"_step\d+", "", name_fixed)

    print(f"➡️ 修正 key: {name_fixed}")

    # ======================================================
    # ✅ reshape activation → (N, 768)
    # ======================================================
    if t.ndim == 4:  # (batch, seq, heads, head_dim)
        batch, seq, heads, head_dim = t.shape
        assert heads * head_dim == HIDDEN_SIZE, f"错误: expected hidden size {HIDDEN_SIZE}, got {heads*head_dim}"
        mat = t.reshape(batch * seq * heads, heads * head_dim)

    elif t.ndim == 3:  # (batch, seq, hidden_size)
        mat = t.reshape(-1, t.shape[-1])

    else:
        print(f"❌ 跳过 {name_fixed}, 无法处理: {t.shape}")
        continue

    print(f"✅ reshape 后: {mat.shape}")

    # ======================================================
    # ⚡ SVD 分解 → 取得 top-r eigen basis
    # ======================================================
    try:
        mat_T = mat.T  # shape => (hidden_size, samples)
        U, _, _ = torch.linalg.svd(mat_T, full_matrices=False)

        basis = U[:, :r].contiguous().float()  # 得到 (hidden_size, r)

        # 仅保存 Q/K/V projection 的 basis
        if any(x in name_fixed for x in [".q_proj", ".k_proj", ".v_proj"]):
            key = f"{name_fixed}_basis"
            eigen_basis[key] = basis.cpu()
            print(f"🎯 保存 basis: {key} → {basis.shape}")
        else:
            print(f"⚠️ 跳过非 QKV 模块: {name_fixed}")

    except Exception as e:
        print(f"❌ SVD 错误 {name_fixed}: {e}")


# ======================================================
# ✅ 保存 eigen basis
# ======================================================
torch.save(eigen_basis, "eigen_basis.pt")
print("\n🚀 完成：eigen_basis.pt 已保存，可用于 main_eigen_attn.py")

