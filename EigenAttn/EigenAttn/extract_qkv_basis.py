import torch
import os

# 设置输入输出路径
qkv_path = "qkv_outputs/final_step.pt"
output_path = "qkv_outputs/eigen_basis.pt"
rank = 768  # 可调整保留的特征子空间维度

# 加载 Q/K/V 张量
print(f"加载 QKV 文件: {qkv_path}")
qkv = torch.load(qkv_path)

# 存储特征基底
basis_dict = {}

for name, tensors in qkv.items():
    tensor = tensors[0]  # 默认取第一个 batch 的张量
    print(f"处理层: {name}，shape: {tensor.shape}")

    # 做 SVD 分解
    try:
        u, s, v = torch.linalg.svd(tensor, full_matrices=False)
    except RuntimeError:
        print(f"⚠️ 警告：SVD 失败，跳过层 {name}")
        continue

    u_trunc = u[:, :rank]  # 取前 r 个特征向量
    basis_dict[name + "_basis"] = u_trunc.cpu()

# 保存所有基底
torch.save(basis_dict, output_path)
print(f"✅ 已保存 Eigen 基底到: {output_path}")

