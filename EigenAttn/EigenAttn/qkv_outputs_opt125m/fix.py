import torch

# 加载原始 eigen_basis.pt 文件
basis_dict = torch.load("final_step.pt")
fixed_dict = {}

for name, u in basis_dict.items():
    print(f"Processing {name}, original shape: {u.shape}")

    # 去掉 batch 维度: [1, 54, 54] -> [54, 54]
    u = u.squeeze(0)

    # 检查 shape 是否为正方的 basis（比如 [54, 54]），直接保存
    if u.shape[0] == u.shape[1]:
        fixed_dict[name] = u
    else:
        print(f"Skipping {name} due to non-square shape: {u.shape}")

torch.save(fixed_dict, "eigen_basis1.pt")
print("✅ Saved eigen_basis_fixed.pt (only square matrices kept)")

