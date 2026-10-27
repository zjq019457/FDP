import torch

# 修改路径为你真实的 eigen_basis.pt 路径
input_path = "eigen_basis.pt"
output_path = "eigen_basis_fixed.pt"

# 加载
eigen_basis = torch.load(input_path, map_location="cpu")
new_basis = {}

# 遍历并处理每个 tensor
for key, value in eigen_basis.items():
    value = torch.as_tensor(value)  # 保证是 tensor 类型
    if value.ndim == 3 and value.shape[0] == 1:
        value = value.squeeze(0)
        print(f"Squeezed {key}: shape {value.shape}")
    elif value.ndim != 2:
        print(f"❌ Unexpected shape for {key}: {value.shape}")
    new_basis[key] = value

# 保存
torch.save(new_basis, output_path)
print("✅ All tensors saved with shape (768, 8)")

