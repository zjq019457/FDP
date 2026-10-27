import torch

# 加载 eigen_basis.pt
eigen_basis = torch.load("qkv_outputs/eigen_basis.pt")

# 遍历 basis_v 并检查每一层的维度
for layer_idx, basis in enumerate(eigen_basis["v"]):
    print(f"[Check] layer {layer_idx}, basis_v shape: {basis.shape}")
    if basis.shape[0] != 4096:
        print(f"❌ [ERROR] layer {layer_idx} basis_v has wrong first dimension: {basis.shape[0]} (expected 4096)")
    else:
        print(f"✅ [OK] layer {layer_idx} basis_v has correct shape")

