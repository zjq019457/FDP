import torch

# 这是你的全局 basis（768×768）
U = torch.load("eigen_basis.pt", map_location="cpu")
print("Loaded global basis:", U.shape)

eigen_dict = {}

for layer in range(12):
    layer_prefix = f"base_model.model.model.decoder.layers.{layer}.self_attn"
    for proj in ["q_proj", "k_proj", "v_proj"]:
        key = f"{layer_prefix}.{proj}_basis"
        eigen_dict[key] = U.clone()
        print("✔ Added:", key)

torch.save(eigen_dict, "eigen_basis_fixed.pt")
print("\n🎉 Saved eigen_basis_fixed.pt (dict format, 36 keys)")

