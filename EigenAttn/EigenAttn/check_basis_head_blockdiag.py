import torch

BASIS_PT="/diskE/jiaqizhang/eigen_basis_activation_keepdim.pt"
L=0  # 先看 layer0
NH=56
HD=128

B=torch.load(BASIS_PT, map_location="cpu")

def ratio(kind):
    key=f"base_model.model.model.decoder.layers.{L}.self_attn.{kind}_proj_basis"
    U=B[key].float()
    h, r = U.shape
    assert h == NH * HD, f"unexpected hidden dim: {h}"
    assert r % NH == 0, f"rank {r} is not divisible by heads {NH}"
    HDL = r // NH

    # 视为 (head_out, hd_out, head_low, hd_low)
    U4 = U.view(NH, HD, NH, HDL)
    # 每个 (i,j) block 的能量
    E = (U4**2).sum(dim=(1,3))   # [NH,NH]
    diag = E.diag().sum().item()
    total = E.sum().item()
    off = total - diag
    return off/total, diag/total

for kind in ["q","k","v"]:
    off, diag = ratio(kind)
    print(f"{kind}: offdiag_ratio={off:.6f}  diag_ratio={diag:.6f}")
