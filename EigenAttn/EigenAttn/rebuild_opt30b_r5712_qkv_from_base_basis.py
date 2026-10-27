import os, json, argparse, torch
from transformers import AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

torch.set_grad_enabled(False)

def getU(B, layer, kind):
    key = f"base_model.model.model.decoder.layers.{layer}.self_attn.{kind}_proj_basis"
    U = B[key].float()  # [d, r]
    return U

def copy_linear_weight_bias(dst, W, b=None):
    dst.weight.copy_(W.to(dtype=dst.weight.dtype))
    if getattr(dst, "bias", None) is not None:
        if b is None:
            dst.bias.zero_()
        else:
            dst.bias.copy_(b.to(dtype=dst.bias.dtype))

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cmp_in", required=True)
    ap.add_argument("--cmp_out", required=True)
    ap.add_argument("--low_json", required=True)
    ap.add_argument("--basis_pt", required=True)
    ap.add_argument("--dtype", default="float16", choices=["float16","bfloat16"])
    ap.add_argument("--keqk", type=int, default=0)  # 1: Uk = Uq for all layers
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    low = json.load(open(args.low_json, "r"))
    B = torch.load(args.basis_pt, map_location="cpu")

    print("Load BASE (cpu)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, local_files_only=True, torch_dtype=dtype, device_map=None, low_cpu_mem_usage=True
    ).cpu().eval()

    print("Load CMP_IN (cpu)...")
    cmpm = OPTForCausalLM_EigenAttn.from_pretrained(
        args.cmp_in, local_files_only=True, torch_dtype=dtype, device_map=None,
        low_cpu_mem_usage=True, low_rank_config=low
    ).cpu().eval()

    n_layers = base.config.num_hidden_layers
    d = base.config.hidden_size
    print(f"n_layers={n_layers} hidden={d} keqk={args.keqk}")

    for i in range(n_layers):
        b_attn = base.model.decoder.layers[i].self_attn
        c_attn = cmpm.model.decoder.layers[i].self_attn

        Uq = getU(B, i, "q")  # [d,r]
        Uk = getU(B, i, "k")
        Uv = getU(B, i, "v")
        if args.keqk == 1:
            Uk = Uq

        # ---- (1) UP = basis (必须和 low 一致) ----
        if hasattr(c_attn, "q_proj_up"): copy_linear_weight_bias(c_attn.q_proj_up, Uq, b=None)
        if hasattr(c_attn, "k_proj_up"): copy_linear_weight_bias(c_attn.k_proj_up, Uk, b=None)
        if hasattr(c_attn, "v_proj_up"): copy_linear_weight_bias(c_attn.v_proj_up, Uv, b=None)

        # ---- (2) LOW = U^T @ W_base ----
        for kind, U, proj in [("q",Uq,"q_proj"), ("k",Uk,"k_proj"), ("v",Uv,"v_proj")]:
            W = getattr(b_attn, proj).weight.detach().float()  # [d,d]
            b = getattr(b_attn, proj).bias.detach().float()    # [d]

            Wlow = (U.t() @ W).contiguous()                    # [r,d]

            low_layer = getattr(c_attn, f"{proj}_low")
            low_layer.weight.copy_(Wlow.to(dtype=low_layer.weight.dtype))

            # bias：优先把 base bias 放到 UP（full-dim），LOW bias 清零
            up_layer = getattr(c_attn, f"{proj}_up", None)
            if up_layer is not None and getattr(up_layer, "bias", None) is not None:
                up_layer.bias.copy_(b.to(dtype=up_layer.bias.dtype))
                low_layer.bias.zero_()
            else:
                # 退化方案：只能近似（投影 bias）
                low_layer.bias.copy_((b @ U).to(dtype=low_layer.bias.dtype))

        # ---- (3) out_proj_up = Wo @ Uv ----
        Wo = b_attn.out_proj.weight.detach().float()  # [d,d]
        bo = b_attn.out_proj.bias.detach().float()    # [d]
        Wup = (Wo @ Uv).contiguous()                  # [d,r]

        c_attn.out_proj_up.weight.copy_(Wup.to(dtype=c_attn.out_proj_up.weight.dtype))
        if getattr(c_attn.out_proj_up, "bias", None) is not None:
            c_attn.out_proj_up.bias.copy_(bo.to(dtype=c_attn.out_proj_up.bias.dtype))

        if i % 8 == 0:
            print("rebuilt layer", i)

    os.makedirs(args.cmp_out, exist_ok=True)
    cmpm.save_pretrained(args.cmp_out, safe_serialization=True)
    print("saved:", args.cmp_out)

if __name__ == "__main__":
    main()
