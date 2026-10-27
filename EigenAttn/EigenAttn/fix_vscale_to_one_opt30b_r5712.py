import os, json, argparse, torch
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--low_json", required=True, help="low_rank_config_layerwise_5712_opt30b.json")
    ap.add_argument("--dtype", default="float16", choices=["float16","bfloat16"])
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    low = json.load(open(args.low_json, "r"))

    print("Load CMP_IN (cpu)...")
    m = OPTForCausalLM_EigenAttn.from_pretrained(
        args.in_dir,
        local_files_only=True,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        low_rank_config=low,          # ✅ 关键：必须传
    ).cpu().eval()

    target_log = 0.0  # log(1.0)
    total = 0
    changed = 0
    bad_before = 0

    for name, mod in m.named_modules():
        # case 1: vscale_log
        if hasattr(mod, "vscale_log"):
            v = getattr(mod, "vscale_log")
            if torch.is_tensor(v):
                total += 1
                vf = v.detach().float()
                if (not torch.isfinite(vf).all()) or (vf.abs().max().item() > 1e-6):
                    bad_before += 1
                v.fill_(target_log)
                changed += 1
            continue

        # case 2: vscale
        if hasattr(mod, "vscale"):
            v = getattr(mod, "vscale")
            if torch.is_tensor(v):
                total += 1
                vf = v.detach().float()
                if (not torch.isfinite(vf).all()) or ((vf - 1.0).abs().max().item() > 1e-6):
                    bad_before += 1
                v.fill_(1.0)
                changed += 1
            continue

        # case 3: vscale_param
        if hasattr(mod, "vscale_param"):
            v = getattr(mod, "vscale_param")
            if torch.is_tensor(v):
                total += 1
                vf = v.detach().float()
                if (not torch.isfinite(vf).all()) or ((vf - 1.0).abs().max().item() > 1e-6):
                    bad_before += 1
                v.fill_(1.0)
                changed += 1
            continue

    print(f"[vscale] touched_total={total}, bad_before={bad_before}, set_to=1.0 (log=0), changed={changed}")

    os.makedirs(args.out_dir, exist_ok=True)
    m.save_pretrained(args.out_dir, safe_serialization=True)
    print("saved to:", args.out_dir)

if __name__ == "__main__":
    main()
