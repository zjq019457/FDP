import os
import torch
import numpy as np

PATH = "./qkv_outputs/eigen_basis.pt"

def summary_tensor(name, t):
    if not torch.is_tensor(t):
        print(f"{name}: not a tensor ({type(t)})")
        return
    t_cpu = t.detach().cpu()
    print(f"{name}: shape={tuple(t_cpu.shape)} dtype={t_cpu.dtype} "
          f"min={t_cpu.min().item():.4e} max={t_cpu.max().item():.4e} "
          f"mean={t_cpu.mean().item():.4e} norm={t_cpu.norm().item():.4e}")

def main():
    print("[Path]", os.path.abspath(PATH))
    obj = torch.load(PATH, map_location="cpu")
    print("[Type]", type(obj))

    # Case 1: tensor
    if torch.is_tensor(obj):
        summary_tensor("eigen_basis", obj)
        return

    # Case 2: dict
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"[Dict] n_keys={len(keys)} sample_keys={keys[:30]}")

        # print each key's structure
        for k in keys[:30]:
            v = obj[k]
            if torch.is_tensor(v):
                summary_tensor(f"[{k}]", v)
            elif isinstance(v, (list, tuple)):
                print(f"[{k}]: {type(v)} len={len(v)} elem0_type={type(v[0]) if len(v)>0 else None}")
                if len(v) > 0 and torch.is_tensor(v[0]):
                    summary_tensor(f"[{k}][0]", v[0])
            else:
                print(f"[{k}]: {type(v)}")

        # Heuristic: layerwise list under common names
        for cand in ["U", "basis", "eigen", "eigen_basis", "layerwise", "layers"]:
            if cand in obj:
                v = obj[cand]
                print(f"\n[Heuristic hit] key='{cand}' type={type(v)}")
                if torch.is_tensor(v):
                    summary_tensor(cand, v)
                elif isinstance(v, (list, tuple)):
                    print(f"{cand}: len={len(v)}")
                    if len(v) > 0 and torch.is_tensor(v[0]):
                        summary_tensor(f"{cand}[0]", v[0])
                break
        return

    # Case 3: list/tuple
    if isinstance(obj, (list, tuple)):
        print(f"[List/Tuple] len={len(obj)} elem0_type={type(obj[0]) if len(obj)>0 else None}")
        if len(obj) > 0 and torch.is_tensor(obj[0]):
            summary_tensor("basis[0]", obj[0])
        return

    print("[Unknown structure]")

if __name__ == "__main__":
    main()
