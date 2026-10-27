# build_eigen_basis_opt.py
import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def svd_basis_from_weight(W: torch.Tensor, rank: int) -> torch.Tensor:
    """
    W: [out_dim, in_dim] (Linear.weight)
    return: U[:, :rank] where W^T = (U S V^T) => basis over input_dim is U of W^T
    """
    X = W.T.float()  # [in_dim, out_dim]
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    r = min(rank, U.shape[1])
    return U[:, :r].contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Local model dir or HF id, e.g. /diskE/opt-30b")
    ap.add_argument(
        "--out",
        type=str,
        default="/diskD/jiaqizhang/eigen_basis_7168_full.pt",
        help="输出路径（默认 D 盘 jiaqizhang，避免写满系统盘）",
    )
    ap.add_argument("--rank", type=int, default=460, help="basis rank r")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="num_gpus=1 时使用的单卡，如 cuda:0",
    )
    ap.add_argument(
        "--num_gpus",
        type=int,
        default=4,
        help=">1 时用 device_map=auto 把权重切到多卡（需设置 CUDA_VISIBLE_DEVICES 或保证可见 GPU 数量）",
    )
    ap.add_argument(
        "--per_gpu_memory",
        type=str,
        default="",
        help='多卡时传给 max_memory，如 "22GiB"；留空则由 accelerate 自动分配',
    )
    args = ap.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dt = dtype_map[args.dtype]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        usage = shutil.disk_usage(str(out_path.parent))
        free_gb = usage.free / (1024**3)
        print(f"[Info] 输出目录剩余空间约 {free_gb:.1f} GiB -> {out_path.parent}")
        if free_gb < 18:
            print("[warn] 全秩 7168 的 pt 约需 14GiB+，空间偏紧可能导致保存失败。")
    except OSError:
        pass

    load_kw = dict(
        torch_dtype=dt,
        low_cpu_mem_usage=True,
    )
    if int(args.num_gpus) <= 1:
        load_kw["device_map"] = {"": args.device}
        print(f"[Load] model={args.model} device_map=single ({args.device})")
    else:
        load_kw["device_map"] = "auto"
        if args.per_gpu_memory.strip():
            n = int(args.num_gpus)
            per = args.per_gpu_memory.strip()
            load_kw["max_memory"] = {i: per for i in range(n)}
            load_kw["max_memory"]["cpu"] = "256GiB"
        print(f"[Load] model={args.model} device_map=auto num_gpus={args.num_gpus}")

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    model.eval()

    layers = model.model.decoder.layers
    n_layers = len(layers)
    hidden = model.config.hidden_size
    print(f"[Info] layers={n_layers} hidden={hidden} rank={args.rank}")

    eigen_basis = {}
    for i, layer in enumerate(layers):
        attn = layer.self_attn
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            lin = getattr(attn, proj_name)
            W = lin.weight.detach().to("cpu")
            B = svd_basis_from_weight(W, args.rank)
            key = (
                f"base_model.model.model.decoder.layers.{i}.self_attn."
                f"{proj_name.replace('_proj', '')}_proj_basis"
            )
            eigen_basis[key] = B.half()
        if (i + 1) % 4 == 0:
            print(f"[Done] layer {i+1}/{n_layers}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    torch.save(eigen_basis, str(out_path))
    print(f"[Save] {out_path} keys={len(eigen_basis)}")


if __name__ == "__main__":
    main()
