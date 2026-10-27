import os, json, argparse, torch
import inspect as _inspect
from typing import Dict, List, Optional

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME
from transformers.utils.generic import ContextManagers
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

try:
    from safetensors.torch import load_file as _safe_load_file
except Exception:  # pragma: no cover
    _safe_load_file = None

torch.set_grad_enabled(False)


def _normalize_low_rank_config(low, n_layers: int):
    """每层 [rank_kq, rank_v] 为整层总秩（num_heads * r_per_head）。"""
    if not isinstance(low, list) or len(low) < n_layers:
        raise ValueError(
            f"low_rank_config 需为长度 >= {n_layers} 的 list，实际 {type(low)} len={len(low)}"
        )
    out = []
    for i in range(n_layers):
        row = low[i]
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            raise ValueError(f"low_rank_config[{i}] 无效: {row!r}")
        out.append([int(row[0]), int(row[1])])
    return out


def getU(B, layer, kind):
    key=f"base_model.model.model.decoder.layers.{layer}.self_attn.{kind}_proj_basis"
    return B[key].float()  # [d,r]


def _load_basis_pt(path: str) -> Dict[str, torch.Tensor]:
    p = path.strip()
    if not p:
        return {}
    if "weights_only" in _inspect.signature(torch.load).parameters:
        return torch.load(p, map_location="cpu", weights_only=True)
    return torch.load(p, map_location="cpu")


def merge_tail_into_basis(
    U_main: torch.Tensor,
    U_tail: Optional[torch.Tensor],
    tail_cols: int,
    alpha: float,
) -> torch.Tensor:
    """
    在**固定列数 r** 下把截断尾方向揉进列正交基（用于 W_low = U^T W、Wup = Wo @ U）。

    原理（与「P = U_p U_p^T + α U_t U_t^T」同 spirit，但实现为固定秩子空间）：
    - 全秩 PCA 中，主基 U_main 对应前 r_main 个特征方向，U_tail 对应被截掉的后续方向；二者列正交。
    - 若简单把 [U_main | α U_tail] 做 QR 再取前 r 列，则前 r 列仍张成 span(U_main)，尾部**不会**进入（主基已占满前 r 个 Gram–Schmidt 步）。
    - 因此这里用「尾部 t 列 + 主基前 (r−t) 列」拼成 r 列再 QR，等价于在 r 维基里用**尾部方向替换**被截断时丢掉的那部分主方向（确定性、无随机尾）。

    alpha 缩小尾部范数，避免一步加太猛；可只对 V 分支试标定（--tail_branches v）。
    """
    r = U_main.shape[1]
    if U_tail is None or alpha == 0.0 or tail_cols <= 0:
        return U_main
    t = min(int(tail_cols), int(U_tail.shape[1]))
    if t == 0:
        return U_main
    tail_part = (float(alpha) * U_tail[:, :t]).float()
    if t >= r:
        Q, _ = torch.linalg.qr(tail_part, mode="reduced")
        return Q[:, :r].to(dtype=U_main.dtype)
    keep = r - t
    main_part = U_main[:, :keep].float()
    M = torch.cat([tail_part, main_part], dim=1)
    Q, _ = torch.linalg.qr(M, mode="reduced")
    return Q[:, :r].to(dtype=U_main.dtype)


def _load_shard_state(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        if _safe_load_file is None:
            raise ImportError("需要 safetensors 以加载 .safetensors 分片")
        return dict(_safe_load_file(path, device="cpu"))
    import inspect as _ins

    if "weights_only" in _ins.signature(torch.load).parameters:
        return torch.load(path, map_location="cpu", weights_only=True)
    return torch.load(path, map_location="cpu")


def _index_json_paths(folder: str) -> List[str]:
    out: List[str] = []
    for name in (SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            out.append(p)
    if not out:
        for f in sorted(os.listdir(folder)):
            if f.endswith(".index.json"):
                out.append(os.path.join(folder, f))
    return out


def load_checkpoint_shape_filtered(model: torch.nn.Module, folder: str, dtype: torch.dtype) -> None:
    """
    分片 checkpoint：只加载与当前 model 形状一致的键。
    自适应 low_rank 与旧 checkpoint（每层固定 5712）在 self_attn 上不一致的键跳过；
    这些权重随后由脚本用 base×basis 重写。
    """
    folder = os.path.abspath(folder)
    single_safe = os.path.join(folder, "model.safetensors")
    single_bin = os.path.join(folder, "pytorch_model.bin")
    if os.path.isfile(single_safe):
        shard_files = [single_safe]
    elif os.path.isfile(single_bin):
        shard_files = [single_bin]
    else:
        idx_paths = _index_json_paths(folder)
        if not idx_paths:
            raise FileNotFoundError(f"未在 {folder} 找到单文件权重或 *.index.json")
        with open(idx_paths[0], "r", encoding="utf-8") as f:
            idx = json.load(f)
        weight_map = idx.get("weight_map", idx)
        shard_files = sorted({os.path.join(folder, name) for name in weight_map.values()})

    msd = model.state_dict()
    matched = 0
    skipped_mismatch = 0

    for path in shard_files:
        shard = _load_shard_state(path)
        to_load: Dict[str, torch.Tensor] = {}
        for k, v in shard.items():
            if k not in msd:
                continue
            if msd[k].shape != v.shape:
                skipped_mismatch += 1
                continue
            if v.is_floating_point():
                v = v.to(dtype=dtype)
            to_load[k] = v

        if not to_load:
            continue
        try:
            model.load_state_dict(to_load, strict=False, assign=True)
        except TypeError:
            model.load_state_dict(to_load, strict=False)
        matched += len(to_load)

    print(
        f"[load_ckpt] 已写入形状匹配的参数: {matched} 个张量；"
        f"跳过形状不一致: {skipped_mismatch}（多为自适应秩 vs checkpoint 固定 5712）"
    )


def _materialize_meta_parameters(model: torch.nn.Module, float_dtype: torch.dtype) -> None:
    """未载入的 meta 参数需在 CPU 上占位，否则后续 copy_ 无法写入。"""
    for name, param in list(model.named_parameters(recurse=True)):
        if not getattr(param, "is_meta", False):
            continue
        path_str, leaf = name.rsplit(".", 1)
        parent = model
        for p in path_str.split("."):
            if not p:
                continue
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        dt = float_dtype if param.is_floating_point() else param.dtype
        buf = torch.nn.Parameter(
            torch.empty(tuple(param.shape), dtype=dt, device=torch.device("cpu")),
            requires_grad=param.requires_grad,
        )
        setattr(parent, leaf, buf)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cmp_in", required=True)
    ap.add_argument("--cmp_out", required=True)
    ap.add_argument("--low_json", required=True)
    ap.add_argument("--basis_pt", required=True)
    ap.add_argument(
        "--tail_basis_pt",
        type=str,
        default="",
        help="可选：truncate_eigen_basis 导出的尾部 *_proj_basis，与 basis_pt 同 key；见 merge_tail_into_basis",
    )
    ap.add_argument(
        "--tail_alpha",
        type=float,
        default=0.0,
        help="尾部列缩放；0 表示不使用 tail",
    )
    ap.add_argument(
        "--tail_cols",
        type=int,
        default=64,
        help="每层每个分支用尾部的列数 t（与主基前 r−t 列拼接后 QR；t 过大则主基保留维数过少）",
    )
    ap.add_argument(
        "--tail_branches",
        type=str,
        default="v",
        help="应用尾部的分支：v / qk / qkv（逗号分隔，如 v 或 q,k,v）",
    )
    ap.add_argument("--dtype", default="float16", choices=["float16","bfloat16"])
    ap.add_argument("--keqk", type=int, default=0)  # 1: Uk=Uq for all layers
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype=="float16" else torch.bfloat16
    with open(args.low_json, "r", encoding="utf-8") as f:
        low_raw = json.load(f)

    # HF from_pretrained 在部分版本/路径下不会把 low_rank_config 传给 EigenAttn.__init__（建图仍用旧秩如 168），
    # 与 checkpoint 权重（5712）冲突。此处用 empty init + load_checkpoint_in_model，保证按 low_json 建图再加载权重。
    cmp_cfg = AutoConfig.from_pretrained(args.cmp_in, local_files_only=True)
    n_layers_cfg = int(cmp_cfg.num_hidden_layers)
    cmp_cfg.__dict__.pop("low_rank_config", None)
    cmp_cfg._attn_implementation = "eager"
    low = _normalize_low_rank_config(low_raw, n_layers_cfg)
    print(f"[low_rank] layer0 = {low[0]} (from {args.low_json})")

    B = _load_basis_pt(args.basis_pt)
    tail_path = (args.tail_basis_pt or "").strip()
    B_tail: Dict[str, torch.Tensor] = _load_basis_pt(tail_path) if tail_path else {}
    tail_branches: Set[str] = set()
    for part in (args.tail_branches or "v").lower().replace(" ", "").split(","):
        if not part:
            continue
        if part in ("q", "k", "v"):
            tail_branches.add(part)
        elif part == "qk":
            tail_branches.update(("q", "k"))
        elif part == "qkv":
            tail_branches.update(("q", "k", "v"))
        else:
            raise ValueError(f"--tail_branches 未知片段: {part!r}")
    use_tail = bool(B_tail) and float(args.tail_alpha) != 0.0 and len(tail_branches) > 0
    if use_tail:
        print(
            f"[tail] loaded tail_basis_pt={tail_path} keys={len(B_tail)} "
            f"alpha={args.tail_alpha} tail_cols={args.tail_cols} branches={sorted(tail_branches)}"
        )

    print("Load BASE (cpu)...")
    base=AutoModelForCausalLM.from_pretrained(
        args.base, local_files_only=True, torch_dtype=dtype, device_map=None, low_cpu_mem_usage=True
    ).cpu().eval()

    print("Load CMP_IN (cpu): init_empty_weights + 形状过滤加载 checkpoint ...")
    with ContextManagers([no_init_weights(_enable=True), init_empty_weights()]):
        cmpm = OPTForCausalLM_EigenAttn(cmp_cfg, low)
    if hasattr(cmpm, "tie_weights"):
        cmpm.tie_weights()
    load_checkpoint_shape_filtered(cmpm, args.cmp_in, dtype)
    _materialize_meta_parameters(cmpm, dtype)
    cmpm.eval()
    q0 = cmpm.model.decoder.layers[0].self_attn.q_proj_low.bias
    print(f"[sanity] layer0 q_proj_low.bias shape = {tuple(q0.shape)} (expect rank_kq={low[0][0]})")

    n_layers=base.config.num_hidden_layers
    d=base.config.hidden_size
    print(f"n_layers={n_layers} hidden={d} keqk={args.keqk}")

    for i in range(n_layers):
        b_attn=base.model.decoder.layers[i].self_attn
        c_attn=cmpm.model.decoder.layers[i].self_attn

        Uq=getU(B,i,"q")  # [d,rq]
        Uk=getU(B,i,"k")
        Uv=getU(B,i,"v")
        if use_tail:
            if "q" in tail_branches:
                Uq = merge_tail_into_basis(Uq, getU(B_tail, i, "q"), args.tail_cols, args.tail_alpha)
            if "k" in tail_branches:
                if args.keqk == 1:
                    if "q" not in tail_branches:
                        Uq = merge_tail_into_basis(Uq, getU(B_tail, i, "k"), args.tail_cols, args.tail_alpha)
                else:
                    Uk = merge_tail_into_basis(Uk, getU(B_tail, i, "k"), args.tail_cols, args.tail_alpha)
            if "v" in tail_branches:
                Uv = merge_tail_into_basis(Uv, getU(B_tail, i, "v"), args.tail_cols, args.tail_alpha)
        if args.keqk == 1:
            Uk = Uq

        # ---- Q/K/V low : Wlow = U^T @ W, blow = b @ U ----
        for kind, U, proj in [("q",Uq,"q_proj"), ("k",Uk,"k_proj"), ("v",Uv,"v_proj")]:
            W=getattr(b_attn, proj).weight.detach().float()  # [d,d]
            b=getattr(b_attn, proj).bias.detach().float()    # [d]

            low_layer=getattr(c_attn, f"{proj}_low")
            r = low_layer.weight.shape[0]
            assert U.shape[0]==d and U.shape[1]==r, f"U{kind} shape {tuple(U.shape)} != (d={d}, r={r})"

            Wlow=(U.t() @ W).contiguous()     # [r,d]
            blow=(b @ U).contiguous()         # [r]

            low_layer.weight.copy_(Wlow.to(dtype=low_layer.weight.dtype))
            low_layer.bias.copy_(blow.to(dtype=low_layer.bias.dtype))

        # ---- out_proj_up : Wup = Wo @ Uv, bias = bo ----
        Wo=b_attn.out_proj.weight.detach().float()  # [d,d]
        bo=b_attn.out_proj.bias.detach().float()    # [d]
        out_up = c_attn.out_proj_up
        r_v = out_up.weight.shape[1]
        assert Uv.shape==(d,r_v), f"Uv shape {tuple(Uv.shape)} != (d={d}, r_v={r_v})"

        Wup=(Wo @ Uv).contiguous()            # [d,r_v]
        out_up.weight.copy_(Wup.to(dtype=out_up.weight.dtype))
        if getattr(out_up, "bias", None) is not None:
            out_up.bias.copy_(bo.to(dtype=out_up.bias.dtype))

        # ---- 关键：用 headwise basis 重建后必须重置 v_rot 和 vscale ----
        # cmp_in 来自 global basis 的 final_model，其 v_rot 是为「全局低秩空间」学的；
        # 换成 headwise 后低秩空间已变，沿用旧 v_rot 会破坏输出甚至导致 cos 反相关。
        if hasattr(c_attn, "v_rot") and torch.is_tensor(getattr(c_attn.v_rot, "weight", None)):
            c_attn.v_rot.weight.copy_(torch.eye(c_attn.v_rot.weight.shape[0], dtype=c_attn.v_rot.weight.dtype))
        if hasattr(c_attn, "vscale_log") and torch.is_tensor(c_attn.vscale_log):
            c_attn.vscale_log.data.fill_(0.0)  # log(1)=0，避免幅度污染

        if i % 8 == 0:
            print("rebuilt layer", i)

    os.makedirs(args.cmp_out, exist_ok=True)
    cmpm.save_pretrained(args.cmp_out, safe_serialization=True)
    print("saved:", args.cmp_out)

if __name__=="__main__":
    main()
