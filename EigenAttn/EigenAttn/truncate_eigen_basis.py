import argparse
import torch
from typing import Any

def split_tensor(t: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    # 只处理 2D basis 矩阵；其他张量原样返回
    if not torch.is_tensor(t) or t.ndim != 2:
        return t, t

    m, n = t.shape

    # 常见两种存法：
    # 1) [in_dim, full_rank] or [7168, 7168] -> 取前 r 列
    if n >= r and m >= r:
        # 优先假设列是“rank 维”（更常见：basis = [d, r]）
        if m == 7168 and n >= r:
            return t[:, :r].contiguous(), t[:, r:].contiguous()
        # 另一种：basis = [r, d]
        if n == 7168 and m >= r:
            return t[:r, :].contiguous(), t[r:, :].contiguous()

    # 兜底：如果某一维正好是 7168，截掉另一维
    if m == 7168 and n >= r:
        return t[:, :r].contiguous(), t[:, r:].contiguous()
    if n == 7168 and m >= r:
        return t[:r, :].contiguous(), t[r:, :].contiguous()

    # 其他形状不动
    return t, t

def walk_split(obj: Any, r: int, stats: dict) -> tuple[Any, Any]:
    if torch.is_tensor(obj):
        before = tuple(obj.shape)
        main_t, tail_t = split_tensor(obj, r)
        after_main = tuple(main_t.shape) if torch.is_tensor(main_t) else before
        after_tail = tuple(tail_t.shape) if torch.is_tensor(tail_t) else before
        if before != after_main:
            stats["changed"] += 1
            stats["changes"].append((before, after_main, after_tail, str(obj.dtype)))
        stats["tensors"] += 1
        return main_t, tail_t

    if isinstance(obj, dict):
        main_d = {}
        tail_d = {}
        for k, v in obj.items():
            mv, tv = walk_split(v, r, stats)
            main_d[k] = mv
            tail_d[k] = tv
        return main_d, tail_d
    if isinstance(obj, list):
        main_l = []
        tail_l = []
        for v in obj:
            mv, tv = walk_split(v, r, stats)
            main_l.append(mv)
            tail_l.append(tv)
        return main_l, tail_l
    if isinstance(obj, tuple):
        main_t = []
        tail_t = []
        for v in obj:
            mv, tv = walk_split(v, r, stats)
            main_t.append(mv)
            tail_t.append(tv)
        return tuple(main_t), tuple(tail_t)

    return obj, obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument(
        "--tail_out_path",
        default="",
        help="可选：输出被截断的尾部 basis；未设置则不写 tail 文件",
    )
    ap.add_argument("--rank", type=int, required=True)
    args = ap.parse_args()

    data = torch.load(args.in_path, map_location="cpu")
    stats = {"tensors": 0, "changed": 0, "changes": []}

    main_data, tail_data = walk_split(data, args.rank, stats)

    print(f"[truncate] total tensors={stats['tensors']} changed={stats['changed']}")
    # 只打印前 30 条变化，避免刷屏
    for i, (b, am, at, dt) in enumerate(stats["changes"][:30]):
        print(f"  #{i}: main {b} -> {am}, tail -> {at}, dtype={dt}")
    if stats["changed"] > 30:
        print(f"  ... ({stats['changed']-30} more)")

    torch.save(main_data, args.out_path)
    print(f"[truncate] saved to: {args.out_path}")
    if args.tail_out_path:
        torch.save(tail_data, args.tail_out_path)
        print(f"[truncate] tail saved to: {args.tail_out_path}")

if __name__ == "__main__":
    main()
