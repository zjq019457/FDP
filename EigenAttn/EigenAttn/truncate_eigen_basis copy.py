import argparse
import torch
from typing import Any

def truncate_tensor(t: torch.Tensor, r: int) -> torch.Tensor:
    # 只处理 2D basis 矩阵；其他张量原样返回
    if not torch.is_tensor(t) or t.ndim != 2:
        return t

    m, n = t.shape

    # 常见两种存法：
    # 1) [in_dim, full_rank] or [7168, 7168] -> 取前 r 列
    if n >= r and m >= r:
        # 优先假设列是“rank 维”（更常见：basis = [d, r]）
        if m == 7168 and n >= r:
            return t[:, :r].contiguous()
        # 另一种：basis = [r, d]
        if n == 7168 and m >= r:
            return t[:r, :].contiguous()

    # 兜底：如果某一维正好是 7168，截掉另一维
    if m == 7168 and n >= r:
        return t[:, :r].contiguous()
    if n == 7168 and m >= r:
        return t[:r, :].contiguous()

    # 其他形状不动
    return t

def walk(obj: Any, r: int, stats: dict) -> Any:
    if torch.is_tensor(obj):
        before = tuple(obj.shape)
        out = truncate_tensor(obj, r)
        after = tuple(out.shape) if torch.is_tensor(out) else before
        if before != after:
            stats["changed"] += 1
            stats["changes"].append((before, after, str(obj.dtype)))
        stats["tensors"] += 1
        return out

    if isinstance(obj, dict):
        return {k: walk(v, r, stats) for k, v in obj.items()}
    if isinstance(obj, list):
        return [walk(v, r, stats) for v in obj]
    if isinstance(obj, tuple):
        return tuple(walk(v, r, stats) for v in obj)

    return obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--rank", type=int, required=True)
    args = ap.parse_args()

    data = torch.load(args.in_path, map_location="cpu")
    stats = {"tensors": 0, "changed": 0, "changes": []}

    new_data = walk(data, args.rank, stats)

    print(f"[truncate] total tensors={stats['tensors']} changed={stats['changed']}")
    # 只打印前 30 条变化，避免刷屏
    for i, (b, a, dt) in enumerate(stats["changes"][:30]):
        print(f"  #{i}: {b} -> {a} dtype={dt}")
    if stats["changed"] > 30:
        print(f"  ... ({stats['changed']-30} more)")

    torch.save(new_data, args.out_path)
    print(f"[truncate] saved to: {args.out_path}")

if __name__ == "__main__":
    main()
