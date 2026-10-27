#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, math
from typing import Dict, Tuple, Optional

import torch


def _load_pt(path: str):
    obj = torch.load(path, map_location="cpu")
    # 有些会包一层 {"state_dict": ...}
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


def _find_basis(sd: Dict, layer: int, kind: str) -> torch.Tensor:
    """
    兼容两种常见 key：
    1) ...decoder.layers.{layer}.self_attn.{q/k/v}_proj_basis
    2) layer_{layer}.*basis_kq / basis_v / q_basis 等
    """
    cand = []

    # 形式 1：完整路径
    key1 = f"decoder.layers.{layer}.self_attn.{kind}_proj_basis"
    for k in sd.keys():
        if key1 in k:
            cand.append(k)

    # 形式 2：layer_x + basis_x
    if not cand:
        layer_tag = f"layer_{layer}"
        for k in sd.keys():
            if layer_tag in k:
                if kind in ("q", "k") and ("basis_kq" in k or "q_basis" in k or "k_basis" in k):
                    cand.append(k)
                if kind == "v" and ("basis_v" in k or "v_basis" in k):
                    cand.append(k)

    if not cand:
        # 最后兜底：包含 layers.{layer} 和 kind 以及 basis
        for k in sd.keys():
            if f"layers.{layer}" in k and kind in k and "basis" in k:
                cand.append(k)

    if not cand:
        raise KeyError(f"Cannot find basis for layer={layer} kind={kind}. "
                       f"Try printing keys in eigen_basis.pt.")

    # 如果找到多个，优先更具体的
    cand = sorted(cand, key=len)
    key = cand[0]
    U = sd[key]
    if not torch.is_tensor(U):
        raise TypeError(f"Found key={key} but value is not a tensor: {type(U)}")

    # squeeze [1, d, r] -> [d, r]
    if U.dim() == 3 and U.shape[0] == 1:
        U = U.squeeze(0)
    return U.contiguous()


def _orth_err(U: torch.Tensor) -> float:
    # ||U^T U - I||_F / ||I||_F
    r = U.shape[1]
    UtU = U.T @ U
    I = torch.eye(r, device=UtU.device, dtype=UtU.dtype)
    num = torch.linalg.norm(UtU - I).item()
    den = torch.linalg.norm(I).item()
    return float(num / (den + 1e-12))


@torch.no_grad()
def _metrics(W: torch.Tensor, U: torch.Tensor, side: str, nsamples: int = 2048, device: str = "cpu") -> Dict:
    """
    W: [out, in]  (PyTorch Linear weight)
    U: [d, r]
    side:
      - "left":  W_hat = U (U^T W)
      - "right": W_hat = (W U) U^T
    激活用随机高斯输入近似（足够判断“方向是否用错”）
    """
    W = W.to(device=device, dtype=torch.float32)
    U = U.to(device=device, dtype=torch.float32)

    # 让 U 列向量正交（避免 U^T U != I 造成投影崩）
    Uo, _ = torch.linalg.qr(U, mode="reduced")  # [d, r]

    if side == "left":
        W_hat = Uo @ (Uo.T @ W)
    elif side == "right":
        W_hat = (W @ Uo) @ Uo.T
    else:
        raise ValueError(side)

    w_rel = (torch.linalg.norm(W - W_hat) / (torch.linalg.norm(W) + 1e-12)).item()

    # activation-level
    # y = x @ W^T, x:[n,in], y:[n,out]
    n = nsamples
    in_dim = W.shape[1]
    X = torch.randn(n, in_dim, device=device, dtype=torch.float32)
    Y = X @ W.T
    Yh = X @ W_hat.T

    # cosine per-sample
    cos = torch.nn.functional.cosine_similarity(Y, Yh, dim=-1)
    rel = torch.linalg.norm(Y - Yh, dim=-1) / (torch.linalg.norm(Y, dim=-1) + 1e-12)

    def pct(t, p):
        k = max(0, min(len(t)-1, int(math.floor(p * (len(t)-1)))))
        return float(torch.kthvalue(t, k+1).values.item())

    cos_sorted, _ = torch.sort(cos)
    rel_sorted, _ = torch.sort(rel)

    return {
        "weight_rel_fro": float(w_rel),
        "act_cos_mean": float(cos.mean().item()),
        "act_cos_p50": float(cos_sorted[len(cos_sorted)//2].item()),
        "act_cos_p95": float(cos_sorted[int(0.95*len(cos_sorted))-1].item()),
        "act_relL2_mean": float(rel.mean().item()),
        "act_relL2_p50": float(rel_sorted[len(rel_sorted)//2].item()),
        "act_relL2_p95": float(rel_sorted[int(0.95*len(rel_sorted))-1].item()),
        "qr_orth_err": float(_orth_err(Uo)),
        "r": int(Uo.shape[1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--eigen_basis", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    out_dir = args.out_dir
    layer = args.layer

    qkv_path = os.path.join(out_dir, "qkv_exports", f"qkv_mats_layer{layer}.pt")
    if not os.path.exists(qkv_path):
        raise FileNotFoundError(f"Missing {qkv_path} (run with --qkv_export first).")

    qkv = _load_pt(qkv_path)
    # 兼容 dict 或 tuple/list
    if isinstance(qkv, dict):
        # --- replace the buggy "or" chains with explicit checks ---
        Wq = qkv.get("Wq", None)
        if Wq is None: Wq = qkv.get("q", None)
        if Wq is None: Wq = qkv.get("Q", None)
        if Wq is None: Wq = qkv.get("wq", None)

        Wk = qkv.get("Wk", None)
        if Wk is None: Wk = qkv.get("k", None)
        if Wk is None: Wk = qkv.get("K", None)
        if Wk is None: Wk = qkv.get("wk", None)

        Wv = qkv.get("Wv", None)
        if Wv is None: Wv = qkv.get("v", None)
        if Wv is None: Wv = qkv.get("V", None)
        if Wv is None: Wv = qkv.get("wv", None)

        if Wq is None or Wk is None or Wv is None:
            # 如果 dict 里不是这些 key，就尝试按排序取前三个 tensor
            ts = [v for v in qkv.values() if torch.is_tensor(v)]
            if len(ts) >= 3:
                Wq, Wk, Wv = ts[:3]
            else:
                raise KeyError(f"Cannot parse qkv mats keys from {qkv_path}. keys={list(qkv.keys())[:20]}")
    else:
        # tuple/list
        Wq, Wk, Wv = qkv

    sd = _load_pt(args.eigen_basis)
    if not isinstance(sd, dict):
        raise TypeError(f"eigen_basis.pt is not a dict-like state_dict: {type(sd)}")

    Uq = _find_basis(sd, layer, "q")
    Uk = _find_basis(sd, layer, "k")
    Uv = _find_basis(sd, layer, "v")

    dev = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    report = []
    for kind, W, U in [("q", Wq, Uq), ("k", Wk, Uk), ("v", Wv, Uv)]:
        orth_raw = _orth_err(U.to(torch.float32))
        m_left  = _metrics(W, U, "left",  device=dev)
        m_right = _metrics(W, U, "right", device=dev)

        # 选更好的（优先看 act_cos_mean，其次 act_relL2_mean）
        better = "right" if (m_right["act_cos_mean"] > m_left["act_cos_mean"]) else "left"

        report.append({
            "kind": kind,
            "U_shape": list(U.shape),
            "orth_err_raw": orth_raw,
            "left": m_left,
            "right": m_right,
            "better": better,
        })

    txt_path = os.path.join(out_dir, f"diag_basis_side_layer{layer}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in report:
            f.write(f"=== layer {layer} kind={r['kind']} U_shape={r['U_shape']} orth_err_raw={r['orth_err_raw']:.6f}\n")
            for side in ("left", "right"):
                m = r[side]
                f.write(f"  [{side}] weight_rel_fro={m['weight_rel_fro']:.4f} "
                        f"act_cos_mean={m['act_cos_mean']:.4f} act_relL2_mean={m['act_relL2_mean']:.4f} "
                        f"qr_orth_err={m['qr_orth_err']:.6f}\n")
            f.write(f"  ==> better_side = {r['better']}\n\n")

    print(f"[OK] wrote {txt_path}")
    print("If your pipeline currently uses basis_side=left(out) but report says better=right,")
    print("you should switch to right(in) projection in eigen_attn.py (or implement auto).")


if __name__ == "__main__":
    main()
