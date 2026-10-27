#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import argparse
from pathlib import Path

import torch

try:
    from safetensors import safe_open
except ImportError:
    raise SystemExit("Please install safetensors first: pip install safetensors")


def flatten_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    na = a.norm()
    nb = b.norm()
    if na.item() == 0 or nb.item() == 0:
        return float("nan")
    return torch.dot(a, b).item() / (na.item() * nb.item())


def principal_subspace_similarity(Uq: torch.Tensor, Uk: torch.Tensor) -> float:
    """
    If Uq, Uk are orthonormal basis matrices with same shape [d, r],
    return ||Uq^T Uk||_F^2 / r in [0, 1] ideally.
    1.0 means identical subspace (not necessarily same signed columns).
    """
    Uq = Uq.float()
    Uk = Uk.float()
    r = min(Uq.shape[1], Uk.shape[1])
    M = Uq[:, :r].T @ Uk[:, :r]
    val = (M.pow(2).sum() / r).item()
    return val


class ShardedSafeTensorReader:
    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = Path(ckpt_dir)
        self.weight_map = {}
        self.single_file = None

        idx = self.ckpt_dir / "model.safetensors.index.json"
        single = self.ckpt_dir / "model.safetensors"

        if idx.exists():
            with open(idx, "r", encoding="utf-8") as f:
                obj = json.load(f)
            self.weight_map = obj["weight_map"]
        elif single.exists():
            self.single_file = str(single)
        else:
            raise FileNotFoundError(
                f"Cannot find model.safetensors or model.safetensors.index.json under {ckpt_dir}"
            )

        self._handles = {}

    def all_keys(self):
        if self.weight_map:
            return list(self.weight_map.keys())
        with safe_open(self.single_file, framework="pt", device="cpu") as f:
            return list(f.keys())

    def get_tensor(self, key: str) -> torch.Tensor:
        if self.weight_map:
            shard = self.ckpt_dir / self.weight_map[key]
            shard = str(shard)
        else:
            shard = self.single_file

        if shard not in self._handles:
            self._handles[shard] = safe_open(shard, framework="pt", device="cpu")
        return self._handles[shard].get_tensor(key)

    def close(self):
        self._handles.clear()


def find_layer_keys(all_keys, layer_idx: int, name: str):
    """
    Robustly find keys that end with:
      layers.{i}.self_attn.{name}
    regardless of prefix.
    """
    pat = re.compile(rf"(?:^|[.])layers\.{layer_idx}\.self_attn\.{re.escape(name)}$")
    return [k for k in all_keys if pat.search(k)]


def pick_one(cands):
    if len(cands) == 0:
        return None
    # Prefer key with base_model prefix if present, otherwise first
    for k in cands:
        if "base_model.model.model.decoder.layers" in k:
            return k
    return cands[0]


def infer_n_layers(all_keys):
    layer_ids = set()
    pat = re.compile(r"(?:^|[.])layers\.(\d+)\.self_attn\.")
    for k in all_keys:
        m = pat.search(k)
        if m:
            layer_ids.add(int(m.group(1)))
    if not layer_ids:
        return 0
    return max(layer_ids) + 1


def compare_tensors(a: torch.Tensor, b: torch.Tensor):
    out = {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "same_shape": tuple(a.shape) == tuple(b.shape),
    }
    if not out["same_shape"]:
        return out

    diff = (a.float() - b.float()).abs()
    out.update({
        "exact_equal": torch.equal(a, b),
        "allclose_1e-8": torch.allclose(a.float(), b.float(), atol=1e-8, rtol=0.0),
        "allclose_1e-6": torch.allclose(a.float(), b.float(), atol=1e-6, rtol=0.0),
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "flatten_cosine": flatten_cosine(a, b),
    })
    return out


def find_layer_keys_in_basis_dict(basis_dict: dict, layer_idx: int, name: str):
    """Match keys like ...layers.{i}.self_attn.{name} in an eigen_basis.pt dict."""
    pat = re.compile(rf"(?:^|[.])layers\.{layer_idx}\.self_attn\.{re.escape(name)}$")
    return [k for k in basis_dict.keys() if pat.search(k) and isinstance(basis_dict[k], torch.Tensor)]


def infer_n_layers_from_basis_dict(basis_dict: dict) -> int:
    layer_ids = set()
    pat = re.compile(r"(?:^|[.])layers\.(\d+)\.self_attn\.")
    for k in basis_dict.keys():
        m = pat.search(k)
        if m:
            layer_ids.add(int(m.group(1)))
    if not layer_ids:
        return 0
    return max(layer_ids) + 1


def main():
    ap = argparse.ArgumentParser(
        description="Compare q_proj_basis vs k_proj_basis (Uq vs Uk). "
        "HF checkpoints often omit basis tensors; use --basis_pt eigen_basis.pt in that case."
    )
    ap.add_argument(
        "--ckpt",
        default="",
        help="path to final_model directory (safetensors). Optional if --basis_pt is set.",
    )
    ap.add_argument(
        "--basis_pt",
        default="",
        help="path to eigen_basis.pt (or any dict of layer tensors). Preferred when ckpt has no *_proj_basis.",
    )
    ap.add_argument("--json_out", default="", help="optional json output path")
    args = ap.parse_args()

    if not args.ckpt and not args.basis_pt:
        raise SystemExit("Provide at least one of: --ckpt DIR, --basis_pt FILE")

    basis_dict: dict = {}
    reader = None
    all_keys: list = []

    if args.basis_pt:
        p = Path(args.basis_pt)
        if not p.is_file():
            raise FileNotFoundError(f"--basis_pt not found: {p}")
        obj = torch.load(p, map_location="cpu")
        if not isinstance(obj, dict):
            raise ValueError(f"expected dict in {p}, got {type(obj)}")
        basis_dict = obj
        print(f"[info] basis_pt={args.basis_pt}")

    if args.ckpt:
        reader = ShardedSafeTensorReader(args.ckpt)
        all_keys = reader.all_keys()
        print(f"[info] ckpt={args.ckpt}")

    n_layers_bd = infer_n_layers_from_basis_dict(basis_dict) if basis_dict else 0
    n_layers_ck = infer_n_layers(all_keys) if all_keys else 0
    n_layers = max(n_layers_bd, n_layers_ck)
    if n_layers == 0:
        raise SystemExit("Could not infer layer count from basis_pt or ckpt.")

    print(f"[info] discovered layers={n_layers} (basis_pt layers={n_layers_bd}, ckpt layers={n_layers_ck})")

    any_q_basis_pt = any("q_proj_basis" in k and "self_attn" in k for k in basis_dict)
    any_k_basis_pt = any("k_proj_basis" in k and "self_attn" in k for k in basis_dict)
    any_q_basis_ck = any(k.endswith("self_attn.q_proj_basis") for k in all_keys) if all_keys else False
    any_k_basis_ck = any(k.endswith("self_attn.k_proj_basis") for k in all_keys) if all_keys else False

    has_basis = (basis_dict and any_q_basis_pt and any_k_basis_ck) or (
        all_keys and any_q_basis_ck and any_k_basis_ck
    )
    # fix typo: any_k_basis_pt
    has_basis = (basis_dict and any_q_basis_pt and any_k_basis_pt) or (
        all_keys and any_q_basis_ck and any_k_basis_ck
    )

    if not has_basis:
        print("[warn] No q_proj_basis / k_proj_basis found in the inputs you provided.")
        print("[warn] HF save usually only stores low-rank weights; Uq/Uk live in eigen_basis.pt (or headwise .pt).")
        print("[warn] Example:  --basis_pt /path/to/eigen_basis.pt  [--ckpt final_model for low-weight compare]")
        return

    results = {
        "ckpt": args.ckpt or None,
        "basis_pt": args.basis_pt or None,
        "n_layers": n_layers,
        "basis_present": True,
        "layers": [],
    }

    exact_equal_layers = []
    same_subspace_layers = []

    def pick_basis_keys(li: int):
        if basis_dict:
            qk = pick_one(find_layer_keys_in_basis_dict(basis_dict, li, "q_proj_basis"))
            kk = pick_one(find_layer_keys_in_basis_dict(basis_dict, li, "k_proj_basis"))
            if qk and kk:
                return qk, kk, "dict"
        if all_keys:
            qk = pick_one(find_layer_keys(all_keys, li, "q_proj_basis"))
            kk = pick_one(find_layer_keys(all_keys, li, "k_proj_basis"))
            if qk and kk:
                return qk, kk, "ckpt"
        return None, None, None

    def pick_low_keys(li: int):
        if all_keys:
            qk = pick_one(find_layer_keys(all_keys, li, "q_proj_low.weight"))
            kk = pick_one(find_layer_keys(all_keys, li, "k_proj_low.weight"))
            if qk and kk:
                return qk, kk, "ckpt"
        if basis_dict:
            qk = pick_one(find_layer_keys_in_basis_dict(basis_dict, li, "q_proj_low.weight"))
            kk = pick_one(find_layer_keys_in_basis_dict(basis_dict, li, "k_proj_low.weight"))
            if qk and kk:
                return qk, kk, "dict"
        return None, None, None

    def get_tensor_for(key: str, src: str) -> torch.Tensor:
        if src == "dict":
            return basis_dict[key]
        if reader is None:
            raise RuntimeError("ckpt reader missing for safetensors key")
        return reader.get_tensor(key)

    for li in range(n_layers):
        q_basis_key, k_basis_key, basis_src = pick_basis_keys(li)
        q_low_key, k_low_key, low_src = pick_low_keys(li)

        layer_info = {
            "layer": li,
            "q_basis_key": q_basis_key,
            "k_basis_key": k_basis_key,
            "q_low_key": q_low_key,
            "k_low_key": k_low_key,
        }

        if q_basis_key is not None and k_basis_key is not None and basis_src is not None:
            Uq = get_tensor_for(q_basis_key, basis_src)
            Uk = get_tensor_for(k_basis_key, basis_src)
            basis_cmp = compare_tensors(Uq, Uk)
            if basis_cmp.get("same_shape", False):
                basis_cmp["subspace_similarity_frob"] = principal_subspace_similarity(Uq, Uk)
            layer_info["basis_cmp"] = basis_cmp

            if basis_cmp.get("exact_equal", False):
                exact_equal_layers.append(li)

            if basis_cmp.get("same_shape", False) and basis_cmp.get("subspace_similarity_frob", 0.0) > 0.999999:
                same_subspace_layers.append(li)

        if q_low_key is not None and k_low_key is not None and low_src is not None:
            Wq = get_tensor_for(q_low_key, low_src)
            Wk = get_tensor_for(k_low_key, low_src)
            layer_info["low_weight_cmp"] = compare_tensors(Wq, Wk)

        results["layers"].append(layer_info)

    if reader is not None:
        reader.close()

    print("\n=== Summary ===")
    print(f"layers with EXACT Uq == Uk: {exact_equal_layers}")
    print(f"count: {len(exact_equal_layers)} / {n_layers}")
    print(f"layers with nearly identical Q/K subspace: {same_subspace_layers}")
    print(f"count: {len(same_subspace_layers)} / {n_layers}")

    # Simple human-readable interpretation
    if len(exact_equal_layers) == n_layers and n_layers > 0:
        print("[conclusion] Uq and Uk tensors are exactly equal in all layers (for the loaded basis source).")
    elif len(exact_equal_layers) == 0:
        print("[conclusion] No layer has exact Uq == Uk (or basis pair missing per layer).")
    else:
        print("[conclusion] Mixed: some layers exact Uq == Uk, some do not.")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[saved] {args.json_out}")


if __name__ == "__main__":
    main()