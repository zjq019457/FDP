#!/usr/bin/env python3
"""Build or reuse operating point: Uniform + FDP with legacy builder pipeline."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
LEGACY = ROOT / "rebuttal_legacy_reproduction"
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    PYTHON_BIN,
    atomic_write_json,
    file_hash_or_empty,
    now_iso,
    op_artifact_dir,
    artifact_dir,
)
# Legacy helpers only after pipeline common is bound.
sys.path.insert(0, str(LEGACY))

MODEL = "/diskD/jiaqizhang/Llama-3"
WIKITEXT = "/home/bai-project/jiaqizhang/wikitext2"
CACHE = artifact_dir("HF_cache", link_under=ROOT / "EigenAttn" / "HF_cache_rebuttal_completion")
LEGACY_ART = Path("/diskF/jiaqizhang/rebuttal_legacy_reproduction_artifacts")
BASIS_FLAT = Path("/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_eb")

# Flat historical basis filenames (eb058/eb045 dirs are empty stubs).
UNIFORM_BASIS_FILES = {
    "045": [
        BASIS_FLAT / "eigen_basis_llama3_8b_wikitext2_eb0p045.pt",
        LEGACY_ART / "r2_uniform_eb045" / "eigen_basis_llama3_8b_wikitext2_eb0p045.pt",
    ],
    "060": [
        BASIS_FLAT / "eigen_basis_llama3_8b_wikitext2_eb0p06.pt",
        BASIS_FLAT / "eigen_basis_llama3_8b_wikitext2_eb0p058.pt",
        Path("/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_eb043/eigen_basis_llama3_8b_wikitext2_eb043_eb0p043.pt"),
    ],
    "090": [
        BASIS_FLAT / "eigen_basis_llama3_8b_wikitext2_eb0p09.pt",
        Path("/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_eb088/eigen_basis_llama3_8b_wikitext2_eb088_eb0p088.pt"),
    ],
}


def _run(cmd, cwd, env, log_prefix, out_root):
    lo, le = out_root / f"{log_prefix}_stdout.log", out_root / f"{log_prefix}_stderr.log"
    with lo.open("w") as fo, le.open("w") as fe:
        cp = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=fo, stderr=fe, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"{log_prefix} failed rc={cp.returncode}; see {le}")


def _resolve_uniform_basis(eb_tag: str) -> Path:
    for cand in UNIFORM_BASIS_FILES.get(eb_tag, []):
        if cand.is_file():
            return cand
    # Last resort: any .pt under flat basis dir or eb043
    for d in (BASIS_FLAT, Path("/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_eb043")):
        pts = sorted(d.glob("*.pt")) if d.is_dir() else []
        if pts:
            return pts[0]
    raise RuntimeError(f"no uniform basis available for eb{eb_tag}")


def _ensure_low_rank_json(model_dir: Path) -> None:
    """Write low_rank_config.json from .pt if missing (eval runners prefer json)."""
    js = model_dir / "low_rank_config.json"
    pt = model_dir / "low_rank_config.pt"
    if not js.exists() and pt.exists():
        import torch

        raw = torch.load(pt, map_location="cpu")
        low = [[int(x[0]), int(x[1])] for x in raw]
        atomic_write_json(js, low)  # plain list; LMClass/transformers choke on Tensor via .pt
    elif js.exists():
        # Normalize wrapped {"low_rank_config": [...]} -> plain list
        try:
            d = json.loads(js.read_text(encoding="utf-8"))
            if isinstance(d, dict) and "low_rank_config" in d:
                atomic_write_json(js, d["low_rank_config"])
        except json.JSONDecodeError:
            pass
    # Prefer JSON over Tensor .pt for zero-shot (GenerationConfig JSON log crash).
    if js.exists() and pt.exists():
        bak = model_dir / "low_rank_config.pt.bak"
        if not bak.exists():
            pt.rename(bak)
        else:
            pt.unlink()
    _sanitize_generation_config(model_dir)


def _sanitize_generation_config(model_dir: Path) -> None:
    """Remove low_rank_config accidentally embedded in generation_config.json.

    Transformers GenerationConfig.__repr__ JSON-encodes the config; a Tensor-valued
    low_rank_config field crashes zero-shot loading.
    """
    gc = model_dir / "generation_config.json"
    if not gc.exists():
        return
    try:
        d = json.loads(gc.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if "low_rank_config" in d:
        d.pop("low_rank_config")
        atomic_write_json(gc, d)


def _model_complete(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    if not ((model_dir / "low_rank_config.json").exists() or (model_dir / "low_rank_config.pt").exists()):
        return False
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        weight_map = json.loads(idx.read_text(encoding="utf-8")).get("weight_map", {})
        return all((model_dir / f).is_file() for f in set(weight_map.values()))
    return any(model_dir.glob("*.safetensors"))


def _kv_audit(low_rank_json: Path) -> dict:
    """Recompute from real ranks. Never reuse a stub retained-ratio constant."""
    sys.path.insert(0, str(HERE))
    from kv_accounting import audit_from_path, dense_kv_bytes  # noqa: WPS433

    pt = low_rank_json.with_suffix(".pt")
    src = low_rank_json if low_rank_json.exists() else pt
    if not src.exists():
        return {
            "realized_kv_bytes": 0,
            "total_rank_sum": 0,
            "dense_kv_bytes": dense_kv_bytes(),
            "error": "missing_low_rank_config",
        }
    audit = audit_from_path(src, seqlen=2048, bytes_per_elem=2, batch=1)
    return {
        "realized_kv_bytes": audit["realized_kv_bytes"],
        "total_rank_sum": audit["total_rank_sum"],
        "dense_kv_bytes": audit["dense_kv_bytes"],
        "retained_ratio": audit["retained_ratio"],
        "ranks_sha256": audit["ranks_sha256"],
        "formula": audit["formula"],
        "source": audit["source"],
    }


def _reuse_mild(out: Path) -> None:
    """Mild = legacy eb045 confirmatory artifacts (read-only copy of manifests)."""
    uni_src = LEGACY_ART / "r2_uniform_eb045"
    fdp_src = LEGACY_ART / "r3_fgeb045"
    for name, src in [("uniform", uni_src), ("fdp", fdp_src)]:
        dst = out / name
        dst.mkdir(parents=True, exist_ok=True)
        fm = src / "final_model"
        if fm.is_dir():
            manifest = {
                "reused_from": str(src),
                "final_model": str(fm),
                "low_rank_config_sha256": file_hash_or_empty(fm / "low_rank_config.json"),
                "sha256_basis": file_hash_or_empty(src / "eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt"),
            }
            atomic_write_json(dst / "BUILD_MANIFEST.json", manifest)
            # Convenience symlink so eval paths OP/.../uniform/final_model resolve.
            link = dst / "final_model"
            if link.is_symlink() or link.exists():
                if link.is_symlink():
                    link.unlink()
                elif link.is_dir() and not any(link.iterdir()):
                    link.rmdir()
            if not link.exists():
                link.symlink_to(fm)
    atomic_write_json(out / "OPERATING_POINT_MANIFEST.json", {
        "point": "mild", "eb_base": 0.045, "reused_legacy": True, "generated_at": now_iso(),
    })


def _try_recover_uniform(uni_out: Path, point: str) -> bool:
    """Reuse a complete prior uniform checkpoint if present (e.g. disk-full interrupted run)."""
    candidates = [
        uni_out,
        LEGACY_ART / "operating_points" / point / "uniform" / "final_model",
    ]
    for src in candidates:
        if not _model_complete(src):
            continue
        _ensure_low_rank_json(src)
        if src.resolve() != uni_out.resolve():
            uni_out.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.is_file():
                    dst = uni_out / f.name
                    if not dst.exists():
                        shutil.copy2(f, dst)
            _ensure_low_rank_json(uni_out)
        return _model_complete(uni_out)
    return False


def _build_point(point: str, eb_base: float, out: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "EigenAttn") + os.pathsep + env.get("PYTHONPATH", "")
    # Avoid HF network during compress; calibration cache is local.
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    eb_tag = f"{int(round(eb_base * 1000)):03d}"  # 0.045->045, 0.060->060, 0.090->090
    basis_hist = Path(f"/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb{eb_tag}")
    step1_hist = Path(f"/home/bai-project/jiaqizhang/step1_out/step1_llama3_8b_eb{eb_tag}.json")
    pl_hist = Path(f"/home/bai-project/jiaqizhang/step1_out/llama3_8b_wikitext2_frag_eb{eb_tag}.json")

    uni_out = out / "uniform" / "final_model"
    uni_out.mkdir(parents=True, exist_ok=True)
    if not _try_recover_uniform(uni_out, point):
        basis_pt = _resolve_uniform_basis(eb_tag)
        # output_dir parent of final_model so main saves once into uni_out
        uni_parent = out / "uniform"
        _run([
            PYTHON_BIN, "-u", str(ROOT / "EigenAttn" / "main_eigen_attn.py"),
            "--model", MODEL, "--net", "Llama-3-8b", "--dtype", "bf16",
            "--error_budget", str(eb_base), "--avg_dim", "4096",
            "--seqlen", "2048", "--nsamples", "64", "--seed", "2",
            "--calib_dataset", "c4", "--compress_only", "--attn_implementation", "eager",
            "--eigen_basis_path", str(basis_pt),
            "--save_dir", str(uni_out),
            "--output_dir", str(uni_parent),
            "--cache_dir", str(CACHE),
        ], ROOT / "EigenAttn", env, f"uni_{point}", out)
        _ensure_low_rank_json(uni_out)
        if not _model_complete(uni_out):
            raise RuntimeError(f"uniform model incomplete after build: {uni_out}")

    # FDP: step1 on uniform -> per_layer_eb -> adaptive basis -> compress
    fdp_root = out / "fdp"
    fdp_root.mkdir(parents=True, exist_ok=True)
    fdp_model = fdp_root / "final_model"
    if _model_complete(fdp_model):
        _ensure_low_rank_json(fdp_model)
    else:
        basis_dir = fdp_root / "adaptive_basis"
        basis_dir.mkdir(exist_ok=True)
        prefix = f"eigen_basis_{point}"
        basis_pt = basis_dir / f"{prefix}_perlayer_eb.pt"
        hist_pt = basis_hist / "eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt"
        if hist_pt.exists():
            shutil.copy2(hist_pt, basis_pt)
            if pl_hist.exists():
                shutil.copy2(pl_hist, fdp_root / "per_layer_eb.json")
        else:
            step1_out = fdp_root / f"step1_eb{eb_tag}.json"
            if step1_hist.exists():
                shutil.copy2(step1_hist, step1_out)
            else:
                _run([
                    PYTHON_BIN, "-u", str(ROOT / "EigenAttn" / "step1_diagnose_llama.py"),
                    "--base", MODEL, "--cmp", str(uni_out), "--tok", MODEL,
                    "--low_json", str(uni_out / "low_rank_config.json"),
                    "--dtype", "bf16", "--seqlen", "512", "--num_seqs", "2",
                    "--out", str(step1_out), "--attn_implementation", "eager",
                    "--gpu", "0", "--device", "cuda",
                ], ROOT / "EigenAttn", env, f"step1_{point}", out)

            pl_out = fdp_root / "per_layer_eb.json"
            if pl_hist.exists() and not pl_out.exists():
                shutil.copy2(pl_hist, pl_out)
            if not pl_out.exists():
                _run([
                    PYTHON_BIN, "-u", str(ROOT / "EigenAttn" / "per_layer_eb_from_step1.py"),
                    "--step1", str(step1_out), "--out", str(pl_out),
                    "--eb_base", str(eb_base), "--lambda_eb", "1.2", "--normalize", "mean",
                    "--freeze_layers", "0", "--wz", "1.0", "--ws", "1.0", "--wv", "1.0", "--wa", "1.0", "--wh", "1.0",
                ], ROOT / "EigenAttn", env, f"pl_{point}", out)

            if not basis_pt.exists():
                _run([
                    PYTHON_BIN, "-u", str(ROOT / "EigenAttn" / "tools" / "build_eigen_basis_llama_error_budget_offline.py"),
                    "--model", MODEL, "--data_dir", WIKITEXT, "--out_dir", str(basis_dir),
                    "--out_prefix", prefix, "--per_layer_eb_json", str(pl_out),
                    "--freeze_layer0_eb", "--layer0_eb", str(eb_base),
                    "--nsamples", "512", "--seqlen", "2048", "--dtype", "bf16",
                    "--attn_implementation", "eager", "--device", "cuda:0", "--seed", "2",
                ], ROOT / "EigenAttn", env, f"basis_{point}", out)

        fdp_model.mkdir(parents=True, exist_ok=True)
        _run([
            PYTHON_BIN, "-u", str(ROOT / "EigenAttn" / "main_eigen_attn.py"),
            "--model", MODEL, "--net", "Llama-3-8b", "--dtype", "bf16",
            "--error_budget", str(eb_base), "--avg_dim", "4096",
            "--seqlen", "2048", "--nsamples", "64", "--seed", "2",
            "--calib_dataset", "c4", "--compress_only", "--attn_implementation", "eager",
            "--eigen_basis_path", str(basis_pt),
            "--save_dir", str(fdp_model),
            "--output_dir", str(fdp_root),
            "--cache_dir", str(CACHE),
        ], ROOT / "EigenAttn", env, f"fdp_{point}", out)
        _ensure_low_rank_json(fdp_model)
        if not _model_complete(fdp_model):
            raise RuntimeError(f"fdp model incomplete after build: {fdp_model}")

    kv_uni = _kv_audit(uni_out / "low_rank_config.json")
    kv_fdp = _kv_audit(fdp_model / "low_rank_config.json")
    atomic_write_json(out / "OPERATING_POINT_MANIFEST.json", {
        "point": point, "eb_base": eb_base, "generated_at": now_iso(),
        "uniform_kv": kv_uni, "fdp_kv": kv_fdp,
        "uniform_model": str(uni_out), "fdp_model": str(fdp_model),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", required=True, choices=["mild", "moderate", "aggressive"])
    ap.add_argument("--eb_base", required=True)
    args = ap.parse_args()
    # Large checkpoints on diskF (headroom); mild only needs manifests/symlinks.
    if args.point == "mild":
        out = op_artifact_dir(f"operating_points/{args.point}", link_under=HERE / "artifacts" / f"OP_{args.point}")
    else:
        out = op_artifact_dir(f"operating_points/{args.point}", link_under=HERE / "artifacts" / f"OP_{args.point}")
    eb = float(args.eb_base)
    if args.point == "mild" and abs(eb - 0.045) < 1e-6:
        _reuse_mild(out)
    else:
        _build_point(args.point, eb, out)
    print(json.dumps({"ok": True, "point": args.point, "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
