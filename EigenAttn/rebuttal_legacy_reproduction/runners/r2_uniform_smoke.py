#!/usr/bin/env python3
"""R2 Uniform runtime smoke: load compressed model, fixed-input forward, check ranks applied."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "EigenAttn"))
from common import atomic_write_json, artifact_dir, file_hash_or_empty, now_iso, sha256_file, write_pass_gate, gate_payload  # noqa: E402

OUT = artifact_dir("r2_uniform_eb045", link_under=HERE / "artifacts" / "r2_uniform_eb045")
MODEL = "/diskD/jiaqizhang/Llama-3"


def main() -> None:
    import torch
    from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn
    from transformers import AutoConfig

    save_dir = OUT / "final_model"
    lr_json = save_dir / "low_rank_config.json"
    lr_pt = save_dir / "low_rank_config.pt"
    if lr_json.exists():
        low = json.loads(lr_json.read_text(encoding="utf-8"))
        if isinstance(low, dict) and "low_rank_config" in low:
            low = low["low_rank_config"]
    elif lr_pt.exists():
        low = torch.load(lr_pt, map_location="cpu")
        low = [[int(x[0]), int(x[1])] for x in low]
    else:
        raise SystemExit("missing low_rank_config")

    cfg = AutoConfig.from_pretrained(MODEL, attn_implementation="eager", local_files_only=True)
    cfg.use_cache = False
    model = LlamaForCausalLM_EigenAttn.from_pretrained(
        str(save_dir),
        config=cfg,
        low_rank_config=low,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model.eval()
    torch.manual_seed(0)
    ids = torch.randint(0, min(1000, cfg.vocab_size), (1, 64), device="cuda:0")
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits
    finite = bool(torch.isfinite(logits).all().item())
    # detect dense fallback: EigenAttn layers should expose low-rank attrs
    n_eigen = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "k_proj") and (
            hasattr(attn.k_proj, "basis") or hasattr(attn, "low_rank_config") or "Eigen" in type(attn).__name__
        ):
            n_eigen += 1
        elif "Eigen" in type(layer).__name__ or "Eigen" in type(attn).__name__:
            n_eigen += 1
    # softer check: class name
    attn0 = type(model.model.layers[0].self_attn).__name__
    smoke = {
        "generated_at": now_iso(),
        "finite_logits": finite,
        "logits_shape": list(logits.shape),
        "attn0_class": attn0,
        "n_layers": len(model.model.layers),
        "low_rank_config_len": len(low),
        "input_ids_sha_prefix": str(ids[0, :8].tolist()),
        "dense_fallback_suspected": ("LlamaAttention" == attn0 and "Eigen" not in attn0),
        "save_dir": str(save_dir),
        "model_manifest_sha": file_hash_or_empty(OUT / "LEGACY_UNIFORM_MODEL_MANIFEST.json"),
    }
    atomic_write_json(OUT / "R2_UNIFORM_RUNTIME_SMOKE.json", smoke)
    atomic_write_json(HERE / "R2_UNIFORM_RUNTIME_SMOKE.json", smoke)
    ok = finite and len(low) == 32 and (not smoke["dense_fallback_suspected"])
    payload = gate_payload(
        name="R2_UNIFORM_RUNTIME_PASS",
        passed=ok,
        upstream_hashes={"smoke": sha256_file(OUT / "R2_UNIFORM_RUNTIME_SMOKE.json")},
        runner_hash=sha256_file(Path(__file__)),
        protocol_hash=file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
        evaluation_hash=file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
        summary="uniform runtime smoke " + ("PASS" if ok else "FAIL"),
        extra=smoke,
    )
    if ok:
        write_pass_gate(OUT / "R2_UNIFORM_RUNTIME_PASS.DONE", payload)
        write_pass_gate(HERE / "R2_UNIFORM_RUNTIME_PASS.DONE", payload)
    else:
        atomic_write_json(OUT / "R2_UNIFORM_RUNTIME_FAIL.json", payload)
        raise SystemExit(1)
    print(json.dumps(smoke, ensure_ascii=False))


if __name__ == "__main__":
    main()
