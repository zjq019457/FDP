#!/usr/bin/env python3
"""R3 FGeb045 runtime smoke."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "EigenAttn"))
from common import (  # noqa: E402
    artifact_dir,
    atomic_write_json,
    file_hash_or_empty,
    gate_payload,
    now_iso,
    sha256_file,
    write_pass_gate,
)

OUT = artifact_dir("r3_fgeb045", link_under=HERE / "artifacts" / "r3_fgeb045")
MODEL = "/diskD/jiaqizhang/Llama-3"


def main() -> None:
    import torch
    from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn
    from transformers import AutoConfig

    save_dir = OUT / "final_model"
    lr_json = save_dir / "low_rank_config.json"
    if not lr_json.is_file():
        raise SystemExit(f"missing {lr_json}")
    low = json.loads(lr_json.read_text(encoding="utf-8"))
    if isinstance(low, dict) and "low_rank_config" in low:
        low = low["low_rank_config"]
    # Use base model config to keep attn implementation consistent with EigenAttn.
    cfg = AutoConfig.from_pretrained(MODEL, attn_implementation="eager", local_files_only=True)
    cfg.use_cache = False
    model = LlamaForCausalLM_EigenAttn.from_pretrained(
        str(save_dir),
        config=cfg,
        low_rank_config=low,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    ids = torch.tensor([[541, 97, 191, 172, 553, 604, 103, 733]], device="cuda:0")
    with torch.no_grad():
        out = model(input_ids=ids)
    logits = out.logits
    attn0 = model.model.layers[0].self_attn
    smoke = {
        "generated_at": now_iso(),
        "finite_logits": bool(torch.isfinite(logits).all().item()),
        "logits_shape": list(logits.shape),
        "attn0_class": type(attn0).__name__,
        "n_layers": len(model.model.layers),
        "low_rank_config_len": len(low) if isinstance(low, list) else None,
        "dense_fallback_suspected": "Eigen" not in type(attn0).__name__,
        "save_dir": str(save_dir),
        "model_manifest_sha": file_hash_or_empty(OUT / "LEGACY_FDP_MODEL_MANIFEST.json"),
    }
    atomic_write_json(OUT / "R3_FDP_RUNTIME_SMOKE.json", smoke)
    passed = smoke["finite_logits"] and not smoke["dense_fallback_suspected"] and smoke["n_layers"] == 32
    payload = gate_payload(
        name="R3_FDP_RUNTIME_PASS",
        passed=passed,
        upstream_hashes={"smoke": sha256_file(OUT / "R3_FDP_RUNTIME_SMOKE.json")},
        runner_hash=sha256_file(Path(__file__)),
        protocol_hash=file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
        evaluation_hash=file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
        summary="R3 FDP runtime smoke",
        extra=smoke,
    )
    if passed:
        write_pass_gate(OUT / "R3_FDP_RUNTIME_PASS.DONE", payload)
    else:
        atomic_write_json(OUT / "R3_FDP_RUNTIME_FAIL.json", payload)
        raise SystemExit("R3 smoke failed")
    print(json.dumps({"ok": True, "smoke": smoke}, ensure_ascii=False))


if __name__ == "__main__":
    main()
