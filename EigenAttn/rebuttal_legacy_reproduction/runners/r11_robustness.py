#!/usr/bin/env python3
"""R11 Robustness sweep: vary calibration size (16,32,64) and lambda_eb (0.6,1.2,2.0).
For each (cal_size, lambda) pair, rebuild eigen basis + compress + measure WikiText-2 PPL."""
from __future__ import annotations

import gc
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "EigenAttn"))
from common import PYTHON_BIN, artifact_dir, atomic_write_json, file_hash_or_empty, gate_payload, now_iso, sha256_file, write_pass_gate  # noqa: E402

OUT = Path("/diskE/jiaqizhang/r11_robustness")  # diskE has more free space
MODEL = "/diskD/jiaqizhang/Llama-3"
WIKITEXT = "/home/bai-project/jiaqizhang/wikitext2"
CACHE = artifact_dir("HF_cache", link_under=ROOT / "EigenAttn" / "HF_cache_rebuttal")

STEP1_JSON = artifact_dir("r3_fgeb045", link_under=HERE / "artifacts" / "r3_fgeb045") / "step1_llama3_8b_uniform_eb045.json"
PL_EB_SCRIPT = ROOT / "EigenAttn" / "per_layer_eb_from_step1.py"
BUILDER = ROOT / "EigenAttn" / "tools" / "build_eigen_basis_llama_error_budget_offline.py"
MAIN = ROOT / "EigenAttn" / "main_eigen_attn.py"

CAL_SIZES = [32, 64, 128]
LAMBDAS = [0.6, 1.2, 2.0]
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
SEQLEN = 2048


def _load_wikitext2_text() -> str:
    raw = Path(WIKITEXT) / "wiki.test.raw"
    if raw.exists():
        return raw.read_text(encoding="utf-8")
    parquet = Path(WIKITEXT) / "test-00000-of-00001.parquet"
    if parquet.exists():
        import pyarrow.parquet as pq
        table = pq.read_table(str(parquet))
        texts = table.column("text").to_pylist()
        return "\n".join(t for t in texts if t.strip())
    raise FileNotFoundError(f"No WikiText-2 test data at {WIKITEXT}")


def _eval_ppl(model, tokenizer) -> float:
    text = _load_wikitext2_text()
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(DEVICE)
    nlls = []
    for i in range(0, input_ids.size(1) - 1, SEQLEN):
        end = min(i + SEQLEN, input_ids.size(1))
        ids = input_ids[:, i:end]
        with torch.no_grad():
            out = model(input_ids=ids, labels=ids)
        nlls.append(out.loss.item() * (end - i - 1))
    return math.exp(sum(nlls) / (input_ids.size(1) - 1))


def _run(cmd, cwd, env, log_prefix):
    log_out = OUT / f"{log_prefix}_stdout.log"
    log_err = OUT / f"{log_prefix}_stderr.log"
    with log_out.open("w") as fo, log_err.open("w") as fe:
        cp = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=fo, stderr=fe, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"{log_prefix} failed rc={cp.returncode}; see {log_err}")


def main() -> None:
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "EigenAttn") + os.pathsep + env.get("PYTHONPATH", "")

    results: List[Dict[str, Any]] = []

    for lam in LAMBDAS:
        for cal in CAL_SIZES:
            tag = f"lam{lam}_cal{cal}"
            print(f"[{now_iso()}] === {tag} ===")
            work = OUT / tag
            work.mkdir(parents=True, exist_ok=True)

            # 1. Per-layer EB
            pl_out = work / "per_layer_eb.json"
            cmd_pl = [
                PYTHON_BIN, "-u", str(PL_EB_SCRIPT),
                "--step1", str(STEP1_JSON),
                "--out", str(pl_out),
                "--eb_base", "0.045",
                "--lambda_eb", str(lam),
                "--normalize", "mean",
                "--freeze_layers", "0",
                "--wz", "1.0", "--ws", "1.0", "--wv", "1.0", "--wa", "1.0", "--wh", "1.0",
            ]
            _run(cmd_pl, ROOT / "EigenAttn", env, f"{tag}_perlayer")

            # 2. Build basis
            basis_dir = work / "basis"
            basis_dir.mkdir(exist_ok=True)
            prefix = f"eigen_basis_{tag}"
            cmd_basis = [
                PYTHON_BIN, "-u", str(BUILDER),
                "--model", MODEL,
                "--data_dir", WIKITEXT,
                "--out_dir", str(basis_dir),
                "--out_prefix", prefix,
                "--per_layer_eb_json", str(pl_out),
                "--freeze_layer0_eb", "--layer0_eb", "0.045",
                "--nsamples", str(cal),
                "--seqlen", "2048",
                "--dtype", "bf16",
                "--attn_implementation", "eager",
                "--device", DEVICE,
                "--seed", "2",
            ]
            _run(cmd_basis, ROOT / "EigenAttn", env, f"{tag}_basis")
            basis_pt = basis_dir / f"{prefix}_perlayer_eb.pt"

            # 3. Compress
            save_dir = work / "final_model"
            run_dir = work / "compress_run"
            save_dir.mkdir(exist_ok=True)
            run_dir.mkdir(exist_ok=True)
            cmd_compress = [
                PYTHON_BIN, "-u", str(MAIN),
                "--model", MODEL,
                "--net", "Llama-3-8b",
                "--dtype", "bf16",
                "--error_budget", "0.045",
                "--avg_dim", "4096",
                "--seqlen", "2048",
                "--nsamples", "64",
                "--seed", "2",
                "--calib_dataset", "c4",
                "--compress_only",
                "--attn_implementation", "eager",
                "--eigen_basis_path", str(basis_pt),
                "--save_dir", str(save_dir),
                "--output_dir", str(run_dir),
                "--cache_dir", str(CACHE),
            ]
            _run(cmd_compress, ROOT / "EigenAttn", env, f"{tag}_compress")

            # Find low_rank_config
            lr_json = save_dir / "low_rank_config.json"
            if not lr_json.exists():
                alt = run_dir / "final_model" / "low_rank_config.json"
                if alt.exists():
                    shutil.copy2(alt, lr_json)
                    pt = alt.with_suffix(".pt")
                    if pt.exists():
                        shutil.copy2(pt, save_dir / "low_rank_config.pt")

            # 4. Eval PPL
            from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn
            from transformers import AutoConfig
            low = json.loads(lr_json.read_text(encoding="utf-8"))
            if isinstance(low, dict) and "low_rank_config" in low:
                low = low["low_rank_config"]
            cfg = AutoConfig.from_pretrained(MODEL, attn_implementation="eager", local_files_only=True)
            cfg.use_cache = False
            model = LlamaForCausalLM_EigenAttn.from_pretrained(
                str(save_dir), config=cfg, low_rank_config=low,
                torch_dtype=DTYPE, low_cpu_mem_usage=True,
                device_map={"": DEVICE}, attn_implementation="eager", local_files_only=True,
            )
            model.eval()
            ppl = _eval_ppl(model, tokenizer)
            del model; gc.collect(); torch.cuda.empty_cache()

            results.append({
                "lambda_eb": lam,
                "cal_size": cal,
                "wikitext2_ppl": round(ppl, 4),
                "tag": tag,
            })
            print(f"  {tag}: PPL={ppl:.4f}")

            # Free disk: remove large model files, keep only config and logs
            for large in (save_dir, run_dir, basis_dir):
                for f in large.rglob("*.safetensors"):
                    f.unlink()
                for f in large.rglob("*.bin"):
                    f.unlink()

    atomic_write_json(OUT / "R11_ROBUSTNESS.json", {
        "generated_at": now_iso(),
        "results": results,
        "cal_sizes": CAL_SIZES,
        "lambdas": LAMBDAS,
    })

    md = ["# R11 Robustness Sweep", "", f"- generated_at: `{now_iso()}`", ""]
    md.append("| λ_eb | Cal Size | WikiText-2 PPL |")
    md.append("|------|----------|----------------|")
    for r in results:
        md.append(f"| {r['lambda_eb']} | {r['cal_size']} | {r['wikitext2_ppl']} |")
    (OUT / "R11_ROBUSTNESS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (HERE / "R11_ROBUSTNESS.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    payload = gate_payload(
        name="R11_ROBUSTNESS_PASS",
        passed=True,
        upstream_hashes={"robustness": sha256_file(OUT / "R11_ROBUSTNESS.json")},
        runner_hash=sha256_file(Path(__file__)),
        protocol_hash=file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
        evaluation_hash=file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
        summary="R11 robustness sweep complete",
    )
    # Copy key results back to main artifact area
    results_dir = artifact_dir("r11_robustness", link_under=HERE / "artifacts" / "r11_robustness")
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / "R11_ROBUSTNESS.json", results_dir / "R11_ROBUSTNESS.json")
    shutil.copy2(OUT / "R11_ROBUSTNESS.md", results_dir / "R11_ROBUSTNESS.md")
    write_pass_gate(results_dir / "R11_ROBUSTNESS_PASS.DONE", payload)
    write_pass_gate(HERE / "R11_ROBUSTNESS_PASS.DONE", payload)
    print(json.dumps({"passed": True, "n_configs": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
 