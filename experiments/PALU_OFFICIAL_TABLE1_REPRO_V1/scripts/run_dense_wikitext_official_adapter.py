#!/usr/bin/env python3
"""Run untouched official PaLu WikiText-2 PPL with verified local official data rows.
Only the Hub dataset resolution and an import-only unused-extension shim are adapted.
"""
import hashlib, os, runpy, sys, types
from pathlib import Path
ROOT = Path("/diskE/jiaqizhang/PALU_OFFICIAL_TABLE1_REPRO_V1")
OFFICIAL = Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/Palu")
DATA = ROOT / "inputs/wikitext2_raw_v1"
FILES = {"test": DATA / "test-00000-of-00001.parquet", "train": DATA / "train-00000-of-00001.parquet"}
EXPECTED = {
    "test": "3ee89cd6a2ab912afd5d01e98867b46a67d9ec7a9eca0910e5e4c3cdd4cc1925",
    "train": "49280f3dd993cbdfd322186194133d5d1935f81e1c91064c239f2a70ebb02701",
}
for split, path in FILES.items():
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != EXPECTED[split]:
        raise RuntimeError(f"pinned {split} parquet hash mismatch: {actual}")
had_calls = {"count": 0}
def _forbidden_hadamard(*args, **kwargs):
    had_calls["count"] += 1
    raise RuntimeError("Hadamard extension was unexpectedly invoked in unquantized Dense PPL")
shim = types.ModuleType("fast_hadamard_transform")
shim.hadamard_transform = _forbidden_hadamard
sys.modules.setdefault("fast_hadamard_transform", shim)
import datasets
_official_load_dataset = datasets.load_dataset
def _offline_official_wikitext(path, *args, **kwargs):
    if path == "wikitext" and args and args[0] == "wikitext-2-raw-v1":
        split = kwargs.get("split")
        if split not in FILES:
            raise RuntimeError(f"unexpected WikiText split request: {split!r}")
        print(f"OFFLINE_DATA_ADAPTER split={split} source={FILES[split]} sha256={EXPECTED[split]}", flush=True)
        return _official_load_dataset("parquet", data_files={split: str(FILES[split])}, split=split)
    return _official_load_dataset(path, *args, **kwargs)
datasets.load_dataset = _offline_official_wikitext
sys.path.insert(0, str(OFFICIAL))
model = os.environ["PALU_REPRO_MODEL_ALIAS"]
sys.argv = [str(OFFICIAL / "run_ppl_eval.py"), "--model_name_or_path", model,
            "--datasets", "wikitext2", "--seqlen", "2048", "--device", "cuda"]
runpy.run_path(str(OFFICIAL / "run_ppl_eval.py"), run_name="__main__")
print(f"HADAMARD_GUARD_CALLS={had_calls['count']}", flush=True)
if had_calls["count"]:
    raise RuntimeError("unexpected Hadamard invocation")
