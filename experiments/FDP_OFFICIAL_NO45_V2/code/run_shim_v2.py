#!/usr/bin/env python3
"""FDP_OFFICIAL_NO45_V2 shim (derived from FDP_OFFICIAL_PROTOCOL_V1). Runs upstream 8bd0d3f either unchanged (official)
or the one-line no-45% copy (EIGENATTN_UPSTREAM=.../upstream_no45; only the Llama loop condition differs).

Inherited non-scientific shims (identical to OFFICIAL_EIGENATTN_REPRO_V1/run_official_eigenattn_shim.py):
 1. RotaryEmbedding inv_freq moved to the position-id device (Transformers 4.44 compatibility).
 2. load_dataset('wikitext','wikitext-2-raw-v1') redirected to the local parquet of the same split.
 3. Observation: every LlamaEigenAttnDecoderLayer the upstream search builds is recorded.
 4. cwd holds a verbatim copy of the cached HF exact_match metric (upstream lm_eval import).

Added for FDP-EB (documented adapter, upstream files untouched):
 5. Per-layer error budget (only if EIGENATTN_PER_LAYER_EB_JSON is set): upstream reads
    args.error_budget in every iteration of the per-layer loop. Right before upstream calls
    decompose_llama_layer(layer, inps, args, num_heads, i, ...) for layer i, args.error_budget is set to
    eb[i]; the loop, the 45% condition, the step-back and everything else run unchanged. After
    eigenattn() returns, args.error_budget is restored to the CLI value.
 6. Observation: the two torch.norm values of each upstream error expression are recorded so every
    evaluated candidate carries its exact error (same tensors, same dtype; values are only read).
"""
from __future__ import annotations

import atexit, hashlib, json, os, runpy, sys, time
from pathlib import Path

UPSTREAM = Path(os.environ.get("EIGENATTN_UPSTREAM", "/diskF/jiaqizhang/OFFICIAL_EIGENATTN_REPRO_V1/upstream"))
EVALS_OUT = os.environ.get("EIGENATTN_EVALS_OUT", "")
METRIC_CWD = Path("/diskF/jiaqizhang/OFFICIAL_EIGENATTN_REPRO_V1/metric_cwd")
TRACE_OUT = os.environ.get("EIGENATTN_TRACE_OUT", "")
PL_JSON = os.environ.get("EIGENATTN_PER_LAYER_EB_JSON", "")
WIKI_DIR = Path("/home/bai-project/jiaqizhang/wikitext2")
SHA_FILES = ["main_eigen_attn.py", "decompose/eigen_attn.py", "decompose/eigen_attn_utils.py",
             "models/modelling_llama_eigen_attn.py", "models/decompose_modules.py", "models/LMClass.py", "datautils.py"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


import torch  # noqa: E402
import datasets  # noqa: E402
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding  # noqa: E402

_rot_fwd = LlamaRotaryEmbedding.forward


def _rot_device_compat(self, x, position_ids):
    if hasattr(self, "inv_freq") and torch.is_tensor(self.inv_freq) and self.inv_freq.device != position_ids.device:
        self.inv_freq = self.inv_freq.to(position_ids.device)
    return _rot_fwd(self, x, position_ids)


LlamaRotaryEmbedding.forward = _rot_device_compat
_orig_load_dataset = datasets.load_dataset
REDIRECTS = []


def _load_dataset_offline(path, name=None, *args, **kwargs):
    if path == "wikitext" and name == "wikitext-2-raw-v1":
        split = kwargs.get("split")
        local = WIKI_DIR / f"{split}-00000-of-00001.parquet"
        REDIRECTS.append({"call": [path, name, split], "local": str(local), "sha256": sha(local)})
        return _orig_load_dataset("parquet", data_files=str(local), split="train")
    return _orig_load_dataset(path, name, *args, **kwargs)


datasets.load_dataset = _load_dataset_offline
sys.path.insert(0, str(UPSTREAM))
os.chdir(str(METRIC_CWD))
import models.decompose_modules as dm  # noqa: E402
import decompose.eigen_attn as ea  # noqa: E402

TRACE = []
CUR = {"layer": None}
_orig_init = dm.LlamaEigenAttnDecoderLayer.__init__


def _traced_init(self, ori_layer, args, basis_kq, rank_kq, basis_v, rank_v, config, layer_idx):
    _orig_init(self, ori_layer, args, basis_kq, rank_kq, basis_v, rank_v, config, layer_idx)
    TRACE.append({"layer": int(layer_idx), "rank_kq": int(rank_kq), "rank_v": int(rank_v),
                  "threshold_after": float(args.eigen_attn_params["threshold"]), "error_budget_in_effect": float(args.error_budget),
                  "t": time.time()})


dm.LlamaEigenAttnDecoderLayer.__init__ = _traced_init
ea.LlamaEigenAttnDecoderLayer = dm.LlamaEigenAttnDecoderLayer

PL_EB = None
if PL_JSON:
    PL_EB = [float(x) for x in json.loads(Path(PL_JSON).read_text())["error_budget_by_layer"]]
    assert len(PL_EB) == 32
_orig_dll = ea.decompose_llama_layer


EVALS = {}


def _dll(layer, inps, args, num_heads, i, *rest, **kw):
    CUR["layer"] = int(i)
    if PL_EB is not None:
        args.error_budget = PL_EB[int(i)]
    out = _orig_dll(layer, inps, args, num_heads, i, *rest, **kw)
    if EVALS_OUT:                                  # observation only: eigenvalues returned by the upstream decomposition
        _, ekq, _, ev = out
        EVALS[int(i)] = {"eval_kq": ekq.detach().float().cpu(), "eval_v": ev.detach().float().cpu()}
    return out


ea.decompose_llama_layer = _dll
_orig_eigenattn = ea.eigenattn


def _eigenattn(lm, args, dataloader, logger=None):
    cli_eb = args.error_budget
    try:
        return _orig_eigenattn(lm, args, dataloader, logger)
    finally:
        args.error_budget = cli_eb


ea.eigenattn = _eigenattn
_NORMS = []


class _TorchNormRecorder:
    """Module proxy for decompose.eigen_attn's global `torch`: forwards everything, records norm values."""
    def __getattr__(self, k):
        return getattr(torch, k)

    def norm(self, *a, **k):
        r = torch.norm(*a, **k)
        _NORMS.append(r)
        if len(_NORMS) == 2:
            n1, n2 = _NORMS
            e = n1 / n2                       # same expression/dtype as upstream
            if TRACE and TRACE[-1]["layer"] == CUR["layer"]:
                TRACE[-1]["err"] = float(e.item()); TRACE[-1]["err_dtype"] = str(e.dtype)
            _NORMS.clear()
        return r


ea.torch = _TorchNormRecorder()

MANIFEST = {"argv": sys.argv[1:], "upstream": str(UPSTREAM), "upstream_sha256": {f: sha(UPSTREAM / f) for f in SHA_FILES},
            "shim_sha256": sha(__file__), "torch": torch.__version__, "datasets": datasets.__version__,
            "per_layer_eb_json": PL_JSON or None, "evals_out": EVALS_OUT or None, "per_layer_eb": PL_EB, "started": time.time()}
try:
    import transformers
    MANIFEST["transformers"] = transformers.__version__
except Exception:
    pass


def _dump():
    if EVALS_OUT and EVALS:
        torch.save(EVALS, EVALS_OUT)
    if TRACE_OUT:
        MANIFEST["ended"] = time.time()
        MANIFEST["dataset_redirects"] = REDIRECTS
        MANIFEST["candidate_trace"] = TRACE
        Path(TRACE_OUT).write_text(json.dumps(MANIFEST, indent=1))


atexit.register(_dump)
sys.argv = [str(UPSTREAM / "main_eigen_attn.py")] + sys.argv[1:]
runpy.run_path(str(UPSTREAM / "main_eigen_attn.py"), run_name="__main__")
