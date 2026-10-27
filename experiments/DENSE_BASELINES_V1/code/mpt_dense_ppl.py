#!/usr/bin/env python3
"""MPT-7B dense (uncompressed) WikiText-2 / C4 PPL under the frozen MPT evaluation binding.
Reuses canonical_ppl.py unchanged (window binding + score_dataset); only the model is the dense base model."""
import importlib.util, json, sys
from pathlib import Path
import torch

E = Path("/diskE/jiaqizhang/MPT7B_FDP_EB_CAMPAIGN_V1/formal_accuracy_20261007T202413Z")
OUT = Path("/diskF/jiaqizhang/DENSE_BASELINES_V1/mpt_dense_ppl")
s = importlib.util.spec_from_file_location("canonical_ppl", str(E / "sources/canonical_ppl.py"))
ppl = importlib.util.module_from_spec(s); sys.modules["canonical_ppl"] = ppl; s.loader.exec_module(ppl)
binding = json.loads(ppl.BINDING.read_text())
windows = ppl.load_bound_windows(binding)
device = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.manual_seed(2)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(str(ppl.MODEL), local_files_only=True, trust_remote_code=False,
                                             torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.config.use_cache = False
model.eval().to(device)
OUT.mkdir(parents=True, exist_ok=True); (OUT / "wiki").mkdir(exist_ok=True); (OUT / "c4").mkdir(exist_ok=True)
wiki = ppl.score_dataset(model, windows["wikitext2"], "wikitext2", device, OUT / "wiki", "dense")
c4 = ppl.score_dataset(model, windows["c4"], "c4", device, OUT / "c4", "dense")
summary = {"schema": "MPT7B_DENSE_PPL_V1", "arm": "dense", "binding": str(ppl.BINDING), "model": str(ppl.MODEL),
           "wiki": wiki, "c4": c4, "J": 0.5 * (wiki["log_ppl"] + c4["log_ppl"])}
(OUT / "PPL_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"wiki": wiki["ppl"], "c4": c4["ppl"], "J": summary["J"]}), flush=True)
