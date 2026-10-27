#!/usr/bin/env python3
"""Official PaLu zero-shot wrapper; hub-only unused metric is guarded, not substituted."""
import os, runpy, sys, types
from pathlib import Path
ROOT=Path("/diskE/jiaqizhang/PALU_OFFICIAL_TABLE1_REPRO_V1")
OFFICIAL=Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/Palu")
# lm-eval 0.4.2 eagerly downloads exact_match at import. The six pinned tasks below
# declare only acc/acc_norm. If this unused metric is ever requested, fail closed.
import evaluate
real_load=evaluate.load
guard={"exact_match_loads":0,"exact_match_compute":0}
class _UnusedExactMatch:
    def compute(self, **kwargs):
        guard["exact_match_compute"]+=1
        raise RuntimeError("exact_match is outside this six-task Table-1 protocol")
def _offline_metric_guard(path,*args,**kwargs):
    if path=="exact_match":
        guard["exact_match_loads"]+=1
        print("UNUSED_METRIC_GUARD exact_match; selected tasks use acc/acc_norm",flush=True)
        return _UnusedExactMatch()
    return real_load(path,*args,**kwargs)
evaluate.load=_offline_metric_guard
had={"calls":0}
def _forbidden_hadamard(*args,**kwargs):
    had["calls"]+=1
    raise RuntimeError("Hadamard was unexpectedly invoked in fp16 Dense zero-shot")
hshim=types.ModuleType("fast_hadamard_transform")
hshim.hadamard_transform=_forbidden_hadamard
sys.modules.setdefault("fast_hadamard_transform",hshim)
sys.path.insert(0,str(OFFICIAL))
model=os.environ["PALU_REPRO_MODEL_ALIAS"]
sys.argv=[str(OFFICIAL/"run_lm_eval.py"),"--model_name_or_path",model,
          "--tasks","openbookqa,hellaswag,piqa,arc_easy,arc_challenge,winogrande"]
runpy.run_path(str(OFFICIAL/"run_lm_eval.py"),run_name="__main__")
print(f"UNUSED_EXACT_MATCH_GUARD loads={guard['exact_match_loads']} compute_calls={guard['exact_match_compute']}",flush=True)
print(f"HADAMARD_GUARD_CALLS={had['calls']}",flush=True)
if guard["exact_match_compute"] or had["calls"]:
    raise RuntimeError("a guarded, non-protocol operation was invoked")
