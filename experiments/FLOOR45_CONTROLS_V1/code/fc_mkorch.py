#!/usr/bin/env python3
"""Generate FLOOR45_CONTROLS_V1/code/orchestrator.py from the frozen CONSTRAINED45 orchestrator:
same scheduler, builder, materializer and evaluators; only paths and the job graph change."""
import shutil
from pathlib import Path

SRC = Path("/diskF/jiaqizhang/FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1/code")
DST = Path("/diskF/jiaqizhang/FLOOR45_CONTROLS_V1/code")
s = (SRC / "orchestrator.py").read_text()
G = "# ----------------------------------------------------------------------------- pipeline graph"
S = "# ----------------------------------------------------------------------------- scheduler"
head, rest = s.split(G); _, tail = rest.split(S)
rep = [('ROOT = Path("/diskF/jiaqizhang/FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1")', 'ROOT = Path("/diskF/jiaqizhang/FLOOR45_CONTROLS_V1")'),
       ('BUILDER = ROOT / "code/build_eigen_basis_llama_error_budget_offline_c45.py"',
        'BUILDER = Path("/diskF/jiaqizhang/FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1/code/build_eigen_basis_llama_error_budget_offline_c45.py")'),
       ('"E": Path("/diskE/jiaqizhang/FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1_SCRATCH")', '"E": Path("/diskE/jiaqizhang/FLOOR45_CONTROLS_V1_SCRATCH")'),
       ('"F": Path("/diskF/jiaqizhang/FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1_SCRATCH")', '"F": Path("/diskF/jiaqizhang/FLOOR45_CONTROLS_V1_SCRATCH")'),
       ('"""FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1 GPU-safe dynamic scheduler.', '"""FLOOR45_CONTROLS_V1 GPU-safe dynamic scheduler (generated from the CONSTRAINED45 orchestrator by fc_mkorch.py).')]
for a, b in rep:
    assert head.count(a) == 1, a
    head = head.replace(a, b)
graph = G + '''
PLAN = rjson(ROOT / "PLAN.json")
for t, tp in PLAN["tiers"].items():
    u, sp = tp["uni_nofloor"], tp["tail45"]
    assert sp["selected"] and u["match_class_vs_uni45"] == "MATCHED" and sp["match_class"] == "MATCHED"
    def chk(ranks, stops, want, name):
        if ranks != want: raise GateFail(f"{name}: builder ranks != CPU replay prediction")
    # A. Uniform without the floor, matched to the floored Uniform
    mk_builder_job(f"uninf_basis_{t}", [], f"fc_uninf_{t}", lambda u=u: [u["eb"]] * 32, lambda u=u: u["eb"], 0.0,
                   lambda r, s, u=u, t=t: chk(r, s, u["ranks"], f"uninf_{t}"))
    mk_mat_job(f"uninf_mat_{t}", [f"uninf_basis_{t}"], f"uninf_basis_{t}", lambda u=u: u["eb"])
    mk_eval_jobs(t, "uniform_nofloor", f"uninf_mat_{t}")
    # user 2026-10-12: arm A needs PPL only (no zero-shot, no LongBench)
    for n in [n for n in JOBS if n.endswith(f"_{t}_uniform_nofloor") and n.split("_")[0] in ("zs", "lb5", "lbnqa")]:
        del JOBS[n]
    JOBS[f"cleanup_{t}_uniform_nofloor"].deps = [f"ppl_wikitext2_{t}_uniform_nofloor", f"ppl_c4_{t}_uniform_nofloor"]
    # B. Spectral-tail with the floor, matched to the floored FDP-EB
    mk_builder_job(f"tail45_basis_{t}", [], f"fc_tail45_{t}", lambda sp=sp: sp["eps_by_layer"], lambda sp=sp: sp["epsilon_base"], 0.45,
                   lambda r, s, sp=sp, t=t: chk(r, s, sp["ranks"], f"tail45_{t}"))
    mk_mat_job(f"tail45_mat_{t}", [f"tail45_basis_{t}"], f"tail45_basis_{t}", lambda sp=sp: sp["epsilon_base"])
    mk_eval_jobs(t, "spectral_tail45", f"tail45_mat_{t}")


'''
# IDREF / identity-gate block of the source graph reads frozen controls; not needed here.
out = head + graph + S + tail
DST.mkdir(parents=True, exist_ok=True)
(DST / "orchestrator.py").write_text(out)
shutil.copy2(SRC / "c45lib.py", DST / "c45lib.py")
print("ok", len(out))
