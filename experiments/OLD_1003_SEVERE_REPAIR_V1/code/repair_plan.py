#!/usr/bin/env python3
"""OLD_1003_SEVERE_REPAIR_V1 - CPU-only, quality-blind frozen plan (user decision 2026-10-10: accept NEAR_MATCHED,
run R1-a, R2, R3). Points are exactly those listed in SEVERE_CONTROL_REPAIR_PLAN.md; this script re-derives them with
the audited full-trace definitions and refuses to write if any rank SHA / B differs from the documented values."""
import json
from pathlib import Path

OUT = Path("/diskF/jiaqizhang/OLD_1003_SEVERE_REPAIR_V1")
O3S = Path("/diskF/jiaqizhang/OLD_1003_SPECTRAL_TAIL_V1")
src = (O3S / "code/truncation_audit.py").read_text().split("A[\"NO_ENTROPY_full_trace_rescan\"] = rescan")[0]
G = {}; exec(src, G)
select, schedule_q, minmax, D, ARMS, B_of, jsha = (G[k] for k in ("select", "schedule_q", "minmax", "D", "ARMS", "B_of", "jsha"))
BF = G["BF"]
audit = json.loads((O3S / "TRUNCATED_CURVE_AUDIT.json").read_text())

uni = select([0.09305876924047846] * 32)
q_noz = minmax([sum(D[p][i] for p in ARMS["NO_Z"]) for i in range(32)])
noz = select(schedule_q(q_noz, 0.093027))
sp = audit["spectral_anchor_sensitivity"]["uniform_full_trace_nearest_plus_0p217"]
assert sp["anchor_rank_sha"] == jsha(uni)
spec = select(schedule_q(sp["q"], sp["selected"]["epsilon_base"]))
want = {"UNIFORM_FULLTRACE": ("823e850e1c7a97346069562efe17e7c0931faaaf635fc38e7f5c24fe2b58c1b2", 151617536),
        "NO_Z_FULLTRACE": ("4eadcda55f2e", 151552000), "SPECTRAL_REANCHORED": ("aaae98d86a8e", 151093248)}
arms = {"UNIFORM_FULLTRACE": {"ranks": uni, "definition": "single EB 0.09305876924047846 on the full trace (= Severe V2 Uniform ranks)", "acc": True},
        "NO_Z_FULLTRACE": {"ranks": noz, "definition": "10/3 schedule d2+d5, base 0.093027, full trace", "epsilon_base": 0.093027,
                           "eps_by_layer": schedule_q(q_noz, 0.093027), "acc": False},
        "SPECTRAL_REANCHORED": {"ranks": spec, "definition": "Spectral-tail anchored at UNIFORM_FULLTRACE ranks, 10/3 schedule, full trace",
                                "epsilon_base": sp["selected"]["epsilon_base"], "q": sp["q"], "q_sha": sp["q_sha"],
                                "eps_by_layer": schedule_q(sp["q"], sp["selected"]["epsilon_base"]), "acc": True}}
for k, a in arms.items():
    a.update({"rank_sha256": jsha(a["ranks"]), "B": B_of(a["ranks"]), "gap_pct_vs_full": 100 * (B_of(a["ranks"]) - BF) / BF,
              "min_rank": min(min(x) for x in a["ranks"]), "match_class": "NEAR_MATCHED (<=1%, user-accepted 2026-10-10)" if abs(B_of(a["ranks"]) - BF) / BF > 0.0015 else "MATCHED"})
    assert a["rank_sha256"].startswith(want[k][0]) and a["B"] == want[k][1], (k, a["rank_sha256"], a["B"])
plan = {"schema": "OLD_1003_SEVERE_REPAIR_PLAN_V1", "quality_scores_read": False, "full_B": BF, "full_rank_sha": G["pn"]["full_rank_sha"],
        "curves": "full trace (fdp_v2_fulltrace_scratch severe report), see OLD_1003_TRUNCATED_CURVE_AUDIT.md",
        "materialization": "fill_arm2.py exact-rank slice of the 10/3 master (sha 638d4748...) -> authoritative fill",
        "user_decision": "2026-10-10: accept NEAR_MATCHED; run R1-a, R2, R3 in parallel", "arms": arms}
OUT.mkdir(parents=True, exist_ok=True)
p = OUT / "REPAIR_PLAN.json"
assert not p.exists(), "refuse overwrite"
p.write_text(json.dumps(plan, indent=1)); p.chmod(0o444)
for k, a in arms.items(): print(k, a["rank_sha256"][:12], a["B"], round(a["gap_pct_vs_full"], 4), a["min_rank"], a["match_class"])
