#!/usr/bin/env python3
"""Pre-registration for FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1, written before any quality evaluation.
Uses only ranks/bytes (CPU replay). No PPL/ACC/LongBench value is read."""
import datetime, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import c45lib as L

ROOT = Path("/diskF/jiaqizhang/FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1")
CAMP = Path("/diskF/jiaqizhang/fdp125_final_campaign_v1/results/llama")
lat = L.uniform45_lattice()
proxy = {}
for t, e in {"light": 0.045, "medium": 0.060, "aggressive": 0.090}.items():
    v = list(json.loads((CAMP / t / "fdp125/provenance.json").read_text())["eb_vector"])
    for be in ([e] if t != "aggressive" else [0.090, 0.080, 0.075]):
        vv = [x * be / e for x in v]; vv[0] = be          # same normalised shape, rescaled to base_e (proxy only)
        rf, sf = L.select(vv, True); rp, sp = L.select([be] * 32, True)
        bf = L.B_of(rf); near = min(lat, key=lambda kv: abs(kv[0] - bf))
        proxy[f"{t}@{be}"] = {"fdp45_B": bf, "parent45_B": L.B_of(rp), "fdp_layers_differing_from_parent": sum(a != b for a, b in zip(rf, rp)),
                              "fdp_stops": L.stopdist(sf), "uniform45_nearest_gap_pct": 100 * (near[0] - bf) / bf,
                              "fdp45_minus_uniform45_min_pct": 100 * (bf - lat[0][0]) / lat[0][0]}
pre = {
 "experiment": "FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1",
 "written_at": datetime.datetime.now().astimezone().isoformat(),
 "quality_scores_read": False,
 "protocol": {"model": "/diskD/jiaqizhang/Llama-3 (Llama-3-8B base)", "calibration": "WikiText-2 train, 512x2048, seed 2 (frozen builder sampler)",
              "avg_dim": 32, "dtype": "bf16", "builder": "frozen offline builder sha e2329411... + C45 adapter (official 45% loop condition only)",
              "error": "frozen builder relative error", "threshold_step": 0.02, "threshold_floor": 0.30, "rank_lattice": "multiples of 8 (num_kv_heads), frozen finalize",
              "fdp": {"signal": "step1_diagnose_llama Dense vs Uniform+45 parent", "components": ["d1", "d2", "d5"], "weights": [1, 1, 0, 0, 1], "lambda": 1.2,
                      "normalisation": "mean-normalised to base_e", "layer0": "rho=0.75 geometric then builder --freeze_layer0_eb at base_e (frozen)"},
              "materializer": "main_eigen_attn.py --compress_only, avg_dim 4096, nsamples 64, c4, bf16 (frozen)",
              "eval": {"ppl": "ppl_instrumented.py sha ffe4e52e (141 Wiki / 256 C4 windows, bf16)", "zs": "zero_shot_instrumented.py sha aa2e2101, 5 tasks, bf16, batch 8",
                       "longbench": "QA-6 frozen manifest (1150 samples) wave2_longbench_eigen.py sha 37ca6cf7, max_len 8192, Aggressive only"}},
 "official_45_rule": "upstream 8bd0d3fa decompose/eigen_attn.py min_thresh=0.45: evaluate candidate, continue while err<eb and rank_kq>460.8 and rank_v>460.8; final = last accepted candidate (step-back). No rank clamp.",
 "tiers_base_e": {"light": 0.045, "medium": 0.060, "aggressive": 0.090},
 "resource_rule": "FDP+45 B is determined by its schedule; Uniform+45 EB = Uniform+45 lattice point minimising |B_u - B_fdp|. |gap|<=0.15% MATCHED, <=1% NEAR_MATCHED, else NOT_SAME_RESOURCE (no same-resource claim).",
 "aggressive_degeneracy_rule": {"definition": "DEGENERATE if the real FDP+45 ranks differ from the Uniform+45 parent ranks in <=4 layers, OR no Uniform+45 lattice point lies within 1% of B_fdp.",
                                "alternates_in_order": [0.080, 0.075], "procedure": "if DEGENERATE at base_e 0.090, rebuild parent/profiling at the next alternate base_e; choice uses ranks/bytes only."},
 "preregistered_optional_ablation": {"NoD2": {"weights": [1, 0, 0, 0, 1], "run_if": "main pipeline complete and GPU/disk available", "adoption_never_based_on_scores": True}},
 "optional_control": {"spectral_tail_same_resource": "only if resources allow; never used to select resource points"},
 "pairing": "per-window PPL NLL and per-item ACC kept; paired bootstrap FDP vs Uniform within tier",
 "proxy_resource_audit_using_historical_eb_shapes": proxy,
}
p = ROOT / "PREREGISTRATION.json"
if p.exists():
    raise SystemExit("PREREGISTRATION.json already exists; not overwriting")
p.write_text(json.dumps(pre, indent=1))
print(hashlib.sha256(p.read_bytes()).hexdigest())
print(json.dumps(proxy, indent=1))
