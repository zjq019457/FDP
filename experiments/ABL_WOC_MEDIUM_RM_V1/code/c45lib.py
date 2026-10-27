"""Shared CPU helpers for FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1 (rank replay, lattice, frozen FDP schedule).

Replay is exact because the frozen builder propagates dense layer outputs (curves are EB-independent);
the CPU audit verified this against 657 frozen builder log points and 9 frozen rank sets.
"""
import hashlib, json, math, sys
from pathlib import Path

ENV = Path("/diskF/jiaqizhang/llama3_full_resource_robustness_v1/ultra_aggressive_v1/envelope_attempt2/ultra_aggressive_curve_full_q_attempt2_perlayer_eb_report.json")
EIGEN = Path("/home/bai-project/jiaqizhang/EigenAttn/EigenAttn")
FULL = 268435456
MT = 0.45 * 1024

_rep = json.loads(ENV.read_text())
CURVES = {l["layer"]: [(t["threshold"], t["rank_kq"], t["rank_v"], t["err"]) for t in l["runs"][-1]["threshold_trace"]] for l in _rep["layers"]}


def json_sha(o):
    return hashlib.sha256(json.dumps(o, separators=(",", ":")).encode()).hexdigest()


def B_of(r):
    return sum((a + b) * 4096 for a, b in r)


def select(ebv, rule45):
    ranks, stops = [], []
    for li in range(32):
        e = ebv[li]; pts = CURVES[li]; best = (pts[0][1], pts[0][2]); stop = "floor"
        for th, rk, rv, er in pts:
            ok45 = (rk > MT and rv > MT) if rule45 else True
            if er <= e and ok45:
                best = (rk, rv)
            if not (er < e and ok45):
                stop = "both" if (not er < e and not ok45) else ("error_budget" if not er < e else "min_thresh_45")
                break
        ranks.append([int(best[0]), int(best[1])]); stops.append(stop)
    return ranks, stops


def stopdist(s):
    return {k: s.count(k) for k in ["error_budget", "min_thresh_45", "both", "floor"]}


def uniform45_lattice():
    errs = sorted({x[3] for c in CURVES.values() for x in c})
    probe = [errs[0] / 2] + [(a + b) / 2 for a, b in zip(errs, errs[1:])] + [10.0]
    lat = {}
    for e in probe:
        r, s = select([e] * 32, True)
        lat.setdefault(B_of(r), (e, r, s))
    return sorted(lat.items())


def derive_eb(step1_path, e, weights):
    """Verbatim copy of the frozen worker schedule (fdp125_final_campaign_v1/llama_fdp125_tier_worker.py)."""
    sys.path.insert(0, str(EIGEN))
    from realloc_rank_from_fragility import fragility_vector, minmax
    per_layer = json.loads(Path(step1_path).read_text())["per_layer"]
    layers = sorted(int(x) for x in per_layer)
    raw, _ = fragility_vector(per_layer, layers, *weights, False)
    st = minmax(raw)
    eps = [e / (1.0 + 1.2 * x) for x in st]
    scale = e / (sum(eps) / len(eps))
    eps = [x * scale for x in eps]
    eps[0] = math.exp(0.75 * math.log(e) + 0.25 * math.log(eps[0]))
    return eps


def match_class(gap_pct):
    a = abs(gap_pct)
    return "MATCHED" if a <= 0.15 else ("NEAR_MATCHED" if a <= 1.0 else "NOT_SAME_RESOURCE")
