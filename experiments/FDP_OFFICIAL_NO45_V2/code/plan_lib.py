"""FDP_OFFICIAL_NO45_V2 planning library (CPU only, quality-blind).
Exact replay of the upstream Llama loop on recorded candidate curves; FDP / Spectral budget schedules; resource matching."""
import hashlib, json, math, sys
from pathlib import Path
import torch

N = Path("/diskF/jiaqizhang/FDP_OFFICIAL_NO45_V2")
V1 = Path("/diskF/jiaqizhang/FDP_OFFICIAL_PROTOCOL_V1")
sys.path.insert(0, "/home/bai-project/jiaqizhang/EigenAttn/EigenAttn")
from realloc_rank_from_fragility import fragility_vector, minmax  # noqa: E402

MT = 0.45 * 1024
BYTES_PER_RANK = 4096
OFFICIAL_EB = {"light": 0.045, "medium": 0.060, "aggressive": 0.090}


def jsha(o): return hashlib.sha256(json.dumps(o, separators=(",", ":")).encode()).hexdigest()
def B_of(r): return sum((a + b) * BYTES_PER_RANK for a, b in r)


def lt(err, dtype, eb):
    """Exactly the upstream comparison `error < args.error_budget` (0-d tensor of err_dtype vs python float)."""
    return bool(torch.tensor(err, dtype=getattr(torch, dtype.replace("torch.", ""))) < eb)


def load_trace_curves(trace_json):
    """Per layer list of candidates [rk, rv, err, dtype, threshold] in evaluation order (all recorded candidates)."""
    tr = json.loads(Path(trace_json).read_text())
    by = {}
    for c in tr["candidate_trace"]:
        by.setdefault(c["layer"], []).append(c)
    out = []
    for li in range(32):
        cs = by[li]
        # the last record of a layer is the post step-back final layer (no loop error of its own)
        out.append({"loop": [[c["rank_kq"], c["rank_v"], c["err"], c["err_dtype"], c["threshold_after"]] for c in cs[:-1]],
                    "final": [cs[-1]["rank_kq"], cs[-1]["rank_v"]]})
    return tr, out


def union_v1_curves():
    """Union (longest per layer) of the recorded official candidate curves of FDP_OFFICIAL_PROTOCOL_V1; asserts identical prefixes."""
    best = [None] * 32
    for rec in sorted((V1 / "records").glob("*/curves.json")):
        if not any(rec.parent.name.startswith(p) for p in ("parent_", "uni_", "uniplus_", "fdp_")):
            continue
        layers = json.loads(rec.read_text())
        for li, L in enumerate(layers):
            loop = [c[:4] for c in L["loop"]]
            if best[li] is None:
                best[li] = loop; continue
            a, b = best[li], loop
            m = min(len(a), len(b))
            assert all(a[k][:3] == b[k][:3] for k in range(m)), (rec, li)
            if len(b) > len(a): best[li] = b
    return [{"loop": [c + [None] for c in L]} for L in best]


def replay(curves, ebv, rule, complete=False):
    """rule: 'official' (45% stop) or 'no45' (threshold floor; requires complete curves to resolve the floor).
    Returns ranks (list) or None if some layer is not covered by the recorded candidates."""
    ranks = []
    for li, L in enumerate(curves):
        prev = (1024, 1024); done = False
        for rk, rv, err, dt, th in L["loop"]:
            ok = lt(err, dt, ebv[li]) and (rule == "no45" or (rk > MT and rv > MT))
            if not ok:
                ranks.append(list(prev)); done = True; break
            prev = (rk, rv)
        if not done:
            if complete:            # the recorded curve ends at the threshold floor: the loop stops there and steps back once
                ranks.append(list(L["loop"][-2][:2]) if len(L["loop"]) >= 2 else [1024, 1024])
            else:
                return None
    return ranks


def fdp_q(step1_json, weights=(1, 0, 0, 0, 1)):
    per = json.loads(Path(step1_json).read_text())["per_layer"]
    layers = sorted(int(x) for x in per)
    raw, _ = fragility_vector(per, layers, *weights, False)
    return minmax(raw)


def schedule(q, base, lam=1.2):
    eps = [base / (1.0 + lam * x) for x in q]
    m = sum(eps) / len(eps)
    eps = [base * x / m for x in eps]
    eps[0] = base
    return eps


def spectral_q(evals_pt, uniform_ranks):
    ev = torch.load(evals_pt, map_location="cpu")
    raw = []
    for li in range(32):
        k = ev[li]["eval_kq"].float().flatten(); v = ev[li]["eval_v"].float()
        if v.dim() == 1: v = v.unsqueeze(0)
        rk, rv = uniform_ranks[li]
        a = float(k[rk // 8:].sum() / k.sum()); b = float(v[:, rv // 8:].sum() / v.sum())
        raw.append(0.5 * a + 0.5 * b)
    return minmax(raw), raw


def f32_flip(err):
    """The upstream test is float32(err) < float32(eb) (verified: 0/10,000 mismatches); it passes iff eb exceeds the
    midpoint between float32(err) and the next float32 value (round-to-nearest)."""
    import numpy as np
    e = np.float32(err); up = np.nextafter(e, np.float32(np.inf))
    return (float(e) + float(up)) / 2.0


def fast_curves(curves):
    for L in curves:
        assert all(c[3] == "torch.float32" for c in L["loop"]), "non-float32 error dtype"
    return [[(c[0], c[1], f32_flip(c[2])) for c in L["loop"]] for L in curves]


def replay_fast(fc, ebv, rule, complete=True):
    ranks = []
    for li, L in enumerate(fc):
        prev = (1024, 1024); done = False
        for rk, rv, flip in L:
            if not (ebv[li] > flip and (rule == "no45" or (rk > MT and rv > MT))):
                ranks.append(list(prev)); done = True; break
            prev = (rk, rv)
        if not done:
            if not complete: return None
            ranks.append(list(L[-2][:2]) if len(L) >= 2 else [1024, 1024])
    return ranks


def match_exact(curves, q, target_B, prefer_base, lo, hi, complete=True, rule="no45"):
    """Exact enumeration of every distinct no45 schedule for base in (lo, hi): the schedule is linear in base
    (eps_l = base * k_l), so it can only change where base * k_l crosses a candidate's float32 flip point.
    One evaluation per interval between consecutive flip points; the chosen schedule is re-verified with the exact
    torch comparison (replay)."""
    fc = fast_curves(curves)
    k = schedule(q, 1.0)
    flips = sorted({f / k[li] for li, L in enumerate(fc) for (_, _, f) in L if lo < f / k[li] < hi})
    bounds = [lo] + flips + [hi]
    seen = {}
    for a, b in zip(bounds, bounds[1:]):
        if b <= a: continue
        mid = (a + b) / 2.0
        r = replay_fast(fc, schedule(q, mid), rule, complete)
        if r is None: continue
        h = jsha(r)
        cand = min(max(prefer_base, a + (b - a) * 1e-6), b - (b - a) * 1e-6)
        if h not in seen or abs(cand - prefer_base) < abs(seen[h][0] - prefer_base):
            seen[h] = (cand, B_of(r), r)
    cands = list(seen.values())
    el = [c for c in cands if abs(c[1] - target_B) <= 0.0015 * target_B]
    if el:
        el.sort(key=lambda c: (abs(c[1] - target_B), abs(c[0] - prefer_base))); cls = "MATCHED"; pick = el[0]
    else:
        nm = sorted([c for c in cands if c[1] <= target_B and target_B - c[1] <= 0.01 * target_B], key=lambda c: (target_B - c[1], abs(c[0] - prefer_base)))
        pick = nm[0] if nm else None; cls = "NEAR_MATCHED" if nm else "NO_MATCH"
    out = {"class": cls, "rule": rule, "method": "exact flip-point enumeration", "n_intervals": len(bounds) - 1, "n_distinct": len(cands), "n_eligible": len(el), "pick": None}
    if pick is not None:
        eps = schedule(q, pick[0])
        exact = replay(curves, eps, rule, complete)
        if exact != pick[2]: raise AssertionError("fast replay disagrees with exact torch replay")
        out["pick"] = {"base": pick[0], "B": pick[1], "gap_pct": 100 * (pick[1] - target_B) / target_B, "ranks": pick[2],
                       "rank_sha": jsha(pick[2]), "eps": eps, "min_rank": min(min(x) for x in pick[2])}
    return out


def match(curves, q, target_B, prefer_base, complete, lo=0.005, hi=0.5, step=1e-5):
    """Scan base; keep distinct schedules; apply the preregistered rule."""
    seen = {}
    uncovered = 0
    k = 0
    base = lo
    while base <= hi:
        eps = schedule(q, base)
        r = replay(curves, eps, "no45", complete)
        if r is None:
            uncovered += 1
        else:
            h = jsha(r)
            if h not in seen or abs(base - prefer_base) < abs(seen[h][0] - prefer_base):
                seen[h] = (base, B_of(r), r)
        k += 1; base = lo + k * step
    cands = list(seen.values())
    el = [c for c in cands if abs(c[1] - target_B) <= 0.0015 * target_B]
    if el:
        el.sort(key=lambda c: (abs(c[1] - target_B), abs(c[0] - prefer_base)))
        cls = "MATCHED"; pick = el[0]
    else:
        nm = [c for c in cands if c[1] <= target_B and target_B - c[1] <= 0.01 * target_B]
        nm.sort(key=lambda c: (target_B - c[1], abs(c[0] - prefer_base)))
        pick = nm[0] if nm else None; cls = "NEAR_MATCHED" if nm else "NO_MATCH"
    return {"class": cls, "n_distinct": len(cands), "n_eligible": len(el), "uncovered_bases": uncovered,
            "pick": None if pick is None else {"base": pick[0], "B": pick[1], "gap_pct": 100 * (pick[1] - target_B) / target_B,
                                               "ranks": pick[2], "rank_sha": jsha(pick[2]), "eps": schedule(q, pick[0])}}
