#!/usr/bin/env python3
"""CPU-only schedules (no evaluation data, no loss probes):
 fdp  : FDP-guided exchange. q = frozen d1/d2/d5 profile of that operating point's Uniform model. Layer 0 excluded (FDP layer-0 rule).
        Receivers = layers by q descending, donors = layers by q ascending; step i pairs the i-th receiver with the i-th donor
        (skipping a layer already used or with no legal path). Receiver path = its lower-rank path below 4096 (tie -> QK);
        donor path = its higher-rank path above 64 (tie -> V). K steps of one 32-rank quantum; B_live is exactly conserved.
 rand : unrestricted random exchange. Seeded; K steps; donor unit uniform over all paths with rank > 64, receiver uniform over all
        paths with rank < 4096; distinct layers; every path used at most once.
K = number of exchanges accepted by the functional exchange at the same operating point (same amount of movement).
usage: l2x_plan.py fdp OP K NAME | rand OP K SEED NAME"""
import pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import l2x_lib as X


def q_of(op):
    p = X.PROFILE068 if op == "e068" else X.ROOT / "profiles" / op / "run" / "LLAMA2_CLEAN_REDESIGN_D1_D2_D5_PROFILE_V1.json"
    d = X.rjson(p)
    if d["compressed_comparator"]["rank_schedule"] != X.uniform_pairs(op): raise SystemExit("profile comparator != Uniform schedule")
    return [float(x) for x in d["q"]], str(p), X.sha_file(p)


def fdp(op, K):
    pairs = X.uniform_pairs(op); B0 = X.B_of(pairs); q, qp, qs = q_of(op)
    L = [l for l in range(1, 32)]
    R = sorted(L, key=lambda l: (-q[l], l)); D = sorted(L, key=lambda l: (q[l], l))
    used, moves, ri, di = set(), [], 0, 0
    while len(moves) < K:
        while ri < len(R) and (R[ri] in used or min(pairs[R[ri]]) >= X.RMAX): ri += 1
        while di < len(D) and (D[di] in used or max(pairs[D[di]]) <= X.rmin(op)): di += 1
        if ri >= len(R) or di >= len(D): break
        r, d = R[ri], D[di]
        if r == d: di += 1; continue
        rp = 0 if (pairs[r][0] <= pairs[r][1] and pairs[r][0] < X.RMAX) else 1
        if pairs[r][rp] >= X.RMAX: rp = 1 - rp
        dp = 1 if (pairs[d][1] >= pairs[d][0] and pairs[d][1] > X.rmin(op)) else 0
        if pairs[d][dp] <= X.rmin(op): dp = 1 - dp
        pairs[r][rp] += X.QUANTUM; pairs[d][dp] -= X.QUANTUM; used |= {r, d}
        moves.append({"receiver": f"L{r:02d}_{'QK' if rp == 0 else 'V'}", "donor": f"L{d:02d}_{'QK' if dp == 0 else 'V'}", "q_receiver": q[r], "q_donor": q[d]})
    assert X.B_of(pairs) == B0 and X.legal(pairs, X.rmin(op)) and len(moves) == K
    return pairs, {"method": "FDP-guided exchange (q only, open loop)", "q_profile": qp, "q_profile_sha256": qs, "moves": moves}


def rand(op, K, seed):
    pairs = X.uniform_pairs(op); B0 = X.B_of(pairs); rng = np.random.Generator(np.random.PCG64(seed))
    units = [(l, p) for l in range(32) for p in (0, 1)]; used, moves = set(), []
    while len(moves) < K:
        don = [u for u in units if u not in used and pairs[u[0]][u[1]] > X.rmin(op)]
        d = don[int(rng.integers(len(don)))]
        rec = [u for u in units if u not in used and u[0] != d[0] and pairs[u[0]][u[1]] < X.RMAX]
        r = rec[int(rng.integers(len(rec)))]
        pairs[d[0]][d[1]] -= X.QUANTUM; pairs[r[0]][r[1]] += X.QUANTUM; used |= {d, r}
        moves.append({"donor": f"L{d[0]:02d}_{'QK' if d[1] == 0 else 'V'}", "receiver": f"L{r[0]:02d}_{'QK' if r[1] == 0 else 'V'}"})
    assert X.B_of(pairs) == B0 and X.legal(pairs, X.rmin(op))
    return pairs, {"method": "unrestricted random exchange", "seed": seed, "moves": moves}


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "fdp": pairs, meta = fdp(a[2], int(a[3])); name = a[4]
    else: pairs, meta = rand(a[2], int(a[3]), int(a[4])); name = a[5]
    p = X.ROOT / "schedules" / name / "SCHEDULE.json"
    if p.exists(): raise SystemExit(f"exists {p}")
    meta.update({"op": a[2], "K": int(a[3]), "rank_pairs": pairs, "B_live": X.B_of(pairs), "rank_sha": X.sha_json(pairs),
                 "uniform_rank_sha": X.sha_json(X.uniform_pairs(a[2])), "evaluation_data_used": False, "script_sha256": X.sha_file(__file__), "at": X.now()})
    X.wjson(p, meta); print(name, meta["B_live"], [m["receiver"] + "<-" + m["donor"] for m in meta["moves"]])
