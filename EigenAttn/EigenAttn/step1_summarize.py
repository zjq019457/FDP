"""
Side-by-side summary of step1_diagnose JSONs for EB024 vs EB008.

Outputs:
  * a per-layer table comparing core metrics
  * a banded summary (early/mid/late layers)
  * a hypothesis tag per layer:
      - "direction-dominated"  if score_only_AV_rel_mse >> value_only_Z_rel_mse
      - "value-dominated"      if value_only_Z_rel_mse >> score_only_AV_rel_mse
      - "balanced/capacity"    if both are large and comparable
      - "entropy-collapse"     if std_ratio_Z > 1.4 and dH(A) < -0.4
      - "drift-only"           if both are small (<<1) and Z_rel_mse <= 0.3
"""
import json, sys, argparse


def load(p):
    with open(p) as f:
        return json.load(f)


def tag(m):
    val = m["value_only_Z_rel_mse"]
    sco = m["score_only_AV_rel_mse"]
    Zr = m["Z_rel_mse"]
    std_r = m["std_ratio_Z"]
    dH = m["entropy_diff_A"]
    tags = []
    if Zr <= 0.30 and val <= 0.20 and sco <= 0.20:
        tags.append("ok/drift-only")
    if std_r > 1.40 and dH < -0.40:
        tags.append("entropy-collapse(sharp)")
    if std_r < 0.55:
        tags.append("Z-magnitude-shrink")
    if sco >= max(0.30, 1.7 * val):
        tags.append("direction-dominated")
    elif val >= max(0.30, 1.7 * sco):
        tags.append("value-dominated")
    elif val >= 0.30 and sco >= 0.30:
        tags.append("balanced(capacity)")
    if not tags:
        tags.append("mild")
    return ",".join(tags)


def per_layer_table(eb024, eb008):
    pl_a = eb024["per_layer"]
    pl_b = eb008["per_layer"]
    assert sorted(pl_a.keys(), key=int) == sorted(pl_b.keys(), key=int)
    layers = sorted(pl_a.keys(), key=int)
    print()
    print("=== Per-layer side-by-side: EB024 (a) vs EB008 (b) ===")
    print(
        f"{'L':>2}  "
        f"{'r24_kq':>6}/{'r24_v':<5}  {'r08_kq':>6}/{'r08_v':<5}  "
        f"{'a:Z_rel':>8}  {'b:Z_rel':>8}  "
        f"{'a:V|A':>7}  {'a:A|V':>7}  "
        f"{'a:dH':>6}  {'a:std':>6}  "
        f"{'a:tag':<35}"
    )
    for li in layers:
        a = pl_a[li]
        b = pl_b[li]
        print(
            f"{int(li):>2}  "
            f"{a['rank_kq']:>6}/{a['rank_v']:<5}  {b['rank_kq']:>6}/{b['rank_v']:<5}  "
            f"{a['Z_rel_mse']:>8.3f}  {b['Z_rel_mse']:>8.3f}  "
            f"{a['value_only_Z_rel_mse']:>7.3f}  {a['score_only_AV_rel_mse']:>7.3f}  "
            f"{a['entropy_diff_A']:>+6.2f}  {a['std_ratio_Z']:>6.2f}  "
            f"{tag(a):<35}"
        )


def banded(d, name, bands):
    pl = d["per_layer"]
    print()
    print(f"=== Banded summary [{name}] ===")
    cols = ["Z_rel_mse", "value_only_Z_rel_mse", "score_only_AV_rel_mse",
            "score_rel_mse_causal", "score_std_ratio_causal", "std_ratio_Z",
            "norm_ratio_Z", "entropy_diff_A"]
    hdr = f"{'band':<14}  {'#L':>3}  " + "  ".join(f"{c[:11]:>11}" for c in cols)
    print(hdr)
    for label, lo, hi in bands:
        keys = [k for k in pl.keys() if lo <= int(k) <= hi]
        means = []
        for c in cols:
            vals = [pl[k][c] for k in keys if c in pl[k]]
            means.append(sum(vals) / max(1, len(vals)))
        line = f"{label:<14}  {len(keys):>3}  " + "  ".join(f"{m:>11.4f}" for m in means)
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eb024", required=True)
    ap.add_argument("--eb008", required=True)
    args = ap.parse_args()
    a = load(args.eb024)
    b = load(args.eb008)

    print(f"=== EB024 summary ({args.eb024}) ===")
    print(json.dumps(a["summary"], indent=2))
    print()
    print(f"=== EB008 summary ({args.eb008}) ===")
    print(json.dumps(b["summary"], indent=2))

    bands = [
        ("L0",       0,  0),
        ("early1-7", 1,  7),
        ("crush8-15",8, 15),
        ("recov16-28",16,28),
        ("late29-44",29,44),
        ("tail45-47",45,47),
    ]
    banded(a, "EB024", bands)
    banded(b, "EB008", bands)
    per_layer_table(a, b)


if __name__ == "__main__":
    main()
