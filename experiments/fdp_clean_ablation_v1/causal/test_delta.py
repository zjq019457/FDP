import json

rep = json.load(open('/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb045/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb_report.json'))

def oracle_lookup(trace, eb, svd_floor=0.30):
    best_rk, best_rv = 1024, 1024
    for step in trace:
        th = step["threshold"]
        if step["err"] <= eb and th >= svd_floor:
            best_rk = step["rank_kq"]
            best_rv = step["rank_v"]
        else:
            break
    return best_rk, best_rv

delta = 0.030
eb_base = 0.045
num_changed = 0
for li in range(32):
    t = rep["layers"][li]["runs"][0]["threshold_trace"]
    r_base = oracle_lookup(t, eb_base)
    r_plus = oracle_lookup(t, eb_base + delta)
    changed = (r_base != r_plus)
    if changed: num_changed += 1
    dr = (r_plus[0] - r_base[0], r_plus[1] - r_base[1])
    print(f"L{li:2d}: Base={r_base} -> Plus={r_plus} (dr={dr}) | changed={changed}")

print(f"Total layers changed with delta={delta}: {num_changed}/32")
