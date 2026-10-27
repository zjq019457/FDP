import json
import torch
import numpy as np

rep = json.load(open('/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb045/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb_report.json'))

eb_base = 0.045
for li in [0, 1, 2, 16, 31]:
    t = rep["layers"][li]["runs"][0]["threshold_trace"]
    # find base step
    s0 = 0
    for i, s in enumerate(t):
        if s["err"] <= eb_base and s["threshold"] >= 0.30:
            s0 = i
        else:
            break
    s1 = min(len(t) - 1, s0 + 1)
    base_s = t[s0]
    plus_s = t[s1]
    d_eb = plus_s["err"] - base_s["err"]
    print(f"L{li:2d}: Base step {s0} (err={base_s['err']:.4f}, rkq={base_s['rank_kq']}, rv={base_s['rank_v']}) -> Step {s1} (err={plus_s['err']:.4f}, rkq={plus_s['rank_kq']}, rv={plus_s['rank_v']}) | d_eb={d_eb:.4f}")
