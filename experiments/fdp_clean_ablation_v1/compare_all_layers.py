import json

with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/authority/FDP_HIST_AUTHORITY.json") as f:
    auth = json.load(f)
ranks090 = auth["historical_ranks"]["aggressive"]["ranks"]

with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/fdp120_ranks.json") as f:
    ranks120 = json.load(f)["ranks"]

with open("/home/bai-project/jiaqizhang/step1_out/llama3_8b_frag_eb090_per_layer.json") as f:
    eb090_d = json.load(f)
eb090_layers = [l["eb_frag_eb090"] for l in eb090_d["layers"]]

with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/per_layer_eb_ultra_120.json") as f:
    eb120_layers = json.load(f)["error_budget_by_layer"]

print(f"{'L':<3} | {'EB090':<7} | {'R090 (kq,v,tot)':<16} | {'EB120':<7} | {'R120 (kq,v,tot)':<16} | Diff")
print("-" * 65)
tot090, tot120 = 0, 0
for i in range(32):
    r090_sum = sum(ranks090[i])
    r120_sum = sum(ranks120[i])
    tot090 += r090_sum
    tot120 += r120_sum
    diff = r120_sum - r090_sum
    print(f"{i:<3} | {eb090_layers[i]:<7.4f} | {ranks090[i][0]:<4} {ranks090[i][1]:<4} {r090_sum:<5} | {eb120_layers[i]:<7.4f} | {ranks120[i][0]:<4} {ranks120[i][1]:<4} {r120_sum:<5} | {diff:+d}")

print("-" * 65)
print(f"TOTAL: 090={tot090}, 120={tot120}, diff={tot120 - tot090}")
