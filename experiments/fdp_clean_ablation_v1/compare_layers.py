import json

with open("/home/bai-project/jiaqizhang/step1_out/llama3_8b_frag_eb090_per_layer.json") as f:
    eb090_d = json.load(f)

eb090_layers = [l["eb_frag_eb090"] for l in eb090_d["layers"]]

with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/per_layer_eb_ultra_120.json") as f:
    eb120_d = json.load(f)

eb120_layers = eb120_d["error_budget_by_layer"]

with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/authority/FDP_HIST_AUTHORITY.json") as f:
    auth = json.load(f)
ranks090 = auth["historical_ranks"]["aggressive"]["ranks"]

with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/fdp120_ranks.json") as f:
    ranks120 = json.load(f)["ranks"]

print(f"{'Layer':<6} | {'EB 090':<10} | {'Ranks 090':<15} | {'EB 120':<10} | {'Ranks 120':<15}")
print("-" * 65)
for i in range(10):
    print(f"{i:<6} | {eb090_layers[i]:<10.4f} | {str(ranks090[i]):<15} | {eb120_layers[i]:<10.4f} | {str(ranks120[i]):<15}")
