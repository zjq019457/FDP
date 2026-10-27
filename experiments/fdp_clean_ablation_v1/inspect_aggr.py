import json

with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/authority/FDP_HIST_AUTHORITY.json") as f:
    auth = json.load(f)

aggr = auth["historical_ranks"]["aggressive"]
for k, v in aggr.items():
    if k != "ranks":
        print(f"{k}: {v}")

print("\nSample aggressive ranks:")
print("Layer 0:", aggr["ranks"][0])
print("Layer 1:", aggr["ranks"][1])
print("Layer 31:", aggr["ranks"][31])

# Also check medium and light ranks
med = auth["historical_ranks"]["medium"]
print("\nMedium total_rank:", sum(sum(r) for r in med["ranks"]))
print("Aggressive total_rank:", sum(sum(r) for r in aggr["ranks"]))
