import json

with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles/step1_llama3_8b_eb090.json") as f:
    s090 = json.load(f)["per_layer"]

print("Keys in layer 0:", list(s090["0"].keys()))
print("Layer 1 in 090:", s090["1"])
