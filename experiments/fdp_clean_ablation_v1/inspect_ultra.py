import json

with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/per_layer_eb_ultra_120.json") as f:
    eb120 = json.load(f)

print("EB120 keys:", list(eb120.keys()) if isinstance(eb120, dict) else len(eb120))
if isinstance(eb120, dict):
    print("EB120 per_layer:", eb120.get("per_layer_eb", eb120))
else:
    print("EB120:", eb120[:5])

# Find historical EB090 json
import glob
print("Searching for eb090 / eb060 jsons:")
for p in glob.glob("/home/bai-project/jiaqizhang/EigenAttn/**/*.json", recursive=True):
    if "eb" in p.lower() and ("090" in p or "060" in p or "per_layer" in p):
        print(p)
