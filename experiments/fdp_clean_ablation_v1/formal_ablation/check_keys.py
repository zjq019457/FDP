import torch

d = torch.load('/diskF/jiaqizhang/fdp_clean_ablation_v1/causal/causal_response_tensor.pt', map_location='cpu')
print("Type:", type(d))
if isinstance(d, dict):
    print("Keys:", list(d.keys()))
    for k in d.keys():
        v = d[k]
        if torch.is_tensor(v):
            print(f"Tensor {k}: shape {v.shape}, dtype {v.dtype}")
        else:
            print(f"Non-tensor {k}: type {type(v)}")
