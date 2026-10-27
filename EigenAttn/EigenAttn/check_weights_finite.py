import torch
from transformers import AutoModelForCausalLM

m="/diskE/opt-30b"
model = AutoModelForCausalLM.from_pretrained(m, local_files_only=True, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
bad=[]
for n,p in model.named_parameters():
    if p is None: 
        continue
    if not torch.isfinite(p).all():
        bad.append(n)
        print("BAD:", n, "dtype", p.dtype, "device", p.device)
print("bad_count:", len(bad))
