#!/usr/bin/env python3
import os, json, math, torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

MODEL_DIR = os.getenv("MODEL_DIR", "/diskD/eigen_outputs_opt30b_r448/final_model")
TOK_DIR   = os.getenv("TOK_DIR",   "/diskE/opt-30b")
BASE_DIR  = os.getenv("BASE_DIR",  "/diskE/opt-30b")
BASIS_PT  = os.getenv("BASIS_PT",  "qkv_outputs/eigen_basis.pt")
LOW_RANK_JSON = os.getenv("LOW_RANK_JSON", "./low_rank_config_layerwise_448_opt30b.json")

MODE = os.getenv("REBUILD_MODE", "right").lower()  # right | left
DEVICE = "cuda"
DTYPE = torch.float16
TEXT = "Hello world. This is a quick sanity check."

def load_sd(model_dir):
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    paths = sorted(set(idx["weight_map"].values()))
    sd = {}
    for fn in paths:
        sd.update(load_file(os.path.join(model_dir, fn), device="cpu"))
    return sd

@torch.no_grad()
def ppl(model, tok, text):
    x = tok(text, return_tensors="pt").to(DEVICE)
    out = model(**x, labels=x["input_ids"])
    loss = float(out.loss.detach().float().item())
    return loss, math.exp(min(loss, 20.0))

def main():
    # configs
    low_rank = json.load(open(LOW_RANK_JSON))
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    L = cfg.num_hidden_layers
    d = cfg.hidden_size
    r_v = int(low_rank[0][1])

    # load basis dict (your format)
    basis = torch.load(BASIS_PT, map_location="cpu")
    # detect prefix
    sample_key = None
    for k in basis.keys():
        if k.endswith(".decoder.layers.0.self_attn.v_proj_basis"):
            sample_key = k
            break
    if sample_key is None:
        # fall back to any layer
        for k in basis.keys():
            if k.endswith(".self_attn.v_proj_basis"):
                sample_key = k
                break
    if sample_key is None:
        raise RuntimeError("Cannot find *.self_attn.v_proj_basis in eigen_basis.pt")

    prefix = sample_key.split(".decoder.layers.")[0]
    print("[Basis] prefix =", prefix)

    V_list = []
    for i in range(L):
        key = f"{prefix}.decoder.layers.{i}.self_attn.v_proj_basis"
        V = basis[key].float()   # [d, r]
        assert V.shape == (d, r_v), (i, V.shape)
        V_list.append(V)

    # load baseline OPT (cpu)
    print("[Baseline] loading:", BASE_DIR)
    base = AutoModelForCausalLM.from_pretrained(BASE_DIR, torch_dtype=torch.float16, device_map={"": "cpu"})
    base.eval()

    # build compressed model + load its sd
    sd = load_sd(MODEL_DIR)
    m = OPTForCausalLM_EigenAttn(cfg, low_rank_config=low_rank).to(DEVICE, dtype=DTYPE)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print("[Load] missing", len(missing), "unexpected", len(unexpected))
    m.tie_weights()
    m.eval()

    # rebuild v_proj_low + out_proj_up in-place
    for i in range(L):
        Wv = base.model.decoder.layers[i].self_attn.v_proj.weight.float()   # [d,d]
        bv = base.model.decoder.layers[i].self_attn.v_proj.bias.float()
        Wo = base.model.decoder.layers[i].self_attn.out_proj.weight.float()
        bo = base.model.decoder.layers[i].self_attn.out_proj.bias.float()
        V  = V_list[i]  # [d,r]

        # out_proj_up always Wo @ V
        out_w = Wo.matmul(V).contiguous()  # [d,r]
        out_b = bo.contiguous()

        # v_proj_low depends on mode; v_proj_low is Linear(d -> r) so weight is [r,d]
        if MODE == "right":
            # v_low = (full_value) @ V  where full_value = x @ Wv^T
            # => v_low = x @ (Wv^T @ V)   so weight = (Wv^T @ V)^T = V^T @ Wv
            v_w = (V.t().matmul(Wv)).contiguous()  # [r,d]
            v_b = (V.t().matmul(bv)).contiguous()  # [r]
        elif MODE == "left":
            # alternative: v_low = x @ (Wv @ V)^T  (often used if you treat V as input-basis)
            v_w = (Wv.matmul(V)).t().contiguous()  # [r,d]
            v_b = (bv.matmul(V)).contiguous()      # [r]
        else:
            raise ValueError("REBUILD_MODE must be right or left")

        layer = m.model.decoder.layers[i].self_attn
        layer.v_proj_low.weight.data.copy_(v_w.to(layer.v_proj_low.weight.dtype))
        layer.v_proj_low.bias.data.copy_(v_b.to(layer.v_proj_low.bias.dtype))

        layer.out_proj_up.weight.data.copy_(out_w.to(layer.out_proj_up.weight.dtype))
        layer.out_proj_up.bias.data.copy_(out_b.to(layer.out_proj_up.bias.dtype))

    loss, pplv = ppl(m, AutoTokenizer.from_pretrained(TOK_DIR, use_fast=True), TEXT)
    print(f"[Result] REBUILD_MODE={MODE} loss={loss:.6f} ppl={pplv:.3f}")

if __name__ == "__main__":
    main()
