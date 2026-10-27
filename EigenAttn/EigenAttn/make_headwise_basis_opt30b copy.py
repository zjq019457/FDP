import os, torch, argparse
from transformers import AutoModelForCausalLM

torch.set_grad_enabled(False)

def top_evecs_from_rows(W_h: torch.Tensor, r_head: int):
    """
    W_h: [head_dim, hidden]  (float32)
    取 W_h W_h^T 的最大 r_head 个特征向量 => [head_dim, r_head]
    """
    C = W_h @ W_h.t()  # [hd,hd]
    # eigh 返回升序
    evals, evecs = torch.linalg.eigh(C)
    U = evecs[:, -r_head:]  # [hd, r]
    return U.contiguous()

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out_pt", required=True)
    ap.add_argument("--rank", type=int, default=5712)
    ap.add_argument("--keqk", type=int, default=1)
    args = ap.parse_args()

    print("Load BASE (cpu)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, local_files_only=True, torch_dtype=torch.float16,
        device_map=None, low_cpu_mem_usage=True
    ).cpu().eval()

    n_layers = base.config.num_hidden_layers
    d = base.config.hidden_size
    nh = base.config.num_attention_heads
    hd = d // nh
    assert args.rank % nh == 0
    r_head = args.rank // nh

    print(f"n_layers={n_layers} hidden={d} heads={nh} head_dim={hd} rank={args.rank} r_head={r_head} keqk={args.keqk}")

    out = {}

    for L in range(n_layers):
        attn = base.model.decoder.layers[L].self_attn

        # base weights: [d,d]
        Wq = attn.q_proj.weight.detach().float()
        Wk = attn.k_proj.weight.detach().float()
        Wv = attn.v_proj.weight.detach().float()

        # 组装 block-diag U: [d, rank]
        Uq = torch.zeros((d, args.rank), dtype=torch.float32)
        Uk = torch.zeros((d, args.rank), dtype=torch.float32)
        Uv = torch.zeros((d, args.rank), dtype=torch.float32)

        for h in range(nh):
            ro = slice(h*hd, (h+1)*hd)          # 输出 head 的 128 维块
            co = slice(h*r_head, (h+1)*r_head)  # 低维 head 的 102 维块

            Uq_h = top_evecs_from_rows(Wq[ro, :], r_head)  # [128,102]
            Uv_h = top_evecs_from_rows(Wv[ro, :], r_head)

            Uq[ro, co] = Uq_h
            Uv[ro, co] = Uv_h

            if args.keqk == 1:
                Uk[ro, co] = Uq_h
            else:
                Uk_h = top_evecs_from_rows(Wk[ro, :], r_head)
                Uk[ro, co] = Uk_h

        # 按你之前 eigen_basis 的 key 命名保存
        out[f"base_model.model.model.decoder.layers.{L}.self_attn.q_proj_basis"] = Uq
        out[f"base_model.model.model.decoder.layers.{L}.self_attn.k_proj_basis"] = Uk
        out[f"base_model.model.model.decoder.layers.{L}.self_attn.v_proj_basis"] = Uv

        if L % 8 == 0:
            print("built headwise basis layer", L)

    os.makedirs(os.path.dirname(args.out_pt) or ".", exist_ok=True)
    torch.save(out, args.out_pt)
    print("saved:", args.out_pt)

if __name__ == "__main__":
    main()
