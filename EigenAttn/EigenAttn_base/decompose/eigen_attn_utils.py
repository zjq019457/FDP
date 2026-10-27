import torch
from functools import partial
import numpy as np
import os
import scipy


def generate_basis_vectors_per_head(feat, threshold, num_heads, args, layer_id):
    U_kq = []
    S_kq = []
    
    if not torch.isfinite(feat).all():
        n_bad = int((~torch.isfinite(feat)).sum().item())
        raise RuntimeError(
            f"Non-finite values in calibration features before SVD (layer_id={layer_id}, "
            f"num_nonfinite={n_bad}). Common fixes: build basis with --dtype bf16 instead of fp16 "
            f"for deep OPT, run on a dedicated GPU (no sharing), reduce --seqlen, or lower GPU load."
        )

    X=torch.transpose(feat.reshape(feat.shape[0],num_heads,-1),0,1)
    dtype = X.dtype
    device = X.device
    X = X.to(torch.float32)


    for i in range(num_heads):
        # X[i] = torch.matmul(X[i], hadamard)
        u,s,v = torch.svd(X[i].t())
        u = u.to(device)

        s_val = s.cpu().numpy()
        s_val = s_val/np.sum(s_val)
        S_kq.append(torch.tensor(s_val))
        U_kq.append(u)
        
    X = X.to(dtype)
    
    return torch.stack(U_kq), torch.stack(S_kq)

def generate_basis_vectors_per_layer(feat, threshold, num_heads, args, layer_id):
    X = feat
    if not torch.isfinite(X).all():
        n_bad = int((~torch.isfinite(X)).sum().item())
        raise RuntimeError(
            f"Non-finite values in calibration features before SVD (layer_id={layer_id}, "
            f"num_nonfinite={n_bad}). Try --dtype bf16, dedicated GPU, or lower --seqlen."
        )
    X=torch.transpose(X.reshape(X.shape[0],num_heads,-1),0,1)
    X = X.reshape(-1, X.shape[-1])
    dtype = X.dtype
    device = X.device
    X = X.to(torch.float32)
    u,s,v = torch.svd(X.t())
    u = u.to(device)

    s_val = s.cpu().numpy()
    s_val = s_val/np.sum(s_val)
    
    S_v = s_val
    k=np.sum(np.cumsum(s_val)<threshold)+1
    U_v = u.to(dtype)
    X = X.to(dtype)
    
    return U_v, torch.tensor(S_v)

@torch.no_grad()
def get_kqv_mpt(layer, fp_inps, args, position_bias, attention_mask):
    
    avg_kqv = []
    kqv = {}
    
    avg_dim = args.eigen_attn_params['avg_dim_features']

    def forward_hook_kqv(m, x, y, name):
        if isinstance(y, tuple):
            y = y[0]
        
        if isinstance(x, tuple):
            x = x[0]
        # x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        y_size = y.shape
        # breakpoint()

        if name in kqv.keys():
            kqv[name] += [y.view(-1,y_size[-1])]
        else:
            kqv[name] = [y.view(-1,y_size[-1])]
        
        dim = len(kqv[name])
        
        if dim >= avg_dim:
            Y = torch.stack(kqv.pop(name)).mean(dim = 0)
            avg_kqv.append(Y)

    
    hooks = []
    for n, m in layer.named_modules():
        if (isinstance(m, torch.nn.Linear)) and  '.Wqkv' in n :
            hooks.append(
                m.register_forward_hook(
                    partial(forward_hook_kqv, name=n)))
        

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for j in range(args.nsamples):
        layer(fp_inps[j].unsqueeze(0), attention_mask = attention_mask, position_bias = position_bias)
    
    del kqv

    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()
    avg_kqv = torch.stack(avg_kqv)
    # avg_kqv = torch.cat(avg_kqv)
    if args.low_rank:
        rank_kq = layer.attn.rank_kq
        avg_q = avg_kqv[:,:,0:rank_kq]
        avg_k = avg_kqv[:,:,rank_kq:2*rank_kq]
        avg_v = avg_kqv[:,:,2*rank_kq:]
    else:
        avg_q, avg_k, avg_v  = avg_kqv.chunk(3, dim=-1)
    
    

    return avg_k, avg_q, avg_v

@torch.no_grad()
def get_kqv_llama(layer, fp_inps, args, attention_mask, position_ids):
    
    avg_k = []
    k = {}

    avg_q = []
    q = {}

    avg_v = []
    v = {}
    
    avg_dim = args.eigen_attn_params['avg_dim_features']

    def forward_hook_k(m, x, y, name):
        if isinstance(y, tuple):
            y = y[0]
        
        if isinstance(x, tuple):
            x = x[0]
        # x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        y_size = y.shape
        # breakpoint()

        if name in k.keys():
            k[name] += [y.view(-1,y_size[-1])]
        else:
            k[name] = [y.view(-1,y_size[-1])]
        
        dim = len(k[name])
        
        if dim >= avg_dim:
            Y = torch.stack(k.pop(name)).mean(dim = 0)
            avg_k.append(Y)
    
    def forward_hook_q(m, x, y, name):
        if isinstance(y, tuple):
            y = y[0]
        
        if isinstance(x, tuple):
            x = x[0]
        # x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        y_size = y.shape
        # breakpoint()

        if name in q.keys():
            q[name] += [y.view(-1,y_size[-1])]
        else:
            q[name] = [y.view(-1,y_size[-1])]
        
        dim = len(q[name])
        
        if dim >= avg_dim:
            Y = torch.stack(q.pop(name)).mean(dim = 0)
            avg_q.append(Y)
    
    def forward_hook_v(m, x, y, name):
        if isinstance(y, tuple):
            y = y[0]
        
        if isinstance(x, tuple):
            x = x[0]
        # x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        y_size = y.shape
        # breakpoint()

        if name in v.keys():
            v[name] += [y.view(-1,y_size[-1])]
        else:
            v[name] = [y.view(-1,y_size[-1])]
        
        dim = len(v[name])
        
        if dim >= avg_dim:
            Y = torch.stack(v.pop(name)).mean(dim = 0)
            avg_v.append(Y)

    
    hooks = []
    for n, m in layer.named_modules():
        if (isinstance(m, torch.nn.Linear)) and  '.k_proj' in n and 'up' not in n :
            hooks.append(
                m.register_forward_hook(
                    partial(forward_hook_k, name=n)))
            
        elif (isinstance(m, torch.nn.Linear)) and  '.q_proj' in n and 'up' not in n:
            hooks.append(
                m.register_forward_hook(
                    partial(forward_hook_q, name=n)))
        
        elif (isinstance(m, torch.nn.Linear)) and  '.v_proj' in n and 'up' not in n:
            hooks.append(
                m.register_forward_hook(
                    partial(forward_hook_v, name=n)))
        

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for j in range(args.nsamples):
        layer(fp_inps[j].unsqueeze(0), attention_mask = attention_mask, position_ids = position_ids)
    
    # del k, q, v

    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()
    avg_k = torch.stack(avg_k)
    avg_q = torch.stack(avg_q)
    avg_v = torch.stack(avg_v)

    return avg_k, avg_q, avg_v


@torch.no_grad()
def get_kqv_opt(layer, fp_inps, args, attention_mask=None):
    
    avg_k = []
    k = {}

    avg_q = []
    q = {}

    avg_v = []
    v = {}
    
    avg_dim = args.eigen_attn_params['avg_dim_features']

    def _to_cpu_feat(y: torch.Tensor) -> torch.Tensor:
        # 立刻把采样特征转到 CPU，避免在 GPU 上堆积大量中间激活。
        y2 = y.detach().reshape(-1, y.shape[-1])
        return y2.to(device="cpu", dtype=torch.float32)

    def forward_hook_k(m, x, y, name):
        if isinstance(y, tuple):
            y = y[0]
        
        if isinstance(x, tuple):
            x = x[0]
        # x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        # breakpoint()
        y_cpu = _to_cpu_feat(y)
        if name in k.keys():
            k[name] += [y_cpu]
        else:
            k[name] = [y_cpu]
        
        dim = len(k[name])
        
        if dim >= avg_dim:
            Y = torch.stack(k.pop(name)).mean(dim = 0)
            avg_k.append(Y)
    
    def forward_hook_q(m, x, y, name):
        if isinstance(y, tuple):
            y = y[0]
        
        if isinstance(x, tuple):
            x = x[0]
        # x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        # breakpoint()
        y_cpu = _to_cpu_feat(y)
        if name in q.keys():
            q[name] += [y_cpu]
        else:
            q[name] = [y_cpu]
        
        dim = len(q[name])
        
        if dim >= avg_dim:
            Y = torch.stack(q.pop(name)).mean(dim = 0)
            avg_q.append(Y)
    
    def forward_hook_v(m, x, y, name):
        if isinstance(y, tuple):
            y = y[0]
        
        if isinstance(x, tuple):
            x = x[0]
        # x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        # breakpoint()
        y_cpu = _to_cpu_feat(y)
        if name in v.keys():
            v[name] += [y_cpu]
        else:
            v[name] = [y_cpu]
        
        dim = len(v[name])
        
        if dim >= avg_dim:
            Y = torch.stack(v.pop(name)).mean(dim = 0)
            avg_v.append(Y)

    
    hooks = []
    for n, m in layer.named_modules():
        if (isinstance(m, torch.nn.Linear)) and  '.k_proj' in n :
            hooks.append(
                m.register_forward_hook(
                    partial(forward_hook_k, name=n)))
            
        elif (isinstance(m, torch.nn.Linear)) and  '.q_proj' in n :
            hooks.append(
                m.register_forward_hook(
                    partial(forward_hook_q, name=n)))
        
        elif (isinstance(m, torch.nn.Linear)) and  '.v_proj' in n and 'v_proj2' not in n :
            hooks.append(
                m.register_forward_hook(
                    partial(forward_hook_v, name=n)))
        

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p0 = next(layer.parameters())
    for j in range(args.nsamples):
        x = fp_inps[j].unsqueeze(0).to(
            device=p0.device, dtype=p0.dtype, non_blocking=True
        )
        am = attention_mask
        if am is not None and torch.is_tensor(am):
            am = am.to(device=x.device)
        layer(x, attention_mask=am, use_cache=False)
    
    del k, q, v

    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()

    avg_k = torch.stack(avg_k)
    avg_q = torch.stack(avg_q)
    avg_v = torch.stack(avg_v)

    return avg_k, avg_q, avg_v


@torch.no_grad()
def get_attn_output_opt(layer, fp_inps, args):
    
    avg_o = []
    o = {}

    
    avg_dim = args.eigen_attn_params['avg_dim_features']

    def forward_hook_o(m, x, y, name):
        if isinstance(y, tuple):
            y = y[0]
        
        if isinstance(x, tuple):
            x = x[0]
        # x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        y_size = y.shape

        if name in o.keys():
            o[name] += [y.view(-1,y_size[-1])]
        else:
            o[name] = [y.view(-1,y_size[-1])]
        
        dim = len(o[name])
        
        if dim >= avg_dim:
            Y = torch.stack(o.pop(name)).mean(dim = 0)
            avg_o.append(Y)
    
    
    hooks = []
    for n, m in layer.named_modules():
        if (isinstance(m, torch.nn.Linear)) and  '.out_proj' in n :
            hooks.append(
                m.register_forward_hook(
                    partial(forward_hook_o, name=n)))
            
        
        

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for j in range(args.nsamples):
        layer(fp_inps[j].unsqueeze(0))
    
    del o

    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()

    avg_o = torch.stack(avg_o)

    return avg_o


@torch.no_grad()
def get_attn_output_mpt(layer, fp_inps, args, position_bias, attention_mask):
    
    avg_o = []
    o = {}

    
    avg_dim = args.eigen_attn_params['avg_dim_features']

    def forward_hook_o(m, x, y, name):
        if isinstance(y, tuple):
            y = y[0]
        
        if isinstance(x, tuple):
            x = x[0]
        # x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        y_size = y.shape

        if name in o.keys():
            o[name] += [y.view(-1,y_size[-1])]
        else:
            o[name] = [y.view(-1,y_size[-1])]
        
        dim = len(o[name])
        
        if dim >= avg_dim:
            Y = torch.stack(o.pop(name)).mean(dim = 0)
            avg_o.append(Y)
    
    
    hooks = []
    for n, m in layer.named_modules():
        if (isinstance(m, torch.nn.Linear)) and  '.out_proj' in n :
            hooks.append(
                m.register_forward_hook(
                    partial(forward_hook_o, name=n)))
            
        
        

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for j in range(args.nsamples):
        layer(fp_inps[j].unsqueeze(0), position_bias= position_bias, attention_mask = attention_mask)
    
    del o

    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()

    avg_o = torch.stack(avg_o)

    return avg_o

def decompose_opt_layer(layer, fp_inps, args, num_heads, layer_id, attention_mask=None):

    # get calib features
    feat_k, feat_q, feat_v = get_kqv_opt(layer, fp_inps, args, attention_mask=attention_mask)


    basis_kq, eval_kq = generate_basis_vectors_per_head(torch.cat([feat_k.view(-1, feat_k.shape[-1]), feat_q.view(-1, feat_q.shape[-1])]), args.eigen_attn_params['threshold'], num_heads, args, layer_id)
    basis_v, eval_v = generate_basis_vectors_per_head(feat_v.view(-1, feat_v.shape[-1]), args.eigen_attn_params['threshold'], num_heads, args, layer_id)

    return basis_kq, eval_kq, basis_v, eval_v

def decompose_mpt_layer(layer, fp_inps, args, num_heads, layer_id, attention_mask, position_bias):

    # get calib features
    feat_k, feat_q, feat_v = get_kqv_mpt(layer, fp_inps, args, position_bias, attention_mask)

    basis_kq, eval_kq = generate_basis_vectors_per_head(torch.cat([feat_k.view(-1, feat_k.shape[-1]), feat_q.view(-1, feat_q.shape[-1])]), args.eigen_attn_params['threshold'], num_heads, args, layer_id)
    basis_v, eval_v = generate_basis_vectors_per_head(feat_v.view(-1, feat_v.shape[-1]), args.eigen_attn_params['threshold'], num_heads, args, layer_id)

    return basis_kq, eval_kq, basis_v, eval_v

def decompose_llama_layer(layer, fp_inps, args, num_heads, layer_id, attention_mask, position_ids):

    # get calib features
    feat_k, feat_q, feat_v = get_kqv_llama(layer, fp_inps, args, attention_mask, position_ids)
    basis_kq, eval_kq = generate_basis_vectors_per_layer(feat_k.view(-1, feat_k.shape[-1]), args.eigen_attn_params['threshold'], num_heads, args, layer_id)
    basis_v, eval_v = generate_basis_vectors_per_head(feat_v.view(-1, feat_v.shape[-1]), args.eigen_attn_params['threshold'], num_heads, args, layer_id)
    return basis_kq, eval_kq, basis_v, eval_v


