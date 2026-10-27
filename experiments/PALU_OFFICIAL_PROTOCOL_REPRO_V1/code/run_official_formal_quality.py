#!/usr/bin/env python3
import argparse, copy, gc, hashlib, json, math, random, sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path("/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1")
REPO=Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/Palu")
DATA=Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/data")
MODEL=Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/models/Llama-2-7b-hf")
SEQ, BYTES, N_BOOT = 2048, 2, 10000
sys.path.insert(0,str(ROOT/"code"))
import profile_official_native_phi as pr
import run_palu_official_repro_and_formal as official

def load(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def write(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def obj_sha(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def resource(sel):
    r=sum(sum(int(x) for x in v) for v in sel.values())
    return {"total_native_rank":r,"factor_bytes_fp16":r*(4096+512)*BYTES,
            "prefill_cache_bytes":r*SEQ*BYTES,"decode_cache_bytes":r*SEQ*BYTES}

def fisher(ratio):
    tag="%.1f"%ratio; p=ROOT/"results"/("fisher_"+tag.replace(".","p")+"_ratio_"+tag+"_fisher_uniform.json")
    d=load(p); return d,{k:[int(x) for x in v] for k,v in d["selection"].items()},p
def schedule(ratio,method):
    tag=("%.1f"%ratio).replace(".","p")
    if method=="fisher": return fisher(ratio)[1],fisher(ratio)[2]
    p=ROOT/"official_exchange_calibration"/("ratio_"+tag)/method/"final_schedule.json"
    d=load(p); return {k:[int(x) for x in v] for k,v in d["selection"].items()},p

def build(device,selection):
    pr.patch_offline_loaders()
    from utils import load_model_and_tokenizer
    from palu.model.modules.svd_linear import HeadwiseLowRankModule
    model,tok=load_model_and_tokenizer(str(MODEL)); model.to(torch.device(device))
    raw={}; parents={}
    for n,m in list(model.named_modules()):
        if ("k_proj" in n or "v_proj" in n) and isinstance(m,torch.nn.Linear):
            raw[n]=m; parents[n]=pr.parent_and_child(model,n)
    factors=torch.load(ROOT/"cache"/"official_whiten_factors_rank512.pt",map_location="cpu",weights_only=False)
    for n,ranks in selection.items():
        mod=pr.build_module(HeadwiseLowRankModule,raw[n],factors[n],ranks,device)
        p,c=parents[n]; setattr(p,c,mod)
    model.eval()
    return model,tok,raw,parents,factors,HeadwiseLowRankModule

@torch.no_grad()
def windows(model,tok):
    wiki_ids,c4_ids=official.load_protocol_ids(tok)
    device=next(model.parameters()).device
    def one(ids):
        out=[]
        n=ids.numel()//SEQ
        for i in range(n):
            batch=ids[:,i*SEQ:(i+1)*SEQ].to(device)
            pos=torch.arange(batch.shape[1],device=device).unsqueeze(0)
            hidden=model.model(batch,position_ids=pos)[0]
            logits=model.lm_head(hidden)
            loss=F.cross_entropy(logits[:,:-1,:].reshape(-1,logits.shape[-1]),
                                 batch[:,1:].reshape(-1),reduction="mean")
            x=float(loss.float().item())
            if not math.isfinite(x): raise FloatingPointError("non-finite formal loss")
            out.append(x)
        return out
    w,c=one(wiki_ids),one(c4_ids)
    return {"wiki":w,"c4":c,"wiki_nll":sum(w)/len(w),"c4_nll":sum(c)/len(c),
            "J":.5*(sum(w)/len(w)+sum(c)/len(c)),
            "wiki_ppl":math.exp(sum(w)/len(w)),"c4_ppl":math.exp(sum(c)/len(c)),
            "wiki_chunks":len(w),"c4_chunks":len(c),"wiki_tokens":len(w)*SEQ,"c4_tokens":len(c)*SEQ}

def ci(delta,seed):
    r=random.Random(seed); out=[]; nw,nc=len(delta["wiki"]),len(delta["c4"])
    for _ in range(N_BOOT):
        w=sum(delta["wiki"][r.randrange(nw)] for _ in range(nw))/nw
        c=sum(delta["c4"][r.randrange(nc)] for _ in range(nc))/nc; out.append(.5*(w+c))
    out.sort(); return out[int(.025*N_BOOT)],out[int(.975*N_BOOT)]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ratio",type=float,choices=[.3,.5,.7],required=True)
    ap.add_argument("--method",choices=["fdp_prior","fdp_native","random_0","random_1","random_2"],required=True)
    ap.add_argument("--device",default="cuda"); a=ap.parse_args()
    tag=("%.1f"%a.ratio).replace(".","p"); out=ROOT/"official_formal_quality"/("ratio_"+tag)/a.method
    out.mkdir(parents=True,exist_ok=True)
    if (out/"status.json").exists(): print(json.dumps({"status":"ALREADY_COMPLETE"})); return
    fisher_doc,fisher_sel,fisher_path=fisher(a.ratio); cand,cand_path=schedule(a.ratio,a.method)
    expected=resource(fisher_sel); got=resource(cand)
    if got!=expected: raise RuntimeError("resource mismatch vs official Fisher: %s != %s"%(got,expected))
    model,tok,raw,parents,factors,klass=build(a.device,fisher_sel)
    control=windows(model,tok)
    for n,ranks in cand.items():
        mod=pr.build_module(klass,raw[n],factors[n],ranks,a.device); p,c=parents[n]; setattr(p,c,mod)
    fixed=windows(model,tok)
    delta={"wiki":[b-a0 for a0,b in zip(control["wiki"],fixed["wiki"])],
           "c4":[b-a0 for a0,b in zip(control["c4"],fixed["c4"])]}
    dj=.5*(sum(delta["wiki"])/len(delta["wiki"])+sum(delta["c4"])/len(delta["c4"]))
    lo,hi=ci(delta,20261001+int(round(a.ratio*10)))
    protocol=ROOT/"PALU_OFFICIAL_PROTOCOL_LOCK.json"
    result={"status":"OFFICIAL_FORMAL_QUALITY_COMPLETE","ratio":a.ratio,"method":a.method,
            "official_baseline_method":"PaLu Automatic (Fisher)","control":control,"fixed":fixed,
            "delta_J":dj,"bootstrap_ci95":[lo,hi],"bootstrap_n":N_BOOT,
            "resource_control":expected,"resource_fixed":got,"resource_match":"EXACT",
            "resource_delta_vs_fisher":{"native_rank":got["total_native_rank"]-expected["total_native_rank"],
                                        "factor_bytes":got["factor_bytes_fp16"]-expected["factor_bytes_fp16"],
                                        "prefill_bytes":got["prefill_cache_bytes"]-expected["prefill_cache_bytes"],
                                        "decode_bytes":got["decode_cache_bytes"]-expected["decode_cache_bytes"]},
            "schedule_path":str(cand_path),"schedule_sha256":sha(cand_path),
            "fisher_result_path":str(fisher_path),"fisher_result_sha256":sha(fisher_path),
            "protocol_path":str(protocol),"protocol_sha256":sha(protocol),
            "q_sha256":load(protocol)["frozen_q_sha"],
            "action_space_sha256":load(ROOT/"PALU_OFFICIAL_NATIVE_ACTION_SPACE.json")["action_space_sha256"],
            "formal_quality_used_to_select":False}
    write(out/"result.json",result); write(out/"status.json",result)
    print(json.dumps({"status":result["status"],"ratio":a.ratio,"method":a.method,
                      "fisher_J":control["J"],"method_J":fixed["J"],"delta_J":dj,
                      "ci95":[lo,hi],"resource":got},indent=2),flush=True)

if __name__=="__main__": main()
