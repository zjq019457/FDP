#!/usr/bin/env python3
import argparse, copy, csv, gc, gzip, hashlib, json, math, random, sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path("/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1")
REPO = Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/Palu")
DATA = Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/data")
MODEL = Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/models/Llama-2-7b-hf")
MANIFEST = Path("/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1/exchange_calibration/manifest.json")
BLOCK, MIN_RANK, MAX_RANK, SEQ, BYTES, N_BOOT, MAX_PROPOSALS = 32, 32, 512, 2048, 2, 10000, 16
sys.path.insert(0, str(ROOT / "code"))
import profile_official_native_phi as pr

def load(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def obj_sha(x): return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def schedule_sha(sel): return obj_sha({k: [int(x) for x in v] for k,v in sorted(sel.items())})
def resource(sel):
    r = sum(sum(int(x) for x in v) for v in sel.values())
    return {"total_native_rank": r, "factor_bytes_fp16": r * (4096 + 512) * BYTES,
            "prefill_cache_bytes": r * SEQ * BYTES, "decode_cache_bytes": r * SEQ * BYTES}

def inputs(tok, m):
    w = tok((DATA/"wiki-train.hf_joined.raw").read_text(encoding="utf-8"),
            return_tensors="pt", add_special_tokens=False).input_ids[0]
    chunks = m["c4"]["chunks"]; wanted = {int(x["doc_index"]) for x in chunks}
    docs = {}; last = max(wanted)
    with gzip.open(m["c4"]["source"], "rt", encoding="utf-8") as f:
        for i,line in enumerate(f):
            if i in wanted: docs[i] = json.loads(line)["text"]
            if i >= last: break
    if wanted != set(docs): raise RuntimeError("missing exchange C4 documents")
    c4 = {}
    for x in chunks:
        ids = tok(docs[int(x["doc_index"])], return_tensors="pt",
                  add_special_tokens=False).input_ids
        c4[int(x["doc_index"])] = ids[:, int(x["token_start"]):int(x["token_end"])]
    return w, c4

@torch.no_grad()
def evaluate(model, m, w, c4):
    dev = next(model.parameters()).device; wiki=[]; c4n=[]
    for x in m["wikitext"]["chunks"]:
        ids=w[int(x["token_start"]):int(x["token_end"])].unsqueeze(0).to(dev)
        pos=torch.arange(ids.shape[1],device=dev).unsqueeze(0)
        logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=pos,use_cache=False).logits
        z=float(F.cross_entropy(logits[:,:-1,:].float().reshape(-1,logits.shape[-1]),
                                ids[:,1:].reshape(-1),reduction="mean").item())
        if not math.isfinite(z): raise FloatingPointError("non-finite Wiki calibration loss")
        wiki.append(z)
    for x in m["c4"]["chunks"]:
        ids=c4[int(x["doc_index"])].to(dev); pos=torch.arange(ids.shape[1],device=dev).unsqueeze(0)
        logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=pos,use_cache=False).logits
        z=float(F.cross_entropy(logits[:,:-1,:].float().reshape(-1,logits.shape[-1]),
                                ids[:,1:].reshape(-1),reduction="mean").item())
        if not math.isfinite(z): raise FloatingPointError("non-finite C4 calibration loss")
        c4n.append(z)
    return {"wiki":wiki,"c4":c4n,"wiki_mean_token_nll":sum(wiki)/len(wiki),
            "c4_mean_token_nll":sum(c4n)/len(c4n),"J":0.5*(sum(wiki)/len(wiki)+sum(c4n)/len(c4n))}

def bootstrap(d, seed):
    r=random.Random(seed); nw,nc=len(d["wiki"]),len(d["c4"]); out=[]
    for _ in range(N_BOOT):
        w=sum(d["wiki"][r.randrange(nw)] for _ in range(nw))/nw
        c=sum(d["c4"][r.randrange(nc)] for _ in range(nc))/nc; out.append(0.5*(w+c))
    out.sort(); return out[int(.025*N_BOOT)],out[int(.975*N_BOOT)]

def fisher(ratio):
    tag="%.1f"%ratio; p=ROOT/"results"/("fisher_"+tag.replace(".","p")+"_ratio_"+tag+"_fisher_uniform.json")
    d=load(p); return d,{k:[int(x) for x in v] for k,v in d["selection"].items()},p

def phi_rows(ratio):
    tag=("%.1f"%ratio).replace(".","p"); p=ROOT/"official_native_marginal"/("ratio_"+tag)/"units.csv"
    if not p.exists(): raise FileNotFoundError(str(p))
    d={}
    with p.open(newline="",encoding="utf-8-sig") as f:
        for x in csv.DictReader(f):
            if x["status"] not in {"COMPLETE","SKIPPED_LEGAL_MIN_RANK"}: raise RuntimeError("unexpected phi status")
            full_module=x["unit"].split("::group",1)[0]
            d[(full_module,int(x["group_id"]))]=x
    if len(d)!=512: raise RuntimeError("phi unit table is not complete: %d"%len(d))
    return d,p

def records(ratio, sel, signal):
    rows, path=phi_rows(ratio); out=[]
    for mod,ranks in sel.items():
        short="k_proj" if "k_proj" in mod else "v_proj"
        for g,rank in enumerate(ranks):
            x=rows[(mod,g)]; q=float(x["q"]); pv=x.get("phi_official","")
            pv=float(pv) if pv not in ("","None","nan") else None
            if signal=="fdp_prior": score=q
            elif signal=="fdp_native": score=pv
            else:
                seed=int(signal.rsplit("_",1)[1]); stable=int(hashlib.sha256(x["unit"].encode()).hexdigest()[:16],16)
                score=random.Random(seed^stable).random()
            out.append({"unit":x["unit"],"module_key":mod,"module":short,"group_id":g,
                        "rank":int(rank),"q":q,"phi_official":pv,"score":score})
    return out,path

def pairs(rs):
    rs=[x for x in rs if x["score"] is not None and math.isfinite(x["score"])]
    ds=sorted([x for x in rs if x["rank"]>MIN_RANK],key=lambda x:(x["score"],x["unit"]))
    rr=sorted([x for x in rs if x["rank"]<MAX_RANK],key=lambda x:(-x["score"],x["unit"]))
    out=[]; ud=set(); ur=set()
    for rec in rr:
        for don in ds:
            if rec["unit"]==don["unit"] or rec["unit"] in ur or don["unit"] in ud: continue
            out.append((don,rec)); ud.add(don["unit"]); ur.add(rec["unit"]); break
        if len(out)>=MAX_PROPOSALS: break
    return out

def parse_unit(u):
    m,g=u.split("::group"); return m,int(g)
def apply(sel,don,rec):
    out=copy.deepcopy(sel); dm,dg=parse_unit(don["unit"]); rm,rg=parse_unit(rec["unit"])
    if out[dm][dg]<=MIN_RANK or out[rm][rg]>=MAX_RANK: return None
    out[dm][dg]-=BLOCK; out[rm][rg]+=BLOCK; return out

def runtime(device, sel):
    pr.patch_offline_loaders()
    from utils import load_model_and_tokenizer
    from palu.model.modules.svd_linear import HeadwiseLowRankModule
    model,tok=load_model_and_tokenizer(str(MODEL)); model.to(torch.device(device))
    raw={}; parents={}
    for n,mod in list(model.named_modules()):
        if ("k_proj" in n or "v_proj" in n) and isinstance(mod,torch.nn.Linear):
            raw[n]=mod; parents[n]=pr.parent_and_child(model,n)
    factors=torch.load(ROOT/"cache"/"official_whiten_factors_rank512.pt",map_location="cpu",weights_only=False)
    installed={}
    for n,ranks in sel.items():
        installed[n]=pr.build_module(HeadwiseLowRankModule,raw[n],factors[n],ranks,device)
        p,c=parents[n]; setattr(p,c,installed[n])
    m=load(MANIFEST); w,c4=inputs(tok,m); model.eval()
    return model,m,w,c4,raw,parents,factors,installed,HeadwiseLowRankModule

def install(raw,parents,factors,sel,names,device,installed,klass):
    for n in names:
        mod=pr.build_module(klass,raw[n],factors[n],sel[n],device); p,c=parents[n]; setattr(p,c,mod); installed[n]=mod

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ratio",type=float,choices=[.3,.5,.7],required=True)
    ap.add_argument("--signal",choices=["fdp_prior","fdp_native","random_0","random_1","random_2"],required=True)
    ap.add_argument("--device",default="cuda"); a=ap.parse_args()
    tag=("%.1f"%a.ratio).replace(".","p"); out=ROOT/"official_exchange_calibration"/("ratio_"+tag)/a.signal
    out.mkdir(parents=True,exist_ok=True)
    if (out/"status.json").exists(): print(json.dumps({"status":"ALREADY_COMPLETE"})); return
    fisher_doc,sel,fp=fisher(a.ratio); action=load(ROOT/"PALU_OFFICIAL_NATIVE_ACTION_SPACE.json")
    qmap=load(ROOT/"PALU_OFFICIAL_Q_GROUP_MAPPING.json"); rs,phip=records(a.ratio,sel,a.signal); ps=pairs(rs)
    if a.signal=="fdp_native" and not any(x["phi_official"] is not None for x in rs): raise RuntimeError("no finite official phi")
    order={"status":"CANDIDATE_ORDER_FROZEN_PRE_QUALITY","ratio":a.ratio,"signal":a.signal,
           "rank_quantum":BLOCK,"minimum_rank":MIN_RANK,"maximum_rank":MAX_RANK,
           "pairs":[{"proposal":i+1,"donor":d,"receiver":r} for i,(d,r) in enumerate(ps)],
           "source_fisher_schedule":str(fp),"source_fisher_schedule_sha256":sha(fp),
           "official_phi_units":str(phip),"official_phi_units_sha256":sha(phip),
           "action_space_sha256":action["action_space_sha256"],"q_mapping_sha256":qmap["mapping_sha256"],
           "formal_quality_used_to_select":False}
    write(out/"candidate_order.json",order)
    model,m,w,c4,raw,parents,factors,installed,klass=runtime(a.device,sel)
    cur=evaluate(model,m,w,c4); base_res=resource(sel)
    common={"status":"EXCHANGE_BASELINE_COMPLETE","ratio":a.ratio,"signal":a.signal,"J":cur["J"],
            "wiki_mean_token_nll":cur["wiki_mean_token_nll"],"c4_mean_token_nll":cur["c4_mean_token_nll"],
            "resource":base_res,"source_fisher_schedule_sha256":sha(fp),"manifest_sha256":sha(MANIFEST),
            "action_space_sha256":action["action_space_sha256"],"q_mapping_sha256":qmap["mapping_sha256"]}
    write(out/"baseline.json",common); current=sel; samples=cur; decisions=[]
    for i,(don,rec) in enumerate(ps,1):
        cand=apply(current,don,rec)
        dec={"ratio":a.ratio,"signal":a.signal,"proposal":i,"donor":don["unit"],"receiver":rec["unit"],
             "donor_score":don["score"],"receiver_score":rec["score"],"accepted":False,
             "status":"INFEASIBLE_NATIVE_EXCHANGE","native_rank_quantum":BLOCK,"current_schedule_sha":schedule_sha(current)}
        if cand is None: decisions.append(dec); continue
        before,after=resource(current),resource(cand); dec.update({"resource_before":before,"resource_after":after,"resource_exact_match":before==after})
        if before!=after: decisions.append(dec); continue
        changed={don["module_key"],rec["module_key"]}; old={n:installed[n] for n in changed}
        install(raw,parents,factors,cand,changed,a.device,installed,klass); got=evaluate(model,m,w,c4)
        delta={"wiki":[b-x for x,b in zip(samples["wiki"],got["wiki"])] ,"c4":[b-x for x,b in zip(samples["c4"],got["c4"])]}
        dj=.5*(sum(delta["wiki"])/len(delta["wiki"])+sum(delta["c4"])/len(delta["c4"]))
        lo,hi=bootstrap(delta,20260919+i); ok=dj<0 and hi<0
        dec.update({"status":"ACCEPTED" if ok else "REJECTED","accepted":ok,"delta_J":dj,
                    "bootstrap_ci_low":lo,"bootstrap_ci_high":hi,"bootstrap_n":N_BOOT,
                    "candidate_schedule_sha":schedule_sha(cand)})
        if ok: current,samples=cand,got
        else:
            for n,oldmod in old.items(): p,c=parents[n]; setattr(p,c,oldmod); installed[n]=oldmod
        decisions.append(dec); gc.collect(); torch.cuda.empty_cache()
    final_res=resource(current); final_sha=schedule_sha(current)
    common={"status":"EXCHANGE_CALIBRATION_COMPLETE","ratio":a.ratio,"signal":a.signal,
            "decisions":decisions,"accepted_transfer_count":sum(int(x["accepted"]) for x in decisions),
            "final_schedule_sha":final_sha,"final_resource":final_res,
            "source_fisher_schedule_sha256":sha(fp),"manifest_sha256":sha(MANIFEST),
            "action_space_sha256":action["action_space_sha256"],"q_mapping_sha256":qmap["mapping_sha256"],
            "formal_quality_used_to_select":False}
    write(out/"decisions.json",common)
    write(out/"final_schedule.json",dict(common,selection=current,status="EXCHANGE_SCHEDULE_FROZEN_PRE_FORMAL_QUALITY"))
    write(out/"LOCK.json",{"status":"OFFICIAL_EXCHANGE_CALIBRATION_LOCKED","ratio":a.ratio,"signal":a.signal,
          "schedule_sha":final_sha,"resource":final_res,"exchange_manifest":str(MANIFEST),
          "exchange_manifest_sha256":sha(MANIFEST),"source_fisher_schedule_sha256":sha(fp),
          "action_space_sha256":action["action_space_sha256"],"q_mapping_sha256":qmap["mapping_sha256"],
          "rank_quantum":BLOCK,"acceptance_rule":"delta_J < 0 and bootstrap_ci_high < 0",
          "bootstrap_n":N_BOOT,"formal_quality_used_to_select":False})
    write(out/"status.json",{"status":"EXCHANGE_CALIBRATION_COMPLETE","ratio":a.ratio,"signal":a.signal,
          "schedule":str(out/"final_schedule.json"),"lock":str(out/"LOCK.json"),
          "schedule_sha":final_sha,"resource":final_res})
    print(json.dumps({"status":"EXCHANGE_CALIBRATION_COMPLETE","ratio":a.ratio,"signal":a.signal,
                      "accepted_transfer_count":sum(int(x["accepted"]) for x in decisions),
                      "schedule_sha":final_sha,"resource":final_res},indent=2),flush=True)

if __name__=="__main__": main()
