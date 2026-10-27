
#!/usr/bin/env python3
import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import random
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1")
REPO = Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/Palu")
DATA = Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/data")
MODEL = Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/models/Llama-2-7b-hf")
MANIFEST = Path("/diskE/jiaqizhang/PALU_FISHER_VS_FUNCTIONAL_MARGINAL_V1/marginal/calibration_manifest.json")
SEQ = 2048
BLOCK = 32
FP16_BYTES = 2

def load(path):
    return json.loads(Path(path).read_text())

def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + chr(10))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def install_hadamard():
    try:
        import fast_hadamard_transform
        return
    except Exception:
        pass
    mod = types.ModuleType("fast_hadamard_transform")
    mod.hadamard_transform = lambda x, scale=1.0: x * scale
    sys.modules["fast_hadamard_transform"] = mod

def patch_offline_loaders():
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "3rdparty" / "fast-hadamard-transform"))
    install_hadamard()
    import palu.data_utils as data_utils
    import palu.decomposition as decomposition
    raw_path = DATA / "wiki-train.hf_joined.raw"
    def local_get(name, tokenizer, model_id, nsamples, seqlen=2048, seed=3):
        if name != "wikitext2":
            return data_utils.get_calib_data(name, tokenizer, model_id, nsamples, seqlen, seed)
        cache = ROOT / "cache" / ("%s_%s_%d_%d_%d.pt" % (name, str(MODEL).replace("/", "_"), nsamples, seqlen, seed))
        if cache.exists():
            return torch.load(cache, map_location="cpu")
        text = raw_path.read_text(encoding="utf-8")
        random.seed(seed)
        data = []
        for _ in range(nsamples):
            i = random.randint(0, len(text) - seqlen - 1)
            enc = tokenizer(text[i:i + seqlen * 10], return_tensors="pt")
            ids = enc.input_ids[:, :seqlen]
            data.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, cache)
        return data
    data_utils.get_calib_data = local_get
    decomposition.get_calib_data = local_get
    return decomposition

def parent_and_child(model, full_name):
    parts = full_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]

def build_module(HeadwiseLowRankModule, raw, factors, ranks, device):
    new = HeadwiseLowRankModule(ranks, raw.in_features, raw.out_features, bias=raw.bias is not None)
    new = new.to(device=device, dtype=raw.weight.dtype)
    for i, rank in enumerate(ranks):
        left, right = factors[i]
        new.U[i].weight.data.copy_(left[:, :rank].to(device=device, dtype=raw.weight.dtype))
        if raw.bias is not None:
            old_bias = raw.bias.data.reshape(len(factors), -1)[i]
            new.U[i].bias.data.copy_(old_bias.to(device=device, dtype=raw.weight.dtype))
    new.VT.weight.data.copy_(torch.cat([pair[1][:rank, :] for pair, rank in zip(factors, ranks)], dim=0).to(device=device, dtype=raw.weight.dtype))
    return new

def c4_docs(path, n=1100):
    docs = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line)["text"])
            if len(docs) == n:
                break
    return docs

@torch.no_grad()
def evaluate(model, manifest, wiki_ids, c4_ids):
    device = next(model.parameters()).device
    total = {"wiki": [0.0, 0], "c4": [0.0, 0]}
    for chunk in manifest["wikitext"]["chunks"]:
        ids = wiki_ids[chunk["token_start"]:chunk["token_end"]].unsqueeze(0).to(device)
        pos = torch.arange(ids.shape[1], device=device).unsqueeze(0)
        out = model.model(ids, attention_mask=torch.ones_like(ids), position_ids=pos, use_cache=False)
        logits = model.lm_head(out[0])
        loss = F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum")
        total["wiki"][0] += float(loss.item())
        total["wiki"][1] += ids.shape[1] - 1
    for chunk in manifest["c4"]["chunks"]:
        ids = c4_ids[chunk["doc_index"]].to(device)
        pos = torch.arange(ids.shape[1], device=device).unsqueeze(0)
        out = model.model(ids, attention_mask=torch.ones_like(ids), position_ids=pos, use_cache=False)
        logits = model.lm_head(out[0])
        loss = F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum")
        total["c4"][0] += float(loss.item())
        total["c4"][1] += ids.shape[1] - 1
    wiki_nll = total["wiki"][0] / total["wiki"][1]
    c4_nll = total["c4"][0] / total["c4"][1]
    return {
        "wiki_nll": wiki_nll,
        "wiki_ppl": math.exp(wiki_nll),
        "c4_nll": c4_nll,
        "c4_ppl": math.exp(c4_nll),
        "J": 0.5 * (wiki_nll + c4_nll),
        "wiki_tokens": total["wiki"][1],
        "c4_tokens": total["c4"][1],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, choices=[0.3, 0.5, 0.7], required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    ratio = "%.1f" % args.ratio
    tag = ratio.replace(".", "p")
    os.chdir(ROOT)
    manifest = load(MANIFEST)
    result = load(ROOT / "results" / ("fisher_" + tag + "_ratio_" + ratio + "_fisher_uniform.json"))
    selection = result["selection"]
    mapping = load(ROOT / "PALU_OFFICIAL_Q_GROUP_MAPPING.json")
    qmap = {(row["layer"], row["module"], row["group_id"]): row["source_q"] for row in mapping["layer_to_group_mapping"]}
    fisher_path = ROOT / "cache" / "_diskF_jiaqizhang_PALU_PAPER_REPRO_AND_FDP_V2_models_Llama-2-7b-hf_calib_fisher_info.pt"
    fisher_cache = torch.load(fisher_path, map_location="cpu", weights_only=False)
    fisher_means = {}
    for name, tensor in fisher_cache.items():
        if "k_proj" not in name and "v_proj" not in name:
            continue
        arr = tensor.reshape(8, -1, tensor.shape[-1])
        fisher_means[name] = [float(arr[i].mean().item()) for i in range(8)]

    decomposition = patch_offline_loaders()
    from utils import load_model_and_tokenizer
    from palu.model.modules.svd_linear import HeadwiseLowRankModule, _per_head_whiten_decomposition_from_weight
    model, tokenizer = load_model_and_tokenizer(str(MODEL))
    model.to(torch.device(args.device))
    device = next(model.parameters()).device
    args_ns = types.SimpleNamespace(calib_dataset="wikitext2", model_id=str(MODEL), use_cache=True)
    decomposition.get_whiten_scale_matrix(model, tokenizer, args_ns, args.device)

    raw_modules = {}
    parents = {}
    for name, module in list(model.named_modules()):
        if ("k_proj" in name or "v_proj" in name) and isinstance(module, torch.nn.Linear):
            raw_modules[name] = module
            parents[name] = parent_and_child(model, name)

    factor_path = ROOT / "cache" / "official_whiten_factors_rank512.pt"
    if factor_path.exists():
        factors = torch.load(factor_path, map_location="cpu", weights_only=False)
    else:
        factors = {}
        for name, raw in raw_modules.items():
            weights = raw.weight.data.reshape(8, -1, raw.in_features)
            factors[name] = []
            for group_id in range(8):
                left, right = _per_head_whiten_decomposition_from_weight(weights[group_id], raw.scaling_diag_matrix, 512)
                factors[name].append((left.detach().cpu().half(), right.detach().cpu().half()))
        torch.save(factors, factor_path)

    baseline_modules = {}
    for name, raw in raw_modules.items():
        baseline_modules[name] = build_module(HeadwiseLowRankModule, raw, factors[name], [int(x) for x in selection[name]], device)
        parent, child = parents[name]
        setattr(parent, child, baseline_modules[name])

    wiki_ids = tokenizer((DATA / "wiki-test.hf_joined.raw").read_text(encoding="utf-8"), return_tensors="pt").input_ids[0]
    docs = c4_docs(DATA / "c4-validation.00000-of-00008.json.gz", manifest["c4"]["pool_docs"])
    needed = sorted({int(row["doc_index"]) for row in manifest["c4"]["chunks"]})
    c4_ids = {i: tokenizer(docs[i], return_tensors="pt").input_ids[:, :SEQ] for i in needed}
    baseline = evaluate(model, manifest, wiki_ids, c4_ids)
    total_rank = sum(sum(int(x) for x in values) for values in selection.values())
    total_factor_bytes = total_rank * (4096 + 512) * FP16_BYTES
    total_cache_bytes = total_rank * SEQ * FP16_BYTES
    outdir = ROOT / "official_native_marginal" / ("ratio_" + tag)
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / "baseline.json", {
        "status": "MARGINAL_BASELINE_COMPLETE",
        "ratio": args.ratio,
        "total_native_rank": total_rank,
        "factor_bytes_fp16": total_factor_bytes,
        "cache_bytes_fp16_seq2048": total_cache_bytes,
        "quality": baseline,
        "manifest": str(MANIFEST),
        "manifest_sha256": sha(MANIFEST),
        "action_space_sha256": load(ROOT / "PALU_OFFICIAL_NATIVE_ACTION_SPACE.json")["action_space_sha256"],
        "quality_used_to_design_selection": False,
    })
    fields = ["ratio", "unit", "layer", "module", "group_id", "q", "fisher_importance", "native_rank_before", "native_rank_after", "native_rank_delta", "factor_bytes_before", "factor_bytes_after", "cache_bytes_before", "cache_bytes_after", "relative_native_resource_reduction", "delta_J", "phi_official", "candidate_wiki_ppl", "candidate_c4_ppl", "status"]
    csv_path = outdir / "units.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, values in selection.items():
            layer = int(name.split(".")[2])
            module = "k_proj" if "k_proj" in name else "v_proj"
            for group_id, before in enumerate(values):
                before = int(before)
                common = {
                    "ratio": args.ratio,
                    "unit": name + "::group" + str(group_id),
                    "layer": layer,
                    "module": module,
                    "group_id": group_id,
                    "q": qmap[(layer, module, group_id)],
                    "fisher_importance": fisher_means.get(name, [None] * 8)[group_id],
                    "native_rank_before": before,
                    "factor_bytes_before": total_factor_bytes,
                    "cache_bytes_before": total_cache_bytes,
                }
                if before <= BLOCK:
                    common.update({
                        "native_rank_after": before,
                        "native_rank_delta": 0,
                        "factor_bytes_after": total_factor_bytes,
                        "cache_bytes_after": total_cache_bytes,
                        "relative_native_resource_reduction": 0.0,
                        "delta_J": "",
                        "phi_official": "",
                        "candidate_wiki_ppl": "",
                        "candidate_c4_ppl": "",
                        "status": "SKIPPED_LEGAL_MIN_RANK",
                    })
                    writer.writerow(common)
                    continue
                alt = [int(x) for x in values]
                alt[group_id] = before - BLOCK
                parent, child = parents[name]
                candidate = build_module(HeadwiseLowRankModule, raw_modules[name], factors[name], alt, device)
                setattr(parent, child, candidate)
                cand = evaluate(model, manifest, wiki_ids, c4_ids)
                setattr(parent, child, baseline_modules[name])
                after_rank = total_rank - BLOCK
                rel = BLOCK / total_rank
                delta = cand["J"] - baseline["J"]
                common.update({
                    "native_rank_after": before - BLOCK,
                    "native_rank_delta": -BLOCK,
                    "factor_bytes_after": after_rank * (4096 + 512) * FP16_BYTES,
                    "cache_bytes_after": after_rank * SEQ * FP16_BYTES,
                    "relative_native_resource_reduction": rel,
                    "delta_J": delta,
                    "phi_official": delta / rel,
                    "candidate_wiki_ppl": cand["wiki_ppl"],
                    "candidate_c4_ppl": cand["c4_ppl"],
                    "status": "COMPLETE",
                })
                writer.writerow(common)
                stream.flush()
                del candidate
                gc.collect()
                torch.cuda.empty_cache()
    write_json(outdir / "status.json", {
        "status": "MARGINAL_PROFILE_COMPLETE",
        "ratio": args.ratio,
        "units_csv": str(csv_path),
        "manifest_sha256": sha(MANIFEST),
        "action_space_sha256": load(ROOT / "PALU_OFFICIAL_NATIVE_ACTION_SPACE.json")["action_space_sha256"],
        "quality_used_to_design_selection": False,
    })
    print(json.dumps({"status": "MARGINAL_PROFILE_COMPLETE", "ratio": args.ratio, "baseline_J": baseline["J"], "output": str(outdir)}, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()

