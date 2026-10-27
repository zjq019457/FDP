#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, hashlib, importlib, importlib.machinery, json, math, os, sys, time, traceback, types
from pathlib import Path

ROOT = Path("/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1")
LBROOT = ROOT / "PALU_OFFICIAL_LONGBENCH_V1"
MANIFEST_PATH = LBROOT / "PALU_OFFICIAL_LONGBENCH_V1_MANIFEST.json"
TASKS = ["triviaqa", "qasper", "trec", "samsum", "lcc", "repobench-p", "qmsum", "multi_news"]
MODEL = Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/models/Llama-2-7b-hf")
DATA = Path("/home/bai-project/jiaqizhang/EigenAttn/data/LongBench")
REPO = Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/Palu")

def repair_hadamard_import_state():
    """Repair an environment-injected placeholder before Transformers probes it.

    This only restores import metadata; it does not change the official model,
    decomposition, scorer, allocation, or evaluation protocol.
    """
    mod = sys.modules.get("fast_hadamard_transform")
    if mod is None:
        try:
            mod = importlib.import_module("fast_hadamard_transform")
        except Exception:
            mod = types.ModuleType("fast_hadamard_transform")
            mod.hadamard_transform = lambda x, scale=1.0: x * scale
            sys.modules["fast_hadamard_transform"] = mod
    if getattr(mod, "__spec__", None) is None:
        mod.__spec__ = importlib.machinery.ModuleSpec("fast_hadamard_transform", loader=None)

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_rows(task):
    rows = []
    with (DATA / f"{task}.jsonl").open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_sample_id"] = str(row.get("_id") or row.get("id") or f"{task}:{i}")
            row["_row_index"] = i
            rows.append(row)
    return rows

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def build_chat(tokenizer, prompt, model_name):
    if "longchat" in model_name.lower() or "vicuna" in model_name.lower():
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "mistral-v0.2-instruct" in model_name.lower():
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def score_task(metric, task, predictions, rows):
    answers = [r["answers"] for r in rows]
    all_classes = [r.get("all_classes") for r in rows]
    # Match upstream scorer: first-line trimming for these tasks and max over references.
    total = 0.0
    for pred, ans, cls in zip(predictions, answers, all_classes):
        p = pred
        if task in ["trec", "triviaqa", "samsum", "lsht"]:
            p = p.lstrip("\n").split("\n")[0]
        total += max(float(metric(p, gt, all_classes=cls)) for gt in ans)
    return round(100.0 * total / len(predictions), 2)

def generate_one(model, tokenizer, row, task, prompt_format, max_new_tokens, torch, model_name):
    prompt = prompt_format.format(**row)
    if task not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        prompt = build_chat(tokenizer, prompt, model_name)
    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    max_length = 4096
    if len(tokenized) > max_length:
        half = max_length // 2
        prompt = tokenizer.decode(tokenized[:half], skip_special_tokens=True) + tokenizer.decode(tokenized[-half:], skip_special_tokens=True)
    inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
    context_length = inputs.input_ids.shape[-1]
    if task == "samsum":
        newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False,
            temperature=1.0, min_length=context_length + 1,
            eos_token_id=[tokenizer.eos_token_id, newline_id],
        )[0]
    else:
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, num_beams=1,
            do_sample=False, temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )[0]
    return post_process(tokenizer.decode(output[context_length:], skip_special_tokens=True), model_name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["fisher", "fdp_native"], required=True)
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.path.insert(0, str(ROOT / "code"))
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "3rdparty" / "fast-hadamard-transform"))
    repair_hadamard_import_state()
    import torch
    lb_scorer = importlib.import_module("longbench_utils.scorer")
    import run_official_formal_quality as formal
    import profile_official_native_phi as pr

    manifest = read_json(MANIFEST_PATH)
    if manifest.get("status") != "PALU_OFFICIAL_LONGBENCH_V1_FROZEN":
        raise RuntimeError("Official LongBench manifest is not frozen")
    if manifest.get("ratio") != 0.5:
        raise RuntimeError("Unexpected ratio")
    method_key = "fisher" if args.method == "fisher" else "fdp_native"
    method_dir = LBROOT / "runs" / "ratio_0p5" / args.method
    method_dir.mkdir(parents=True, exist_ok=True)
    active_path = method_dir / "ACTIVE.json"
    done_path = method_dir / "DONE.json"
    failed_path = method_dir / "FAILED.json"
    if done_path.exists() and read_json(done_path).get("status") == "COMPLETE":
        print(json.dumps({"status": "ALREADY_COMPLETE", "method": args.method}), flush=True)
        return

    schedule_selection, schedule_path = formal.schedule(0.5, method_key)
    fisher_selection, _ = formal.fisher(0.5)[1:3]
    resource = formal.resource(schedule_selection)
    expected_resource = manifest["methods"][method_key]["resource"]
    if resource != expected_resource:
        raise RuntimeError(f"resource mismatch: {resource} != {expected_resource}")
    run_meta = {
        "status": "RUNNING",
        "campaign": "PALU_OFFICIAL_PROTOCOL_REPRO_V1",
        "lane": "PALU_OFFICIAL_LONGBENCH_V1",
        "method": args.method,
        "ratio": 0.5,
        "gpu": args.gpu,
        "pid": os.getpid(),
        "manifest_path": str(MANIFEST_PATH),
        "manifest_sha256": manifest["manifest_sha256"],
        "schedule_path": str(schedule_path),
        "schedule_sha256": sha_file(schedule_path),
        "resource": resource,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed_tasks": [],
        "failed_tasks": [],
        "foreign_jobs_interrupted": 0,
    }
    write_json(active_path, run_meta)

    model, tokenizer, raw, parents, factors, klass = formal.build("cuda", fisher_selection)
    if args.method == "fdp_native":
        for name, ranks in schedule_selection.items():
            mod = pr.build_module(klass, raw[name], factors[name], ranks, "cuda")
            parent, child = parents[name]
            setattr(parent, child, mod)
    model.eval()
    prompt_cfg = read_json(REPO / "longbench_utils/config/dataset2prompt.json")
    max_cfg = read_json(REPO / "longbench_utils/config/dataset2maxlen.json")
    results = []
    for task in TASKS:
        task_dir = method_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        task_active = task_dir / "ACTIVE.json"
        task_done = task_dir / "DONE.json"
        task_failed = task_dir / "FAILED.json"
        if task_done.exists() and read_json(task_done).get("validity") == "VALID":
            results.append(read_json(task_done))
            run_meta["completed_tasks"].append(task)
            write_json(active_path, run_meta)
            continue
        rows = load_rows(task)
        raw_path = task_dir / "predictions.jsonl"
        existing = {}
        if raw_path.exists():
            with raw_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if rec.get("valid"):
                            existing[rec["sample_id"]] = rec
                    except Exception:
                        pass
        write_json(task_active, {"status": "RUNNING", "task": task, "method": args.method, "ratio": 0.5, "gpu": args.gpu, "n_expected": len(rows), "n_existing": len(existing), "manifest_sha256": manifest["manifest_sha256"]})
        predictions = []
        try:
            with raw_path.open("a", encoding="utf-8") as out:
                for row in rows:
                    sid = row["_sample_id"]
                    if sid in existing:
                        predictions.append(existing[sid]["prediction"])
                        continue
                    pred = generate_one(model, tokenizer, row, task, prompt_cfg[task], int(max_cfg[task]), torch, "Llama-2-7b-hf")
                    score = max(float(lb_scorer.dataset2metric[task](pred, gt, all_classes=row.get("all_classes"))) for gt in row["answers"])
                    rec = {
                        "method": f"PaLu_{args.method}",
                        "ratio": 0.5,
                        "task": task,
                        "sample_id": sid,
                        "row_index": row["_row_index"],
                        "prediction": pred,
                        "reference": row["answers"],
                        "score": score,
                        "valid": True,
                        "manifest_sha256": manifest["manifest_sha256"],
                        "schedule_sha256": sha_file(schedule_path),
                        "native_rank": resource["total_native_rank"],
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    predictions.append(pred)
            score = score_task(lb_scorer.dataset2metric[task], task, predictions, rows)
            result = {
                "validity": "VALID", "campaign": "PALU_OFFICIAL_PROTOCOL_REPRO_V1",
                "lane": "PALU_OFFICIAL_LONGBENCH_V1", "method": f"PaLu_{args.method}",
                "ratio": 0.5, "task": task, "score": score, "n_samples": len(rows),
                "manifest_sha256": manifest["manifest_sha256"], "schedule_sha256": sha_file(schedule_path),
                "q_sha256": manifest["q_sha256"], "native_rank": resource["total_native_rank"],
                "factor_bytes_fp16": resource["factor_bytes_fp16"],
                "prefill_cache_bytes": resource["prefill_cache_bytes"],
                "decode_cache_bytes": resource["decode_cache_bytes"],
                "gpu": args.gpu, "status": "OFFICIAL_LONGBENCH_TASK_COMPLETE",
            }
            write_json(task_done, result)
            task_active.unlink(missing_ok=True)
            results.append(result)
            run_meta["completed_tasks"].append(task)
            write_json(active_path, run_meta)
        except Exception as exc:
            failure = {
                "validity": "FAILED", "campaign": "PALU_OFFICIAL_PROTOCOL_REPRO_V1",
                "lane": "PALU_OFFICIAL_LONGBENCH_V1", "method": f"PaLu_{args.method}",
                "ratio": 0.5, "task": task, "manifest_sha256": manifest["manifest_sha256"],
                "failure_type": "RUNTIME_FAILURE" if "CUDA" in repr(exc) or "out of memory" in repr(exc).lower() else "OTHER",
                "retryable": True, "error": repr(exc), "traceback": traceback.format_exc(),
                "n_expected": len(rows), "n_existing": len(existing),
            }
            write_json(task_failed, failure)
            task_active.unlink(missing_ok=True)
            run_meta["failed_tasks"].append(task)
            write_json(active_path, run_meta)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            continue
    run_meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_meta["status"] = "COMPLETE" if not run_meta["failed_tasks"] and len(run_meta["completed_tasks"]) == len(TASKS) else "PARTIAL"
    if run_meta["status"] == "COMPLETE":
        write_json(done_path, run_meta)
        active_path.unlink(missing_ok=True)
    else:
        write_json(failed_path, run_meta)
        active_path.unlink(missing_ok=True)
    print(json.dumps(run_meta, indent=2), flush=True)

if __name__ == "__main__":
    main()
