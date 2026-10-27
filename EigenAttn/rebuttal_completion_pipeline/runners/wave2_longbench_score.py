#!/usr/bin/env python3
"""Score WAVE2 LongBench predictions via slim/longbench_eval.scorer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "slim"))
from common import atomic_write_json, now_iso  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    import longbench_eval as lbe  # type: ignore

    pred_root = Path(args.pred_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"generated_at": now_iso(), "methods": {}}
    for method_dir in sorted(p for p in pred_root.iterdir() if p.is_dir() and p.name != "scores"):
        method = method_dir.name
        method_scores = {}
        for pred_file in sorted(method_dir.glob("*.jsonl")):
            task = pred_file.stem
            preds, answers, lengths = [], [], []
            all_classes = None
            with pred_file.open("r", encoding="utf-8") as f:
                for ln in f:
                    obj = json.loads(ln)
                    preds.append(obj["pred"])
                    answers.append(obj.get("answers") or [])
                    lengths.append(int(obj.get("length") or 0))
                    if obj.get("all_classes") is not None:
                        all_classes = obj["all_classes"]
            if not preds:
                continue
            try:
                score = lbe.scorer(task, preds, answers, lengths, all_classes)
            except Exception as e:
                method_scores[task] = {"error": repr(e), "n": len(preds)}
                continue
            if score is not None:
                method_scores[task] = {"n": len(preds), "score": score}
        results["methods"][method] = method_scores
        atomic_write_json(out_dir / f"{method}_scores.json", method_scores)

    atomic_write_json(out_dir / "LONGBENCH_WAVE2_SCORES.json", results)
    art = HERE / "artifacts" / "long_context"
    art.mkdir(parents=True, exist_ok=True)
    atomic_write_json(art / "LONG_CONTEXT_8K_STATUS.json", {
        "profile": "8k",
        "generated_at": now_iso(),
        "eval_executed": True,
        "subset": True,
        "limit_per_task": 20,
        "max_len": 8192,
        "methods": list(results["methods"].keys()),
        "scores_path": str(out_dir / "LONGBENCH_WAVE2_SCORES.json"),
        "note": "WAVE2 real EigenAttn generate+score; English subset with limit_per_task=20",
    })
    print(json.dumps({"ok": True, "methods": list(results["methods"].keys())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
