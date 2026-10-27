#!/usr/bin/env python3
"""WAVE2 final status rollup — honest executed vs blocked."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import atomic_write_json, now_iso  # noqa: E402


def _load(p: Path):
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {"path": str(p), "unreadable": True}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()
    root = Path(args.out_root)

    report = {
        "generated_at": now_iso(),
        "mcfix_aggregate": _load(root / "MCFIX_AGGREGATE.json"),
        "longbench_scores": _load(root / "longbench" / "scores" / "LONGBENCH_WAVE2_SCORES.json"),
        "opt30b": _load(root / "opt30b" / "OPT30B_WAVE2_STATUS.json"),
        "qwen": _load(root / "qwen" / "QWEN_WAVE2_GATE.json"),
        "external": _load(root / "external" / "EXTERNAL_WAVE2_STATUS.json"),
        "component": _load(root / "component_ablation" / "COMPONENT_ABLATION_WAVE2.json"),
    }
    executed = []
    blocked = []
    if report["mcfix_aggregate"]:
        executed.append("MCFIX_AGGREGATE")
    if report["longbench_scores"]:
        executed.append("LONGBENCH_8K_SUBSET")
    if (report["opt30b"] or {}).get("eval_executed"):
        executed.append("OPT30B_WIKI_PPL")
    elif report["opt30b"]:
        blocked.append("OPT30B:" + str((report["opt30b"] or {}).get("blocked")))
    if (report["qwen"] or {}).get("blocked_reason"):
        blocked.append("QWEN:" + report["qwen"]["blocked_reason"][:120])
    if (report["external"] or {}).get("blocked_reason"):
        blocked.append("EXTERNAL:" + report["external"]["blocked_reason"][:120])
    if (report["component"] or {}).get("eval_executed"):
        executed.append("COMPONENT_ANCHORS_PPL")

    report["summary"] = {"executed": executed, "blocked_or_gated": blocked}
    atomic_write_json(root / "WAVE2_FINAL_STATUS.json", report)
    art = HERE / "artifacts" / "wave2"
    art.mkdir(parents=True, exist_ok=True)
    atomic_write_json(art / "WAVE2_FINAL_STATUS.json", report)
    (HERE / "WAVE2_STATUS.md").write_text(
        "# WAVE2 Status\n\n"
        f"- Generated: {report['generated_at']}\n"
        f"- Executed: {', '.join(executed) or '(none)'}\n"
        f"- Blocked/gated: {len(blocked)} item(s)\n"
        f"- Details: `{root / 'WAVE2_FINAL_STATUS.json'}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "executed": executed, "blocked": len(blocked)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
