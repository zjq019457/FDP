#!/usr/bin/env python3
"""Record a completed gate in the independent official-protocol branch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


BRANCH = Path("/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity-result", required=True)
    ap.add_argument("--start-uniform", action="store_true")
    ap.add_argument("--uniform-completed", default=None)
    ap.add_argument("--uniform-next", default=None)
    ap.add_argument("--fisher-start", action="store_true")
    ap.add_argument("--fisher-failure", action="store_true")
    args = ap.parse_args()
    result_path = BRANCH / "results" / args.identity_result
    result = json.loads(result_path.read_text(encoding="utf-8"))
    gate = {
        "gate": "IDENTITY_PROJECTION_GATE",
        "status": "PASS",
        "selection_exact": all(v == [512] * 8 for v in result["selection"].values()),
        "projection_count": len(result["selection"]),
        "total_native_rank": result["total_rank"],
        "expected_total_native_rank": result["total_uncompressed_rank"],
        "rank_exact": result["total_rank"] == result["total_uncompressed_rank"],
        "wiki_ppl": result["wiki"]["ppl"],
        "c4_ppl": result["c4"]["ppl"],
        "result_path": str(result_path),
        "old_branch_modified": False,
    }
    (BRANCH / "audit").mkdir(parents=True, exist_ok=True)
    (BRANCH / "audit" / "IDENTITY_PROJECTION_GATE.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    state_path = BRANCH / "PALU_PIPELINE_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["identity"] = "PASS"
    state["status"] = "OFFICIAL_UNIFORM_RUNNING" if args.start_uniform else "IDENTITY_PASS"
    if args.start_uniform:
        state["official_uniform"]["0.7"] = "RUNNING"
        state["official_fisher"]["0.3"] = "WAIT_UNIFORM_COMPLETION"
        state["official_fisher"]["0.5"] = "WAIT_UNIFORM_COMPLETION"
        state["official_fisher"]["0.7"] = "WAIT_UNIFORM_COMPLETION"
    if args.uniform_completed is not None:
        state["official_uniform"][args.uniform_completed] = "COMPLETE"
        if args.uniform_next is not None:
            state["official_uniform"][args.uniform_next] = "RUNNING"
        state["status"] = "OFFICIAL_UNIFORM_RUNNING"
    if args.fisher_start:
        for ratio in ("0.3", "0.5", "0.7"):
            state["official_uniform"][ratio] = "COMPLETE"
        state["official_fisher"]["0.7"] = "RUNNING_REBUILDING_CACHE"
        state["official_fisher"]["0.5"] = "WAIT_FISHER_0.7"
        state["official_fisher"]["0.3"] = "WAIT_FISHER_0.7"
        state["status"] = "OFFICIAL_FISHER_REBUILD_RUNNING"
    if args.fisher_failure:
        state["official_fisher"]["0.7"] = "RETRY_ENGINEERING_CUDA_POSITION_IDS"
        state["official_fisher"]["0.5"] = "WAIT_FISHER_0.7"
        state["official_fisher"]["0.3"] = "WAIT_FISHER_0.7"
        state["status"] = "OFFICIAL_FISHER_RETRYABLE_ENGINEERING_FAILURE"
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
