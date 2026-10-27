#!/usr/bin/env python3
"""Freeze FIXED_FDP_SPECIFICATION.md from protocol JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import atomic_write_json, file_hash_or_empty, now_iso, sha256_file  # noqa: E402

PROTO = HERE / "FIXED_FDP_SCHEDULE_PROTOCOL.json"


def main() -> None:
    d = json.loads(PROTO.read_text(encoding="utf-8"))
    c = d["confirmatory_config"]
    f = d["fragility_formula"]
    lines = [
        "# Fixed FDP Schedule Specification",
        "",
        f"- generated_at: `{now_iso()}`",
        f"- protocol_sha256: `{sha256_file(PROTO)}`",
        "",
        "## Confirmatory Config (FGeb045)",
        "",
        f"- eb_base: `{c['eb_base']}`",
        f"- lambda_eb: `{c['lambda_eb']}`",
        f"- normalize: `{c['normalize']}`",
        f"- weights: `{json.dumps(c['weights'])}`",
        f"- freeze_layers: `{c['freeze_layers']}`",
        f"- entropy_collapse: `{c['entropy_collapse']}`",
        f"- clipping_rails: `{c['clipping_rails']}`",
        "",
        "## Fragility Formula",
        "",
        f"- Z_l_rel: {f['Z_l_rel']}",
        f"- S_l_rel: {f['S_l_rel']}",
        f"- V_l_rel: {f['V_l_rel']}",
        f"- A_l_rel: {f['A_l_rel']}",
        f"- aggregation: {f['aggregation_order']}",
        f"- s_l: `{f['s_l']}`",
        f"- eb_l: `{f['eb_l']}`",
        "",
        "## Operating Points (eb_base only varies)",
        "",
    ]
    for name, spec in d["operating_points"].items():
        lines.append(f"- **{name}**: target ratio `{spec['target_retained_ratio']}`, eb_base `{spec['eb_base_candidate']}`")
    lines.extend([
        "",
        "## Robustness Policy",
        "",
        "- lambda=1.2 is confirmatory; lambda in {0.6, 2.0} and cal_size sweeps are sensitivity only.",
        "- Best robustness result (e.g. lambda=2.0, cal=32, PPL=6.475) must NOT replace main config.",
        "",
        "## KV Bytes",
        "",
        f"- {d['kv_bytes_computation']['formula']}",
        "- Controls must match FDP **actual** realized bytes, not nominal retained ratio.",
    ])
    md = "\n".join(lines) + "\n"
    (HERE / "FIXED_FDP_SPECIFICATION.md").write_text(md, encoding="utf-8")
    atomic_write_json(HERE / "FREEZE_PROTOCOL_MANIFEST.json", {
        "generated_at": now_iso(),
        "protocol_sha256": sha256_file(PROTO),
        "spec_path": str(HERE / "FIXED_FDP_SPECIFICATION.md"),
    })
    print(json.dumps({"ok": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
