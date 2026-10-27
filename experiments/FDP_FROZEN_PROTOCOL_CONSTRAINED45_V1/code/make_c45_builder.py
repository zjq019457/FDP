#!/usr/bin/env python3
"""Create the adapted builder for FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1 from the frozen offline builder.

Only two kinds of edits, each asserted to apply exactly once:
  (a) path-only: the copy lives outside EigenAttn/tools, so sibling-tool and repo roots are pinned to the
      frozen locations (same modules, same files);
  (b) the official EigenAttn Llama 45% loop condition (upstream 8bd0d3fa decompose/eigen_attn.py,
      min_thresh=0.45), behind --official_min_thresh (default 0.0 = disabled = frozen behaviour).
      Timing follows upstream: the candidate is evaluated first, then the loop continues only while
      err < eb AND rank_kq > t*max_rank_kq AND rank_v > t*max_rank_v (max ranks = k_proj/v_proj rows).
      A candidate violating the condition is never accepted (upstream step-back to the previous candidate).
      Builder semantics kept: threshold floor 0.30, step 0.02, accept err <= eb, rank lattice/finalize.
"""
import difflib, hashlib
from pathlib import Path

SRC = Path("/home/bai-project/jiaqizhang/EigenAttn/EigenAttn/tools/build_eigen_basis_llama_error_budget_offline.py")
DST = Path("/diskF/jiaqizhang/FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1/code/build_eigen_basis_llama_error_budget_offline_c45.py")
FROZEN_SHA = "e23294114d29da8428229534967daff8f494c8f4078bb3c76db02065b6141336"

src = SRC.read_text()
assert hashlib.sha256(src.encode()).hexdigest() == FROZEN_SHA, "frozen builder changed"

EDITS = [
    ("_TOOLS = Path(__file__).resolve().parent\n",
     "_TOOLS = Path(\"/home/bai-project/jiaqizhang/EigenAttn/EigenAttn/tools\")  # C45: pinned to frozen tools dir (path-only)\n"),
    ("    repo_root = Path(__file__).resolve().parents[1]\n",
     "    repo_root = _TOOLS.parent  # C45: pinned to frozen repo root (path-only)\n"),
    ("    args = ap.parse_args()\n",
     "    ap.add_argument(\n"
     "        \"--official_min_thresh\",\n"
     "        type=float,\n"
     "        default=0.0,\n"
     "        help=\"C45: official EigenAttn Llama loop condition rank > t*max_rank for both QK and V (upstream t=0.45); 0 disables.\",\n"
     "    )\n"
     "    args = ap.parse_args()\n"),
    ("            while err < float(eb) and th > float(args.svd_threshold_floor):\n",
     "            # C45: official 45% loop condition (upstream 8bd0d3fa); max ranks as upstream (k_proj/v_proj rows).\n"
     "            _mt45 = float(args.official_min_thresh)\n"
     "            _max_rk45 = int(layer.self_attn.k_proj.weight.shape[0])\n"
     "            _max_rv45 = int(layer.self_attn.v_proj.weight.shape[0])\n"
     "\n"
     "            def _ok45(rk, rv):\n"
     "                return _mt45 <= 0 or (int(rk) > _mt45 * _max_rk45 and int(rv) > _mt45 * _max_rv45)\n"
     "\n"
     "            _eval_rk, _eval_rv = _max_rk45, _max_rv45  # upstream initialises ranks to max before the loop\n"
     "            while err < float(eb) and th > float(args.svd_threshold_floor) and _ok45(_eval_rk, _eval_rv):\n"),
    ("                if err <= float(eb):\n                    best_th = float(th)\n",
     "                _eval_rk, _eval_rv = int(rank_kq), int(rank_v)  # C45: ranks of the candidate just evaluated\n"
     "                if err <= float(eb) and _ok45(rank_kq, rank_v):\n                    best_th = float(th)\n"),
    ("            th = float(best_th)\n            rank_kq = int(best_rank_k)\n",
     "            _efail = not (err < float(eb))\n"
     "            _rfail = not _ok45(_eval_rk, _eval_rv)\n"
     "            _stop_reason = \"both\" if (_efail and _rfail) else (\"error_budget\" if _efail else (\"min_thresh_45\" if _rfail else \"floor\"))\n"
     "            th = float(best_th)\n            rank_kq = int(best_rank_k)\n"),
    ("                    \"threshold_trace\": trace,\n                }\n",
     "                    \"threshold_trace\": trace,\n"
     "                    \"official_min_thresh\": float(_mt45),\n"
     "                    \"stop_reason\": _stop_reason,\n"
     "                }\n"),
]

out = src
for old, new in EDITS:
    n = out.count(old)
    assert n == 1, (n, old[:80])
    out = out.replace(old, new)
DST.write_text(out)
diff = "".join(difflib.unified_diff(src.splitlines(True), out.splitlines(True), str(SRC), str(DST)))
DST.with_suffix(".diff").write_text(diff)
print("written", DST, hashlib.sha256(out.encode()).hexdigest())
print(diff)
