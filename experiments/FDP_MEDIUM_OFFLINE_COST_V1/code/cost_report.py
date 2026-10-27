#!/usr/bin/env python3
"""FDP_MEDIUM_OFFLINE_COST_V1: SHA audit against the frozen Medium artifacts + report files. Read-only on frozen data."""
import csv, hashlib, json
from pathlib import Path

ROOT = Path("/diskF/jiaqizhang/FDP_MEDIUM_OFFLINE_COST_V1"); S = ROOT / "stages"; W = ROOT / "work"
CAMP = Path("/diskF/jiaqizhang/fdp125_final_campaign_v1/results/llama/medium")
CTRL = Path("/diskE/jiaqizhang/FDP125_STRICT_FORMAL_CLOSURE_V1/resource_search_retry_gpu2_v2/medium/eb_0p0567")
GiB = 2 ** 30


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def jsha(o): return hashlib.sha256(json.dumps(o, separators=(",", ":")).encode()).hexdigest()
def rj(p): return json.loads(Path(p).read_text())
def B(r): return sum((a + b) * 4096 for a, b in r)


T = {p.stem.replace(".timing", ""): rj(p) for p in S.glob("*.timing.json")}
raw = rj(ROOT / "MEDIUM_STAGE_TIMINGS_RAW.json")
assert "errors" not in raw, raw.get("errors")

# ---------------- SHA / identity audit
par, fdp = rj(CAMP / "uniform/provenance.json"), rj(CAMP / "fdp125/provenance.json")
pr = rj(W / "fdp_chain/uniform_parent/final_model/low_rank_config.json")
fr = rj(W / "fdp_chain/fdp125/final_model/low_rank_config.json")
uf = W / "uniform_comparator/final_model/low_rank_config.json"
if not uf.exists(): uf = W / "uniform_comparator/model/low_rank_config.json"
ur = rj(uf)
st_new, st_old = rj(W / "fdp_chain/step1.json"), rj(CAMP / "step1.json")
pl_new, pl_old = st_new["per_layer"], st_old["per_layer"]
maxdiff = 0.0; keys_equal = sorted(pl_new) == sorted(pl_old)
for l in pl_old:
    for k, v in pl_old[l].items():
        if isinstance(v, (int, float)) and isinstance(pl_new.get(l, {}).get(k), (int, float)):
            maxdiff = max(maxdiff, abs(float(v) - float(pl_new[l][k])))
E = T["E_fdp_score_aggregation_eb"]
audit = {
    "parent_uniform": {"rank_sha": jsha(pr), "frozen_rank_sha": par["rank_sha"], "B_live": B(pr), "frozen_B_live": par["B_live"],
                       "ranks_equal": pr == par["ranks"],
                       "basis_sha": sha(W / "fdp_chain/uniform_parent/basis/fdp125_llama_medium_uniform_perlayer_eb.pt"), "frozen_basis_sha": par["basis_sha"]},
    "step1_profile": {"file_sha": sha(W / "fdp_chain/step1.json"), "frozen_file_sha": "67780d18efcd02d353a2a43c2941cd8a7c3cc1e3ce3b282d5cb8b24dfa44c858",
                      "per_layer_keys_equal": keys_equal, "max_abs_numeric_diff_per_layer": maxdiff},
    "fdp_eb_vector": {"eb_sha": E["eb_sha"], "frozen_eb_sha": fdp["eb_sha"], "max_abs_diff_vs_frozen": max(abs(a - b) for a, b in zip(E["eb_vector"], fdp["eb_vector"]))},
    "fdp": {"rank_sha": jsha(fr), "frozen_rank_sha": fdp["rank_sha"], "B_live": B(fr), "frozen_B_live": fdp["B_live"], "ranks_equal": fr == fdp["ranks"],
            "basis_sha": sha(W / "fdp_chain/fdp125/basis/fdp125_llama_medium_fdp125_perlayer_eb.pt"), "frozen_basis_sha": fdp["basis_sha"]},
    "uniform_comparator": {"low_rank_config_file_sha": sha(uf), "frozen_file_sha": "db207077467e658317a1972240a42b211212563f38b3e819210b90006a1615cb",
                           "B_live": B(ur), "frozen_B_live": 199917568, "ranks_equal": ur == rj(CTRL / "low_rank_config.json")},
}
for k, v in audit.items():
    checks = [v[a] == v["frozen_" + a] for a in ("rank_sha", "B_live", "basis_sha", "eb_sha", "low_rank_config_file_sha") if a in v and "frozen_" + a in v]
    v["PASS"] = all(checks) and v.get("ranks_equal", True)
audit["step1_profile"]["PASS"] = audit["step1_profile"]["per_layer_keys_equal"] and audit["step1_profile"]["max_abs_numeric_diff_per_layer"] == 0.0
(ROOT / "MEDIUM_SHA_AUDIT.json").write_text(json.dumps(audit, indent=1))

# ---------------- timings / memory
def m(s): return T[s]["wall_s"] / 60
uni = ["U_uniform_comparator_basis_rank_search", "U_uniform_comparator_materialize"]
fdp_st = ["B_parent_uniform_basis_rank_search", "C_parent_uniform_materialize", "D_fdp_step1_profiling", "E_fdp_score_aggregation_eb",
          "F_fdp_pass2_basis_rank_search", "G_fdp_materialize"]
U_tot = sum(m(s) for s in uni); F_tot = sum(m(s) for s in fdp_st)
prof = m("C_parent_uniform_materialize") + m("D_fdp_step1_profiling") + m("E_fdp_score_aggregation_eb")
search = m("F_fdp_pass2_basis_rank_search") + m("G_fdp_materialize")
def peak(stages, key): return max(T[s].get(key, 0) for s in stages)
def tpk(stages): return max(T[s].get("torch", {}).get("torch_peak_allocated_bytes", 0) for s in stages)
def tpr(stages): return max(T[s].get("torch", {}).get("torch_peak_reserved_bytes", 0) for s in stages)
stage_rows = []
labels = {"B_parent_uniform_basis_rank_search": "A+B parent Uniform data/calibration + basis + rank search (eb 0.060)",
          "C_parent_uniform_materialize": "C parent Uniform materialize/save (needed for profiling)",
          "D_fdp_step1_profiling": "D FDP Step 1 Dense+compressed forwards and diagnostics",
          "E_fdp_score_aggregation_eb": "E score aggregation, MinMax, mean-normalised EB (CPU)",
          "F_fdp_pass2_basis_rank_search": "F FDP Pass-2 basis + adaptive rank search",
          "G_fdp_materialize": "G FDP checkpoint materialize/save",
          "U_uniform_comparator_basis_rank_search": "Uniform comparator data/calibration + basis + rank search (eb 0.0567)",
          "U_uniform_comparator_materialize": "Uniform comparator materialize/save"}
for s in fdp_st + uni:
    t = T[s]; mk = t.get("marker_offsets_s", {})
    stage_rows.append({"stage": s, "description": labels[s], "chain": "Uniform" if s.startswith("U_") else "FDP-EB", "wall_s": round(t["wall_s"], 2),
                       "wall_min": round(t["wall_s"] / 60, 3), "load_and_tokenize_s": round(mk.get(r"\[per_layer_eb\] loaded") or 0, 1) if "basis" in s else None,
                       "to_first_layer_output_s": round(mk.get(r"\[layer 00\]\[eb=") or 0, 1) if "basis" in s else None,
                       "peak_nvml_gpu_GiB": round(t.get("peak_nvml_gpu_bytes_process_tree", 0) / GiB, 2),
                       "torch_peak_allocated_GiB": round(t.get("torch", {}).get("torch_peak_allocated_bytes", 0) / GiB, 2),
                       "torch_peak_reserved_GiB": round(t.get("torch", {}).get("torch_peak_reserved_bytes", 0) / GiB, 2),
                       "peak_cpu_rss_GiB": round(t.get("peak_cpu_rss_bytes", 0) / GiB, 2),
                       "disk_after_stage_GiB": round(sum((t.get("disk_bytes_after_stage") or {}).values()) / GiB, 2) if t.get("disk_bytes_after_stage") else None,
                       "measurement": "MEASURED"})
(ROOT / "MEDIUM_STAGE_TIMINGS.json").write_text(json.dumps({"stages": stage_rows, "benchmark_wall_s": raw["benchmark_wall_s"]}, indent=1))
mem = {"Uniform": {"peak_nvml_gpu_GiB": peak(uni, "peak_nvml_gpu_bytes_process_tree") / GiB, "torch_peak_allocated_GiB": tpk(uni) / GiB,
                   "torch_peak_reserved_GiB": tpr(uni) / GiB, "peak_cpu_rss_GiB": peak(uni, "peak_cpu_rss_bytes") / GiB},
       "FDP-EB": {"peak_nvml_gpu_GiB": peak(fdp_st[:4] + fdp_st[5:] + ["F_fdp_pass2_basis_rank_search"], "peak_nvml_gpu_bytes_process_tree") / GiB,
                  "torch_peak_allocated_GiB": tpk([s for s in fdp_st if s != "E_fdp_score_aggregation_eb"]) / GiB,
                  "torch_peak_reserved_GiB": tpr([s for s in fdp_st if s != "E_fdp_score_aggregation_eb"]) / GiB,
                  "peak_cpu_rss_GiB": peak(fdp_st, "peak_cpu_rss_bytes") / GiB},
       "per_stage": {r["stage"]: {k: r[k] for k in ("peak_nvml_gpu_GiB", "torch_peak_allocated_GiB", "torch_peak_reserved_GiB", "peak_cpu_rss_GiB", "disk_after_stage_GiB")} for r in stage_rows},
       "gpu_sampling": "NVML per-process usedGpuMemory summed over the stage's own process tree, 0.5 s sampling; not whole-card usage"}
(ROOT / "MEDIUM_MEMORY_PROFILE.json").write_text(json.dumps(mem, indent=1))
summary = {"uniform_total_min": U_tot, "fdp_total_min": F_tot, "fdp_extra_min": F_tot - U_tot, "overhead_ratio": F_tot / U_tot,
           "profiling_min_C_D_E": prof, "profiling_share_of_fdp_total": prof / F_tot, "step1_only_min": m("D_fdp_step1_profiling"),
           "step1_share_of_fdp_total": m("D_fdp_step1_profiling") / F_tot, "pass2_search_and_materialize_min_F_G": search,
           "parent_build_min_B": m("B_parent_uniform_basis_rank_search")}
with open(ROOT / "MEDIUM_OFFLINE_COST.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["Metric", "Uniform", "FDP-EB", "Measurement"])
    w.writerow(["Total offline construction time (min)", f"{U_tot:.1f}", f"{F_tot:.1f}", "MEASURED (sum of measured stages)"])
    w.writerow(["Additional functional profiling time (min)", "-", f"{prof:.1f}", "MEASURED (C+D+E)"])
    w.writerow(["Additional rank allocation/search time (min)", "-", f"{search:.1f}", "MEASURED (F+G)"])
    w.writerow(["Peak GPU memory, NVML process tree (GiB)", f"{mem['Uniform']['peak_nvml_gpu_GiB']:.1f}", f"{mem['FDP-EB']['peak_nvml_gpu_GiB']:.1f}", "MEASURED"])
    w.writerow(["Peak GPU memory, torch allocated (GiB)", f"{mem['Uniform']['torch_peak_allocated_GiB']:.1f}", f"{mem['FDP-EB']['torch_peak_allocated_GiB']:.1f}", "MEASURED"])
    w.writerow(["Peak CPU RSS (GiB)", f"{mem['Uniform']['peak_cpu_rss_GiB']:.1f}", f"{mem['FDP-EB']['peak_cpu_rss_GiB']:.1f}", "MEASURED"])
    w.writerow(["Offline overhead ratio", "1.00x", f"{F_tot / U_tot:.2f}x", "DERIVED from measured totals"])
    w.writerow(["Absolute extra time (min)", "-", f"{F_tot - U_tot:.1f}", "DERIVED"])
    w.writerow(["Profiling share of FDP total", "-", f"{100 * prof / F_tot:.1f}% (Step 1 alone {100 * m('D_fdp_step1_profiling') / F_tot:.1f}%)", "DERIVED"])
(ROOT / "MEDIUM_COST_SUMMARY.json").write_text(json.dumps({"summary": summary, "audit_pass": {k: v["PASS"] for k, v in audit.items()}}, indent=1))
print(json.dumps(summary, indent=1)); print(json.dumps({k: v["PASS"] for k, v in audit.items()}))
for k, v in audit.items(): print(k, {a: (str(b)[:16] if isinstance(b, str) else b) for a, b in v.items()})
for r in stage_rows: print(r)
