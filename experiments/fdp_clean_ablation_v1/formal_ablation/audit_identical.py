
import hashlib, json
from pathlib import Path

p = Path('/diskF/jiaqizhang/fdp_clean_ablation_v1/formal_ablation/CORE_FORMAL_PER_WINDOW')

print("=== CHECK WIKI_WINDOWS MD5 AND SIZES ===")
targets = ['b80472cc', '76bd0eb1', '31635586', 'fa4831cd', 'fdf1ef67']
for pref in targets:
    matches = list(p.glob(f"{pref}*"))
    if matches:
        d = matches[0]
        w = d / "wiki_windows.jsonl"
        if w.exists():
            h = hashlib.md5(w.read_bytes()).hexdigest()
            print(f"{pref}: md5={h} size={w.stat().st_size} mtime={w.stat().st_mtime}")
        else:
            print(f"{pref}: wiki_windows.jsonl does not exist")

print("\n=== CHECK COMPANION DISKF LOG FOR HOW WIKI PPL WAS RUN ===")
log_f = Path('/diskF/jiaqizhang/fdp_clean_ablation_v1/formal_ablation/companion_diskf.log')
if log_f.exists():
    for line in log_f.read_text().splitlines():
        if "Starting" in line or "Wiki" in line or "FAILED" in line:
            print(line)
