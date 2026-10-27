import csv
import json
from pathlib import Path


central = Path('/diskE/jiaqizhang/ICLR_HETEROGENEOUS_KVCACHE_FORMAL_V1')
path = central / 'ICLR_CROSS_COMPRESSOR_MASTER.csv'
rows = list(csv.DictReader(path.open(newline='', encoding='utf-8')))
fields = rows[0].keys()
rows = [r for r in rows if r.get('Compressor') not in {'SnapKV', 'xKV'}]


def row(**kw):
    d = {f: '' for f in fields}
    d.update(kw)
    return d


snap_source = '/diskF/jiaqizhang/FDP_NATIVE_TRANSFER_V2_WORK/FINAL_FDP_CORE_V2/reports/SNAPKV_MASTER_STATUS.csv'
for tier, j0, j1, dj, rank, pre, dec, cls in [
    ('LIGHT cached-AR', 1.8704675183904493, 1.8702847124329138, -0.00018280595753571464, 98304, 1610612736, 1611137024, 'NEUTRAL_INCONCLUSIVE'),
    ('MEDIUM cached-AR', 1.8767179414976303, 1.8758506582148953, -0.0008672832827346811, 65536, 1073741824, 1074266112, 'NEUTRAL_INCONCLUSIVE'),
    ('AGGRESSIVE cached-AR', 1.8992336000406942, 1.8960952665450015, -0.0031383334956926567, 32768, 536870912, 537395200, 'VALID_POSITIVE'),
]:
    rows.append(row(
        Compressor='SnapKV', **{'Base model': 'external SnapKV formal root'},
        **{'Native resource axis': 'retained-token quota', 'Native allocation unit': 'retained tokens', 'Official/native baseline': 'Cached-AR Native Control', 'Native baseline importance signal': 'native selector', 'Transferable FDP signal': 'frozen q', 'Native functional signal': 'frozen SnapKV native signal', 'Legal native action': 'quota transfer', 'Representative budget': tier, 'Actual baseline resource': rank, 'Actual reallocated resource': rank, 'Budget delta': 0, 'Budget matched?': 'EXACT', 'Primary benchmark': 'paired cached-AR Wiki/C4', 'Official baseline quality': j0, 'FDP Prior quality': '', 'Native Phi quality': j1, 'Delta Prior vs baseline': '', 'Delta Native vs baseline': dj, 'Random-control delta': '', 'Profile overhead': 'external report', 'Calibration overhead': 'external report', 'Inference overhead': 'external report', 'Official reproduction status': 'VALID_EXTERNAL_REPORT', 'Formal evidence status': cls, 'Provenance': snap_source}
    ))

xkv_source = '/diskF/jiaqizhang/xkv_native_runtime_fixed_v1/reports/XKV_FIXED_MEDIUM_RESULTS.csv'
rows.append(row(
    Compressor='xKV', **{'Base model': 'Llama-3.1 external xKV root'},
    **{'Native resource axis': 'group K/V SVD rank capacity', 'Native allocation unit': '8 native four-layer groups', 'Official/native baseline': 'Native Control', 'Native baseline importance signal': 'xKV native group scorer', 'Transferable FDP signal': 'frozen FDP q', 'Native functional signal': 'xKV native marginal', 'Legal native action': 'group SVD capacity exchange', 'Representative budget': 'MEDIUM', 'Actual baseline resource': 7680, 'Actual reallocated resource': 7680, 'Budget delta': 0, 'Budget matched?': 'EXACT', 'Primary benchmark': 'paired cached-AR Wiki/C4', 'Official baseline quality': 2.1214695880792465, 'FDP Prior quality': '', 'Native Phi quality': 2.114496913273961, 'Delta Prior vs baseline': '', 'Delta Native vs baseline': -0.006972674805285628, 'Random-control delta': '', 'Profile overhead': 'external report', 'Calibration overhead': 'external report', 'Inference overhead': 'external report', 'Official reproduction status': 'VALID_EXTERNAL_REPORT', 'Formal evidence status': 'XKV_FIXED_VALID_POSITIVE_NONCOLLAPSED', 'Provenance': xkv_source}
))

with path.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(fields)); w.writeheader(); w.writerows(rows)
(central / 'ICLR_CROSS_COMPRESSOR_MASTER.md').write_text('# ICLR cross-compressor master\n\nRows are imported from each compressor\'s own frozen/native baseline report. Raw PPL/J values are not compared across different models as a universal leaderboard; the meaningful quantity is each compressor\'s within-model delta at matched native resource.\n\nPaLu: 12-cell intrinsic formal matrix complete; ratio-0.5 zero-shot is externally blocked by missing lm-eval datasets. SnapKV: cached-AR three-tier rows imported with frozen classifications. xKV: MEDIUM runtime-fixed paired cached-AR row imported.\n')
state_path = Path('/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1/PALU_PIPELINE_STATE.json')
state = json.loads(state_path.read_text())
state['cross_compressor_master'] = 'updated with PaLu, SnapKV cached-AR, and xKV MEDIUM valid report rows'
state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
