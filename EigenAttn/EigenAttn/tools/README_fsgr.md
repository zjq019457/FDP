# FSGR Offline Basis Rebuild

`rebuild_eigen_basis_fsgr.py` rebuilds an existing `eigen_basis.pt` into a structured basis file:
- segment-shared basis (default on K/V)
- fisher-like rank allocation from calibration activations
- optional orthogonal Procrustes alignment per layer

## Method Summary

FSGR = Fisher-guided Shared-Basis Reconstruction:
1. Compute layer-wise subspace similarity `S_ij = ||U_i^T U_j||_F^2 / r`.
2. Segment layers (`contiguous` default, optional `global`, `none`).
3. Build segment master basis from mean projection operator (via low-rank PCA).
4. Allocate effective rank (`uniform` / `energy` / `fisher`).
5. Optional orthogonal Procrustes alignment for each layer within shared segment.
6. Write compatible output keys:
   - `...q_proj_basis`
   - `...k_proj_basis`
   - `...v_proj_basis`

## Default Stable Strategy

- `share_targets=kv` (Q not shared)
- `cluster_mode=contiguous`, `num_segments=3`
- `rank_budget_mode=fisher`
- `orth_align=1`

## Run Command

```bash
python tools/rebuild_eigen_basis_fsgr.py \
  --model /diskE/opt-30b \
  --basis_in qkv_outputs/eigen_basis.pt \
  --basis_out qkv_outputs/eigen_basis_fsgr.pt \
  --calib_dataset c4 \
  --calib_data_dir /diskE/jiaqizhang/c4-en/en \
  --nsamples 64 \
  --seqlen 256 \
  --max_rank 5712 \
  --cluster_mode contiguous \
  --num_segments 3 \
  --share_targets kv \
  --rank_budget_mode fisher \
  --orth_align 1 \
  --dtype fp32 \
  --seed 2
```

## Dry Run

Only analysis, no rebuilt basis write:

```bash
python tools/rebuild_eigen_basis_fsgr.py \
  --model /diskE/opt-30b \
  --basis_in qkv_outputs/eigen_basis.pt \
  --basis_out qkv_outputs/eigen_basis_fsgr.pt \
  --calib_dataset c4 \
  --calib_data_dir /diskE/jiaqizhang/c4-en/en \
  --rank_budget_mode fisher \
  --dry_run 1
```

## Compare Mode

Compare statistics between two basis files:

```bash
python tools/rebuild_eigen_basis_fsgr.py \
  --model /diskE/opt-30b \
  --basis_in qkv_outputs/eigen_basis.pt \
  --basis_out qkv_outputs/eigen_basis_fsgr.pt \
  --calib_dataset c4 \
  --calib_data_dir /diskE/jiaqizhang/c4-en/en \
  --rank_budget_mode fisher \
  --dry_run 1 \
  --compare_basis qkv_outputs/eigen_basis_fsgr.pt
```

## Output Structure

Given `basis_out = qkv_outputs/eigen_basis_fsgr.pt`, script writes:

- `qkv_outputs/eigen_basis_fsgr.pt`
- `qkv_outputs/eigen_basis_fsgr.analysis/`
  - `basis_integrity_summary.json`
  - `similarity_q.pt`, `similarity_q.csv`
  - `similarity_k.pt`, `similarity_k.csv`
  - `similarity_v.pt`, `similarity_v.csv`
  - `rank_alloc_q.json`
  - `rank_alloc_k.json`
  - `rank_alloc_v.json`
  - `shared_k_segment0.pt` ... (if shared)
  - `shared_v_segment0.pt` ... (if shared)
  - `summary.json`
  - `compare_basis_stats.json` (if `--compare_basis` provided)

## Notes

- Fisher-like score is implemented as a stable proxy using calibration activations:
  - collect per-layer input second moment `E[x^2]`
  - project into basis directions by `(U^2)^T E[x^2]`
- This avoids unstable/expensive full gradient collection and is suitable for offline rebuild.
- CPU-first design; use `--fisher_device cuda:0` if you want faster collection.

