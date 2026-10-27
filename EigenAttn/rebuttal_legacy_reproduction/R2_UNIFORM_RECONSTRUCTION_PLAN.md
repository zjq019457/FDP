# R2 Uniform Reconstruction Plan

- generated_at: `2026-07-28T09:43:17.592154-07:00`
- Historical `final_model` missing → rebuild via legacy compress; basis still present → hash+copy into isolation.

## Basis
- source: `/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_eb/eigen_basis_llama3_8b_wikitext2_eb0p045.pt`
- isolated: `/home/bai-project/jiaqizhang/EigenAttn/rebuttal_legacy_reproduction/artifacts/r2_uniform_eb045/eigen_basis_llama3_8b_wikitext2_eb0p045.pt`
- sha256: `1186efc90a38ef353fdd6c0c05d254ee74c2d9acd8c50e74aa0ed8dac5eddee8`

## Compress (main_eigen_attn.py)

```bash
CUDA_VISIBLE_DEVICES=0 /home/bai-project/anaconda3/envs/eigen/bin/python -u /home/bai-project/jiaqizhang/EigenAttn/EigenAttn/main_eigen_attn.py \
  --model /diskD/jiaqizhang/Llama-3 --net Llama-3-8b --dtype bf16 \
  --error_budget 0.045 --avg_dim 4096 --seqlen 2048 --nsamples 64 --seed 2 \
  --calib_dataset c4 --compress_only --attn_implementation eager \
  --eigen_basis_path /home/bai-project/jiaqizhang/EigenAttn/rebuttal_legacy_reproduction/artifacts/r2_uniform_eb045/eigen_basis_llama3_8b_wikitext2_eb0p045.pt \
  --save_dir /home/bai-project/jiaqizhang/EigenAttn/rebuttal_legacy_reproduction/artifacts/r2_uniform_eb045/final_model \
  --output_dir /home/bai-project/jiaqizhang/EigenAttn/rebuttal_legacy_reproduction/artifacts/r2_uniform_eb045/compress_run \
  --cache_dir /home/bai-project/jiaqizhang/EigenAttn/EigenAttn/HF_cache
```

## Eval
- After runtime smoke: 4-GPU PPL/zero-shot using instrumented runners with --cmp final_model.

