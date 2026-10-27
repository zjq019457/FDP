# Stage R1–R3 Execution Plan

- generated_at: `2026-07-28T08:37:06.490231-07:00`

## R1 Dense (4-GPU parallel)
- GPU0: WikiText-2 PPL
- GPU1: C4 PPL
- GPU2: HellaSwag, ARC-Challenge, ARC-Easy
- GPU3: WinoGrande, PIQA
- Aggregate after all DONE → DENSE_PARITY_GATE

## R2 Uniform
- Build/audit single GPU (default 0)
- Eval split like R1 across 4 GPUs

## R3 Legacy FDP
- Step1 single GPU → per-layer EB CPU → adaptive basis single GPU → eval 4-GPU

## Forbidden until parity
- Qwen/OPT/8K/operating-points/KQ-V/robustness/matched-controls

