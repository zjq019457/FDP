# PaLu official-protocol audit

## Authority

- Paper: PaLu, ICLR 2025, Table 1/Table 8 and Appendix G.
- Repository: pinned commit `bb22666e2ef96707e8dd21d93fc00146c2e0d615`.
- Official conversion: `compress.py` -> `compress_model` -> `HeadwiseLowRankModule.from_linear_whiten`.

## Resolved protocol

- Model: Llama-2-7B HF pretrained.
- G-LRD group size: 4.
- Llama-2-7B has 32 attention/KV heads, therefore 8 native groups per K/V projection, each 512 output dimensions.
- Repository ratio is retained native rank fraction: 0.7/0.5/0.3 correspond to 30%/50%/70% compression.
- Fisher search: 2048 WikiText-2 samples, sequence length 1024.
- Whitening: 256 WikiText-2 samples, sequence length 2048.
- Quality: WikiText-2 test and C4 validation, sequence length 2048; C4 scans 1100 documents and uses 256 chunks.

## Root cause of the old mismatch

The old anchor-schema branch uses two rank slots per K/V projection, while official Llama-2 G-LRD uses eight groups per K/V projection. It also bypasses the official `compress_model` factorization path and uses a 32-sample anchor calibration for its schedule. Uniform therefore diverges even before FDP is applied.

The old branch is preserved and is not overwritten. Its values remain valid only as an internal anchor-schema comparison.

## Engineering repair

The official runner initially failed because the optional `fast_hadamard_transform_cuda` extension was unavailable. A pure-Python import fallback is used only for the non-quantized FP16 path, where the Hadamard routine is not invoked. Quantized results are outside this branch.

## Gate order

1. Full-rank identity.
2. Official Uniform at retained fractions 0.7/0.5/0.3.
3. Official Fisher/Automatic at the same fractions.
4. Reproduction classification.
5. Official-native FDP exchange calibration and formal matrix.
