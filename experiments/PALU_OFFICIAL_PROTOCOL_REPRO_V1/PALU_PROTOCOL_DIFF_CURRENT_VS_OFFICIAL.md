# Current anchor-schema vs official PaLu protocol

| Item | Current branch | Official branch | Status |
|---|---|---|---|
| Llama-2 G-LRD groups per K/V projection | 2 slots | 8 groups (group size 4) | MISMATCH |
| Factorization | precomputed anchor factor cache + manual module build | official whiten decomposition | MISMATCH |
| Fisher calibration | 32 samples | 2048 samples, seqlen 1024 | MISMATCH |
| Native action space for FDP | 2-slot schedule | official 8-group schedule | PENDING PORT |
| Quantized Hadamard extension | unavailable | unused by FP16 path | ENGINEERING FALLBACK |

No current result is overwritten.
