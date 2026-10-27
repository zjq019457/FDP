# OOM Policy

1. Batch size is 1 and methods run sequentially on one GPU.
2. A CUDA OOM is retried once only after discarding the partially written cache, garbage collection, and `torch.cuda.empty_cache()`.
3. A second OOM writes prediction `[OOM]`, status `oom`, the exact sample identity, and the exception metadata, then continues to the next sample.
4. `failed` is reserved for non-OOM per-sample exceptions. `missing` means no canonical row exists at finalization. Neither may be relabeled as OOM.
5. Raw full-set scoring assigns zero contribution to `oom`, `failed`, and `missing`, while preserving each status separately in manifests.
6. Raw historical predictions are immutable. Offload fills are separate artifacts and never replace raw predictions or efficiency results.
