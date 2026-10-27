# Bootstrap Policy

- Use pairwise common-valid samples and a fixed seed of 0.
- Run 10,000 paired bootstrap replicates.
- Within each task, resample paired sample IDs with replacement to that task's common-valid N; both methods use the identical sampled IDs and multiplicities.
- Recompute the frozen official task metric from sampled predictions and references, then compute the arithmetic mean of all 17 task scores for every replicate.
- Report observed macro delta, bootstrap mean delta, 2.5/97.5 percentile CI, P(delta > 0), seed, per-task N, and total common-valid N.
- Do not bootstrap pooled per-sample metric values across tasks.
