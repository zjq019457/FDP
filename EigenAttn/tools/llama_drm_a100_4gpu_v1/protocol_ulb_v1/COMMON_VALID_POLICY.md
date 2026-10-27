# Common-Valid Policy

For a method set S, the global common-valid set is the intersection of sample IDs whose canonical status is `ok` for every method in S. Pairwise common-valid uses the same definition for exactly two methods. OOM, failed, missing, duplicate, and identity-mismatched rows are excluded from the applicable intersection but are never deleted from source artifacts.

Scores are recomputed per task on the retained paired sample IDs with the frozen official metric, then macro-averaged across the 17 tasks. The 3950 samples must never be pooled into a single weighted average. Every export records S, the ordered sample-ID hash per task, retained N, excluded N, and exclusion reasons.
