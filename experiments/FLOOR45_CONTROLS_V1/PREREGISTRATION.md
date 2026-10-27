FLOOR45_CONTROLS_V1  (registered 2026-10-12T14:22:25-07:00, before any build or quality read)
User request 2026-10-12: (A) official EigenAttention Uniform with vs without the 45% rank floor at the same resource;
(B) Spectral-tail under the 45% floor.
Source (read-only): FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1 (frozen C45 builder sha 601bf0fd..., same calibration, materializer, evaluators).
A: uniform_nofloor_<tier> = official_min_thresh 0, single EB = no-floor lattice point nearest floored Uniform B (all MATCHED <=0.124%).
B: spectral_tail45_<tier> = main-table Spectral-tail score at floored Uniform ranks, FDP-EB mapping, floor on, matched to floored FDP-EB (all MATCHED <=0.06%).
Selection used ranks/bytes only (PLAN.json, quality_scores_read=false). Gates: builder ranks == CPU replay; window/item counts.
Evaluation: WikiText-2/C4 PPL (141/256 windows), 5-task ZS, LongBench QA-6 at aggressive. Paired comparisons: uniform_nofloor vs floored Uniform; spectral_tail45 vs floored FDP-EB and floored Uniform.
Checkpoints deleted only after all evaluations pass, with per-file size+sha256 manifest; low_rank_config kept.
AMENDMENT 1 (2026-10-12T14:26:00-07:00): uniform_nofloor arms evaluate WikiText-2/C4 PPL only (user request). No quality read before this change.
