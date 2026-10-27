FDP_PROFILE_ROBUSTNESS_V1 (registered 2026-10-13T00:06:46-07:00, before any profile is read)
Question (user-approved 2026-10-13): is the FDP profile robust to the probe inputs? The paper profile uses two fixed passages (83/97 tokens, EOS-padded to 512).
Rebuild the 0.74x EigenAttention model (eps_base 0.06, frozen C45 builder with official_min_thresh 0, same calibration/materializer); gate: ranks == frozen fdp125 medium uniform low_rank_config.
Profiles: (a) original two-passage probe (reproduction check vs frozen step1.json); (b) 8 WikiText-2 train passages x 512 tokens, no padding.
Analysis (CPU): Spearman/top-k overlap of q_l vs frozen q_l; replay of the FDP-EB schedule (lambda 1.2, mean-normalised, layer0=base) on the frozen no-floor curves; rank changes and B_live. No quality evaluation; the paper schedule is not changed.
