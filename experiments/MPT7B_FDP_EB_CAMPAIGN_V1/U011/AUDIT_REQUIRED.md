# AUDIT_REQUIRED: U011 resource gate

Quality was not started. No Wiki, C4, J, bootstrap, or zero-shot number exists for this FDP schedule.

The frozen mapping at `epsilon_base = 0.011`, with Layer-0 rank-search epsilon held at `0.011`, realizes:

- Uniform actual B_live: `851,705,856` B
- FDP actual B_live: `889,323,520` B
- Census equals the rank-geometry prediction
- Absolute gap: `37,617,664` B
- Relative gap: `4.4167%`

MPT campaign locks do not preregister a numeric resource tolerance. The Llama enumeration code labels a gap above `1%` as FRONTIER and a gap within `0.15%` as STRICT_MATCH. The executed Llama-3 medium formal cell was about `0.69%` and was still evaluated, so `0.15%` was not a hard abort in that run. A `4.42%` gap is outside both of those labels.

Closing the gap would require changing the global epsilon scalar. That changes the tier operating point. It was not done. q, lambda, the profile, and the rank-search rule were not changed. Formal quality was not used.

Engineering gates that did pass, and can be reused if a resource rule is later locked:

- 12 missing breakpoint layers were measured from the Uniform trace boundary. No rank was extrapolated.
- Parity on layers 0, 15, 16, and 31: relative L2 `0`, cosine `1`.
- Two clean processes produced the same logits SHA `c9b180e6eec9e46e949565401125f1039418845961eedb73b9ca47e68cc29723` and the same realization SHA.
- Cached decode of 16+1 tokens inside the 2048 ALiBi bias passed. A decode that grows past 2048 does not, because the prebuilt bias stops at 2048; that path was not used as a gate.

`MPT_U011_RESOURCE_MATCH_PASS = false`
