# PaLu Official Native Action Space

Status: FROZEN_AUDIT

## Official semantics

- Model: Llama-2-7B, 32 layers, 32 attention heads, 32 KV heads, head dimension 128.
- G-LRD group size: 4 contiguous KV heads. Each K/V projection therefore has 8 groups, each with output dimension 512.
- A native unit is (layer, K-or-V module, group_id). There are 512 units per tier.
- K and V ranks are separate in the official rank_search.py; no K:V equality constraint is imposed.
- Group boundaries are preserved. A unit can move only in 32-rank quanta.
- Legal rank range: 32 through 512 inclusive.

## Memory formulas

For one group rank r, the official factorization creates a VT contribution of r x 4096 and a U contribution of 512 x r, so factor elements are 4608r. At FP16 this is 9216r bytes. One 32-rank action therefore changes factor storage by 294,912 bytes.

The latent K/V cache accounting is sum(K/V group ranks) x sequence_length x dtype_bytes; at sequence length 2048 and FP16, one 32-rank unit is 131,072 cache bytes. Official result files report realized prefill and decode cache bytes from this same K/V latent allocation.

## q mapping

Frozen q is replicated unchanged to both K and V and all 8 contiguous groups. No old anchor mapping, interpolation, averaging, or quality feedback is used.

## phi decision

The old phi is VALID_FOR_OLD_ANCHOR_SCHEMA_ONLY: the old branch has 2 slots/module and 65,536 total capacity, while the official branch has 8 groups/module and 262,144 full capacity. Therefore PALU_OFFICIAL_NATIVE_MARGINAL_V1 must profile new official units before formal FDP.
