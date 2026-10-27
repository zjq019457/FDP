# Llama-3 Layer-0 Epsilon Semantics Audit

Status: read-only replay; no MPT schedule or quality was started.

The frozen replay gives `epsilon_raw[0] = 0.06` and `epsilon_final[0] = 0.0655374012229054`. The global normalization factor is `1.42348161736466`. The provenance value `0.0655374` is the post-normalization/effective Layer-0 value produced by the rho interpolation, while the lock phrase “Layer 0 epsilon = e” refers to the raw/base epsilon before normalization.

`rho=0.75` is used only in the Layer-0 geometric interpolation between base `e` and the globally normalized Layer-0 epsilon. It is not used to construct q, the global normalization scalar, or rank projection. The frozen rank SHA is reproduced exactly from the frozen rank schedule.

## Verdict

- `NO_METHOD_CONFLICT`
- `LAYER0_LOCK_REFERS_TO_PRE_NORMALIZATION_EPSILON`
- MPT remains unauthorized: `MPT_FDP_SCHEDULE_AUTHORIZED=false`.

The MPT transfer rule is therefore: apply the raw Layer-0 rule first, then let Layer 0 participate in the same global normalization/effective-epsilon procedure; do not post-normalization clamp epsilon_0 back to e.

Provenance: allocator source SHA `3799526d7e0ff0e05df6d7e5f2c267e9f09b65d81551ca622985e4a4cf181a6e`, method-lock SHA `8a8e95ae4df92203b5ad9afdbec4ed244004b57750e81a41b2f24bc5b880df4f`, frozen rank SHA `f0c462759f53f77214eb7fdce65ea311b5890fd0cd7c6c3eaa061f4f07bf211b`.
