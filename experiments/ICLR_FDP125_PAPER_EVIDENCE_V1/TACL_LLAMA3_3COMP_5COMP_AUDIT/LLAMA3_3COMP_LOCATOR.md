# Llama-3 3-component locator

- Exact explicit subset: `11001` = `{d1,d2,d5}`.
- Search artifact: `/diskF/jiaqizhang/fdp_component_subset_search_v2/results/medium/11001/`.
- Provenance SHA256: `f63af76b4259c35af36351e4fa289cdf18ebf6c44edffb5f04f132f14e0f93da`.
- Rank file SHA256: `3800ab0b3a3cefe076db21bd5be47515ec3583a21eee811985f57758ddd6f512`; canonical rank SHA `f0c462759f53f77214eb7fdce65ea311b5890fd0cd7c6c3eaa061f4f07bf211b`.
- Basis report SHA256: `bc91269c1deed44ab1fbf973e66e90417145c327b5a6db5d679a1c69eb3c585e`.
- Raw Pass1 profile SHA256: `d0d5d7653caae9c272d97ae38ef4dd0c3718573bde3250df650c028ca828dba4`. The raw profile is present; the referenced compressed Pass1 model directory is absent, so the profile observation was not reacquired.
- Subset search quality: WikiText-2 PPL 7.1594697725; C4 PPL 11.9987105178; `J=2.2266175556`. Classification: `DIAGNOSTIC / POST_HOC_SELECTED` relative to the earlier five-component freeze.
- Current formal Llama-3 Medium cell lock SHA256 `29fbf129266e09da0f08303fc4ea32f2a5291b20e1af020f25a07ec96c802cd6` uses the same rank SHA; it identifies `meta-llama/Meta-Llama-3-8B`, not Llama-3.1. Current-cell results are reported separately in the provenance JSON.
