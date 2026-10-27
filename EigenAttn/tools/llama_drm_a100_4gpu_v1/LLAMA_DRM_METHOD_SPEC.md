# LLAMA_DRM_METHOD_SPEC — LLAMA_DRM_A100_4GPU_V1

Frozen LLaMA configuration and method invariants for the isolated 4×A100 DRM pipeline.
Nothing in this document is aspirational: every value is enforced by `common.py`,
asserted by `audit/stage0_cpu_audit.py`, and emitted into each result directory as
`protocol/PROTOCOL_IDENTITY.json`.

## 1. Model identity

| Field | Value |
| --- | --- |
| Model | Llama-3-8B (base, not Instruct) |
| Resolved path | first existing of `/mnt/d/jiaqizhang/llama3`, `/data/jiaqizhang/llama3`, `/d/jiaqizhang/llama3`, `/home/bai-project/jiaqizhang/llama3` |
| Revision | `UNKNOWN_LOCAL_SNAPSHOT` |
| Architecture | `LlamaForCausalLM`, 32 layers, 32 heads, 8 KV heads, head_dim 128 |
| `max_position_embeddings` | 8192 |
| Chat template | **none** |

The local snapshot carries no Hugging Face revision metadata. The literal string
`UNKNOWN_LOCAL_SNAPSHOT` is recorded rather than a plausible-looking commit hash,
because a fabricated revision would make the identity record silently wrong.

Because this is a base model with no chat template, `build_chat()` returns the prompt
unchanged. LongBench prompts are already fully formatted, so wrapping them in an
invented template would change the evaluated input.

## 2. Frozen decoding and DRM configuration

| Knob | Value | CLI |
| --- | --- | --- |
| dtype | bfloat16 | `--dtype bf16` |
| batch size | 1 | (runner is single-sample) |
| decoding | greedy | `do_sample=False`, `num_beams=1` |
| attention backend | sdpa | `--backend sdpa` |
| page (block) size | 64 | `--block_size 64` |
| block strategy | fixed | `--block_strategy fixed` |
| token unit | 8 | `--token_unit_size 8` |
| gamma (swap threshold) | 0.9 | `--swap_threshold 0.9` |
| local query window | 4 | `--local_query_window 4` |
| prune layers | 10, 20, 30 | `--prune_layers 10,20,30` |
| retain tokens | 8192, 4096, 2048 | `--retain_tokens 8192,4096,2048` |
| prompt truncation | 7500 | `--max_model_len 7500` |
| rescue ratio (main DRM-Add) | 0.05 | `--rescue_ratio 0.05` |

`sdpa` is the frozen backend because `flash_attn` is not installed in the `llamaqkv`
environment. `--backend auto` and `--backend flash_attention_2` both resolve to `sdpa`
before the model loads, so no worker pays a failed-load retry.

`--max_model_len 7500` leaves 692 tokens of headroom above the largest generation
budget (512 for the summarization tasks), keeping prompt + generation inside Llama-3's
8192-position RoPE range.

## 3. Rescue budget M

Retained pages per prune layer are `retain_tokens / block_size`:

| Prune layer | Retain tokens | Retained pages K | M at ratio 0.05 |
| --- | --- | --- | --- |
| 10 | 8192 | 128 | 6 |
| 20 | 4096 | 64 | 3 |
| 30 | 2048 | 32 | 2 |

M follows directly from the authoritative arithmetic in
`bridgekv/rescuekv_scoring.py::apply_rescue_policy`:

```python
m_rescue = int(round(k_active * float(rescue_ratio)))
if rescue_ratio > 0.0 and k_active > 0 and inactive:
    m_rescue = max(1, m_rescue)
m_rescue = min(m_rescue, len(inactive))
```

M is a function of `(k_active, rescue_ratio)` only. It cannot depend on which score
ranks the candidates, which is what makes the Random control budget-identical to DRM.

## 4. Method CLI mapping (authoritative)

Every arm uses `--method sliminfer_original` except Full-KV, which uses `--method fullkv`.

| Arm | `--method` | `--rescue_policy` | `--rescue_score` | `--rescue_ratio` | Budget |
| --- | --- | --- | --- | --- | --- |
| SlimInfer-Original | `sliminfer_original` | `none` | `delta_o` | — | K |
| SlimInfer-Larger | `sliminfer_original` | `sliminfer_larger` | `delta_o` | (0.05 internal) | K+M |
| Random-Add | `sliminfer_original` | `add` | `random` | 0.05 | K+M |
| DRM-Add | `sliminfer_original` | `add` | `drm` | 0.05 | K+M |
| DRM-Swap | `sliminfer_original` | `budget_neutral_drm_swap` | `drm` | — | K |
| Full-KV | `fullkv` | — | — | — | full |

Random-Add additionally requires an explicit `--seed N`; replicates use seeds 0, 1, 2.
An unseeded Random arm is not auditable, so the seed is never left implicit.

## 5. Method invariants

**I1 — Add arms share one budget.** `add` and `sliminfer_larger` both alias to
`additive_rescue` and report `final_budget_type="K+M"`. SlimInfer-Larger, Random-Add and
DRM-Add therefore retain the same number of pages; only the selection differs.

**I2 — Swap is budget-neutral.** `budget_neutral_drm_swap` drops exactly as many active
pages as it rescues and reports `final_budget_type="K"`, so DRM-Swap consumes the same
memory as SlimInfer-Original.

**I3 — Random differs from DRM in exactly one dimension.** The generated CLI for
`random_add` and `drm_add` differs only in `rescue_score` (plus the mandatory seed).
Same policy, same ratio, same M, same active set A_K, same candidate pool.

**I4 — Random is seeded and reproducible.** `cache_utils` computes random rescue scores
from `cfg["seed"]` through a CPU `torch.Generator`, so a replicate is bit-reproducible
and independent of device or launch order.

**I5 — Random is not a separate code path.** The random branch lives inside the shared
rescue-scoring function alongside `drm`, so both arms traverse identical block layout,
candidate construction and policy application code.

**I6 — Full-KV is an unmodified baseline.** It loads through
`AutoModelForCausalLM`, never through the DRM `LlamaForCausalLM`, so no hook can touch it.

**I7 — CC and ACC families are unreachable.** `bridgekv.cc_topology_policies` and
`bridgekv.acc_sliminfer` export empty method sets and raising dispatchers. This keeps
the authoritative `cache_utils.py` byte-identical while making forbidden methods dead
code that fails loudly if ever reached.

## 6. Evaluation protocol

- 17 official LongBench tasks, 3950 samples total.
- Six-task subset: `qasper`, `hotpotqa`, `2wikimqa`, `musique`, `gov_report`,
  `repobench-p` — 1500 samples.
- Each task is scored independently with the official scorer; the macro average is the
  arithmetic mean over task scores. **Pooled sample averaging is forbidden.**
- Fully offline. `datasets.load_dataset` is removed from the runner: a missing task file
  raises `FileNotFoundError` instead of fetching an unverified dataset revision.

## 7. Compatibility notes

- `transformers` 4.57.3 no longer exports `transformers.utils.LossKwargs`.
  `adapters/transformers_compat.py` restores it, and `modeling_llama.py` also carries a
  local `try/except` stub. SlimInfer/DRM hooks are untouched.
- `utils/__init__.py` tolerates a missing `matplotlib` (needed only for NIAH plots) so
  that `import utils.cache_utils` works in the `llamaqkv` environment.
- Nine `bridgekv` modules and `tools/qwen_pipeline_common.py` were copied verbatim from
  the authoritative bundle to satisfy `cache_utils.py` imports.
- `transformers` 4.55+ requires `Cache.__init__` to receive exactly one of `layers` or
  `layer_class_to_replicate`. The authoritative `SlimInferCache` calls `super().__init__()`
  with no arguments, so the compat layer patches `Cache.__init__` to default to an empty
  layer list rather than editing the authoritative cache source.

### 7.1 SDPA prefill causality (required correctness fix)

The authoritative bundle was validated with `flash_attn` 2.7.4 on `transformers` 4.53.0.
This environment has no `flash_attn`, so the frozen backend is `sdpa` — a path the
upstream code never exercised end-to-end, and it was **silently wrong**:

- SlimInfer's prefill deliberately sets `current_attention_mask = None`, because the
  `flash_attention_2` branch supplies causality separately via `is_causal=True`.
- The `sdpa` branch passed that `None` mask straight through together with a hardcoded
  `is_causal=False`. With no mask and no causal flag, PyTorch computes **bidirectional**
  attention, so during prefill every prompt token attended to its own future.
- Symptom: the model still ran with no error, but generation collapsed into degenerate
  repetition and **every SlimInfer/DRM arm scored 0.0** while Full-KV scored normally.
  Nothing short of a scored run exposed it.

Fix in `runtime/models/modeling_llama.py`: the `sdpa` branch derives the flag from mask
presence — a `None` mask means prefill and uses `is_causal=True`. Prefill queries and the
KV pool cover the same ordered token set (verified `q_len == kv_len`, e.g. 4000×4000 and
2016×2016 at the last prune layer), so SDPA's top-left aligned causal mask is exactly the
standard causal mask. This reproduces the validated flash-attention-2 semantics rather
than introducing new behaviour; a length mismatch raises instead of mis-aligning.

Measured on `qasper`, 10 samples, after the fix (before: all DRM arms 0.0):

| Arm | Score |
| --- | --- |
| Full-KV | 14.77 |
| SlimInfer Original | 16.04 |
| DRM-Add | 16.11 |
| DRM-Swap | 16.81 |

Stage 0 pins the invariant (`B_compat` group): it asserts the flag is derived rather than
hardcoded, that a `None` mask maps to `is_causal=True`, that length mismatch fails loudly,
and it numerically proves `is_causal=True` equals an explicit additive causal mask while
unmasked SDPA differs.

## 8. Storage

The repository volume (`/`) is effectively full, so results are pinned to
`/diskF/jiaqizhang/llama_drm_a100_4gpu_v1_results` via `RESULTS_ROOT.txt`
(override with `LLAMA_DRM_RESULTS_ROOT`). Stage 0 fails if the results volume has less
than 20 GiB free, so the run refuses to start rather than dying mid-sweep on ENOSPC.
