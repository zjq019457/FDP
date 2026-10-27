# Layer-0 Epsilon Semantics Audit

状态：只读重放。没有生成 MPT FDP schedule，没有跑 Wiki/C4/zero-shot，没有修改历史 method lock 或 Llama-3 provenance。

## 结论

`0.06` 和 `0.06553740122290536` 是同一条正式执行链上的两个阶段，不是两套方法。

正式 Llama-3 medium FDP-EB 的执行顺序是：

1. `q` 为 raw `d1+d2+d5` 求和后再做跨层 MinMax。本 profile 上 `q_0 = 0`。
2. `epsilon_raw_l = 0.06 / (1 + 1.2 * q_l)`。因此 `epsilon_raw_0 = 0.06`。
3. 全部 32 层，包括 Layer 0，做 global mean normalization。因子是 `1.42348161736466`。Layer 0 变成 `0.08540889704187958`。
4. `derive_eb` 只对 Layer 0 做 `rho=0.75` 几何插值，得到 `0.06553740122290536`。这个数被写入 provenance 的 `eb_vector`。
5. canonical builder 以 `--freeze_layer0_eb --layer0_eb 0.06` 启动，在 rank search 之前把 Layer 0 设回 `0.06`。

因此 lock 里的 “effective epsilon_0 exactly to e after intermediate schedule generation” 指的是第 5 步的 rank-search epsilon。`0.0655374` 是第 4 步的中间量，不是 rank search 使用的 epsilon。

判定：`NO_METHOD_CONFLICT`。

MPT 要复现的是这套实际执行语义：mean normalization 之后，rank-search 的 Layer-0 epsilon 是该档的 base `e`。不要把 `0.0655374` 当作 MPT 的搜索 epsilon，也不要为了 MPT 改 lambda、q 或 component。

## 三个问题

A. lock 的 “Layer 0 epsilon = e” 指 rank search 的 effective epsilon，也就是 builder override 之后的值。它不是 `epsilon_raw`，也不是 provenance 里保存的数。

B. `0.06553740122290536` 是 mean normalization 之后、rho 插值之后、builder override 之前的值。用正式 campaign step1 重放时，整条 stored schedule 与 provenance 的最大绝对误差是 `2.78e-17`，round-8 `eb_sha` 都是 `2a8f73b4f5d550be9052b0bce8f66dfcd720087d2872f1c412dbb559c5c9eaee`。

C. `rho=0.75` 只出现在 `derive_eb` 的 Layer-0 插值：

`eps[0] = exp(0.75*log(e) + 0.25*log(epsilon_mean_normalized[0]))`

它不进入 q、normalization 因子或 rank projection。builder 随后丢掉这个 Layer-0 结果。sibling parent run 的日志是 `layer0 EB 0.065237 -> 0.060000`。成功的 fdp125 basis 命令同样带 `--freeze_layer0_eb --layer0_eb 0.06`；该次日志随 scratch 删除，命令留在 provenance 里。builder 文件时间是 2026-07-31，早于 2026-09-16 的正式运行，SHA 与 method lock 一致。

## 被拒绝的解释

“Layer 0 参加 mean normalization 后，最终 epsilon 就是 0.0655374，因此不能再 clamp 回 e” 不成立。只做 mean normalization 时 Layer 0 是 `0.08540889704187958`。`0.0655374` 是其后的 rho 插值，而正式 builder 又把它设回 `0.06`。

先前一份 Llama 审计把 rho 中间量标成 `epsilon_final`，并认为 lock 只是在说 pre-normalization。该文件未改。它没有计入 provenance 命令里实际请求的 builder override。

## Rank SHA

磁盘上的 formal rank 文件 SHA 是 `f0c462759f53f77214eb7fdce65ea311b5890fd0cd7c6c3eaa061f4f07bf211b`，与 method lock 的 replay anchor 相同。Layer 0 的存档 rank 是 `[808, 992]`。

这次没有重跑 GPU basis builder。成功的 threshold trace 已随 basis 目录删除，所以 `FORMAL_LLAMA3_RANK_SHA_REPRODUCED = false`。这是 artifact identity，不是一次新的 rank replay。

另外，method lock 指向的 clean-ablation step1（`d0d5d765…`）不能精确重放 provenance。能精确重放的是正式 resume 实际读取的 campaign step1（`67780d18…`）。历史 lock 未改；跨模型 reproduction 以实际执行的 campaign step1 和 builder 命令为准。

## Profile 是否按档位重做

Llama worker 对每个 tier 先构建该档 Uniform，再对该 Uniform 跑 2×512 step1，然后 `derive_eb`。q 是 operating-point-specific 的。MPT 不能把 U011 profile 套到 U015 或 U019。

U011 已有 profile 可以复用，不重跑：

- 文件 SHA `43fcd1ad06b25e81c20015975dbc71944a65e7c12d1864adf3be6a84dc7cf4fd`
- 32 层，`Z_rel_mse`、`score_rel_mse_causal`、`entropy_diff_A` 均有限
- 2 条固定 prompt，序列长度 512
- `quality_scoring = false`
- attempt 1 是 model load 之前的字段名错误，日志保留

## 为什么还没有写出 U011 FDP schedule

授权已经放开，但 schedule 还不能诚实生成。U011 Uniform 的 rank trace 在 `epsilon=0.011` 的第一次失败处停止。按冻结映射，32 层里有 12 层的 rank-search epsilon 比该失败误差更松，已保存的 trace 里没有对应 legal breakpoint。没有外推 rank，也没有用手改 schedule。

下一步是沿已保存的 MPT basis，对缺段层继续 frozen threshold descent，再用 global scalar 把 actual `B_live` 匹配到 `851,705,856` B，容差 `0.15%`。这一步仍然 quality-blind。匹配完成之前不跑 formal PPL。

## Flags

```
RAW_EPS0_EQUALS_E = true
FINAL_EPS0_EQUALS_E = true
FINAL_EPS0_EQUALS_0P0655374 = false
FORMAL_LLAMA3_RANK_SHA_REPRODUCED = false
LOCK_TEXT_IS_PRENORM_SEMANTICS = false
REAL_METHOD_CONFLICT = false
MPT_FDP_SCHEDULE_AUTHORIZED = true
```

`epsilon_final` 在 CSV 中表示 builder override 之后、进入 rank search 的值。`epsilon_after_rho_pre_override` 才是 provenance 中的 `0.06553740122290536`。
