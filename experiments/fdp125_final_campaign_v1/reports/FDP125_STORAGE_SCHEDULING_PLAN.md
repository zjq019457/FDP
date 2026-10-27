# FDP-125 存储调度规划

更新时间：2026-09-13 13:14 PDT

## 当前容量

| 盘 | 总容量 | 已用 | 当前可用 | 本轮用途 |
|---|---:|---:|---:|---|
| D | 457.4 GiB | 341.2 GiB | 92.96 GiB | 只读原始模型与数据集；禁止本轮写入 |
| E | 457.4 GiB | 329.2 GiB | 104.84 GiB | basis、压缩模型和临时分片 |
| F | 457.4 GiB | 398.2 GiB | 35.92 GiB | 小型结果、日志、哈希、rank/EB、per-window NLL |

当前 FDP-125 scratch 约 26 KiB，F 盘 campaign compact root 约 3.9 MiB。Aggressive 正在 basis 阶段；Medium 的首次启动已因外部 SIGTERM 退出且 scratch 已回收。

## 单个 Qwen/Llama worker 的空间生命周期

1. `basis`：E 盘逐步增加，预计约 15–17 GiB。
2. `materialize`：basis 与压缩模型短暂共存，单 worker 峰值按 32 GiB 规划。
3. materialize 验证成功后立刻删除 basis，降至约 15–17 GiB。
4. Wiki/C4/zero-shot 读取压缩模型；per-window NLL 原子写入 F。
5. DONE 标记验证后删除压缩模型、分片和 state scratch，E 盘回到基线。

规划值采用 32 GiB/worker；在取得逐阶段遥测前，不把估计值伪装为实测精确峰值。

## 水位与并发规则

- E 盘硬停水位：15 GiB 可用；任何新 heavy stage 都不得越过。
- E 盘调度保留：每个进入 heavy stage 的 worker 预留 32 GiB，再保留全局 15 GiB。
- 当前 E 盘 104.84 GiB 可支持：
  - 2 个同时处于峰值的 Qwen/Llama worker：预计峰值占用约 64 GiB，峰值时仍余约 40.84 GiB。
  - 理论上 3 个为 96 GiB、仅余约 8.84 GiB，低于 15 GiB 水位，因此禁止。
- GPU evaluation 阶段若 basis 已删除，可以与另一个 worker 的 basis/materialization 重叠。
- 启动判定必须使用全局 reservation，而不能让多个 worker 各自只检查 `free >= 47 GiB`，否则存在并发竞态。

## 当前及近期排程

| 顺序 | GPU/任务 | 状态 | E 盘策略 |
|---|---|---|---|
| 1 | GPU3 Llama Aggressive | RUNNING | 独占当前 heavy-stage reservation；完成每个方法后滚动删除 |
| 2 | Llama Medium retry | QUEUED | Aggressive 离开 basis/materialize 峰值后重启；失败 scratch 不复用 |
| 3 | Qwen Light core | QUEUED | 与处于 evaluation 的 Llama 可重叠；最多保持两个 heavy reservations |
| 4 | Qwen Medium/Aggressive | QUEUED | 单档完成即清理，不预物化下一档 |
| 5 | OPT-30B | LATER | 先实测一个 worker 峰值；未获得峰值前只允许一个 materialization |

## OPT-30B 规则

- 不沿用 Qwen/Llama 的 32 GiB 假设。
- 第一个 OPT state 单 worker运行，并记录 basis 峰值、basis+model 共存峰值和清理后基线。
- 只有满足 `free >= N * measured_peak + 15 GiB` 时，才允许 N 个 OPT heavy stage 并发。
- basis hash/provenance 完成后立即删 basis；PPL/zero-shot 完成并写入 DONE 后立即删模型。

## 永久保留与可删除边界

永久保留在 F：配置、命令、代码/数据 manifest SHA、EB/rank 向量及 SHA、basis canonical SHA、B_live、资源比例、per-window NLL、PPL、统计结果、状态与日志。

完成验证后删除于 E：完整 basis、压缩 checkpoint、临时 serialization shard、state scratch。原始 Llama/Qwen/OPT 模型和数据集不删除，D 盘不写入。

