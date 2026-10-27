# SEVERE_CONTROL_REPAIR_PLAN — 修复 10/3 Uniform 和 No-Z 的最小实验计划

状态：**只是计划，没有执行任何一项**，等你决定。所有候选方案都只根据资源选出，没有用任何质量指标。
依据：`OLD_1003_TRUNCATED_CURVE_AUDIT.md`、`TRUNCATED_CURVE_AUDIT.json`。

## 0. 需要你先决定的一点

用完整曲线时，Uniform 和 No-Z 都**没有落在 ±0.15% 以内的点**，这是 rank 格点的离散性决定的，不是选点的问题。两者最近的点都在 +0.17% 到 +0.22% 之间。

| 选项 | 内容 | 评价 |
|---|---|---|
| A（推荐） | 接受 NEAR_MATCHED（≤1%），选**比 Full 多字节**的那一侧 | 对照组拿到的资源比 FDP 多，对 FDP 更不利，所以是保守的，审稿人容易接受 |
| B | 选比 Full 少字节的一侧（−0.28% / −0.33%） | 对照组资源更少，对 FDP 有利，不推荐 |
| C | 放宽 Uniform 的定义，比如允许第 0 层单独设 EB | 等于改变了 Uniform 的定义，不推荐 |

## R1. 修正 Uniform（必需：只有这样 Full vs Uniform 才是干净对照）

| 项目 | 值 |
|---|---|
| rank | 完整曲线，单一 EB = 0.0930587692，SHA **823e850e…**，B = 151,617,536（**+0.217%**，最小 rank 56） |
| 与历史 Uniform 的差异 | 第 0 层 656/960→808/992；第 3 层 344/792→280/768 |
| 备注 | 和 V2 Severe Uniform 的 rank 完全相同 |

**R1-a：严格同路径（推荐）**

步骤和 10/3 Spectral 完全一样：
1. 用 `fill_arm2.py` 从 10/3 master（sha 638d4748…）切片并做 authoritative fill（CPU，约 1 分钟）。
2. 跑 `verify_o3s.py` 的同款核验：rank、basis、V3 算子一致性、CPU 双进程 canary。
3. 跑 Wiki/C4 PPL，再跑五任务零样本。
4. 计算配对 bootstrap：Full − Uniform（PPL 和准确率）、Spectral − Uniform。

| 资源 | 估计 |
|---|---|
| GPU | 1 张卡约 15 分钟（参考本次实际耗时：PPL 两项各约 1.5 分钟，零样本约 11 分钟） |
| 磁盘 | 15 GB，用完按清单删除 |

**R1-b：零 GPU 替代方案（只能标"跨路径"）**

直接使用已有的 V2 Uniform 评测结果。它和 10/3 路径在算子上等价：basis 32/32 层相同，功能误差 ≤3.7e-5。但它以 fp16 存储，不是逐位相同。

用它算出的现有数据：
- Full − V2 Uniform：ΔJ −0.159 [−0.175, −0.143]；准确率 +0.69 pp [−0.06, +1.43]。

如果论文需要准确率的显著性结论，**R1-a 不能省**。

## R2. 修正 No-Z（如果论文保留 No-Z 消融，就需要）

| 项目 | 值 |
|---|---|
| 方案 | 完整曲线，10/3 公式（d2+d5），base **0.093027** |
| B | 151,552,000（**+0.173%**），rank SHA **4eadcda5…**，最小 rank 64 |
| 另一侧的点 | base 0.093106，B 150,798,336（−0.325%），只有第 0 层不同 |

步骤：
1. `fill_arm2.py` 生成模型。
2. 核验（同上）。
3. **只测 Wiki/C4 PPL**，和原 No-Z 一样，不测准确率。
4. 计算 Full − No-Z 的配对 bootstrap。

| 资源 | 估计 |
|---|---|
| GPU | 约 4 分钟 |
| 磁盘 | 15 GB，用完删除 |

## R3. 用修正 Uniform 重新锚定的 Spectral（可选）

只有在论文把 Uniform 换成 R1 的版本，并且要求 Spectral 的锚点也一致时才需要。

| 项目 | 值 |
|---|---|
| 锚点 | 823e850e |
| 方案 | base 0.095212，B 151,093,248（−0.130%），rank SHA aaae98d8… |
| 与冻结 Spectral 的差异 | 第 2、3、24 层 |
| 资源 | GPU 约 15 分钟（PPL + 零样本），磁盘 15 GB |

如果不做 R3，就保留冻结的 Spectral（7bf576fd），并在文中注明它的锚点是历史 Uniform。

## 执行方式（如果批准）

- 在 `FDP_OFFICIAL_PROTOCOL_V1` 的调度器中新增作业，每个模型做完删一个，只用一张 GPU。
- 新方案先写入新的只读 PLAN 文件（`REPAIR_PLAN.json`），所有旧文件保持不动。
- 结果写入新目录 `quality/UNIFORM_FULLTRACE`、`quality/NO_Z_FULLTRACE`，历史的 `quality/UNIFORM`、`quality/NO_Z` 保持原样。
- 先修好调度器的 bug（核验作业缺少 post 函数）；这只是工程修复，不改变任何算法。

| 方案 | GPU 时间 | 峰值磁盘 |
|---|---|---|
| 最小（R1-a + R2） | 约 20 分钟 | 15 GB |
| 全部（加 R3） | 约 35 分钟 | 15 GB |

**磁盘**：/diskE 剩 26 GB，其中有 15 GB 是保留中的 10/3 Spectral checkpoint。建议先按清单删掉它。

## 不需要重跑的部分

- 10/3 Full、No-S、No-Entropy：完全在曲线覆盖范围内，结论不变。
- 冻结的 10/3 Spectral：正式评测已完成，全部核验通过。
- V2 Severe：用的是完整曲线（eb=1.0），不受影响。
- 官方协议（FDP_OFFICIAL_PROTOCOL_V1）：用的是官方 builder 的逐层 trace，不是这份 envelope 曲线文件，不受影响。
- Light/Medium/Aggressive 三档：**不在本次审计范围内**。如果其中有选点也是用 envelope 截断曲线回放的，应该用同样的覆盖检查再核一遍。这是纯 CPU 检查，可以另行安排。
