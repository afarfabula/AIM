# Anchor-Aware MME Variants

## 背景

当前 `samp` 路线里的 `anchor` 保护和 `bipartite merge` 之间有一个结构性问题：

- merge 核心仍然使用固定的偶数位 / 奇数位二分：
  - `a = metric[..., ::2, :]` 作为 `source`
  - `b = metric[..., 1::2, :]` 作为 `target`
- `anchor` 目前只通过 `protected_idx` 和 `protected_penalty` 去降低被当作 `source` 并掉的概率。
- 这样做的隐含前提是：重要的 `anchor` 能比较均匀地分布到偶数位和奇数位。
- 但真实情况里，`anchor` 很可能因为空间位置或先前的 token 顺序，集中落在某一侧。
- 一旦 `anchor` 基本都落在 `source` 或基本都落在 `target`，现有“保护 source”的做法就会失真：
  - 全落在 `source`：保护过强，merge 自由度变差
  - 全落在 `target`：保护几乎失效，anchor-aware 变成伪命题

所以这一轮的核心不是继续调 `anchor` 数量，而是先让 `anchor` 和 `source/target` 划分真正解耦。

## 这一轮的总改动

### 1. 给 merge 核心加了 `token_order`

在 `llava/model/token_merge.py` 里，`bipartite_soft_matching_merge()` 现在支持一个新的可选参数：

- `token_order`

它的作用是：

- 在进入固定偶奇分组之前，先按我们指定的顺序重排 token
- 然后再走原有的 `a=even, b=odd` 匹配逻辑

这样就能把“谁当 source / 谁当 target”的决定权，从原始 token 顺序里拿出来。

### 2. 做了一个 anchor-aware 的基类

在 `llava/model/token_prune_strategies.py` 里新增了两层基础实现：

- `_AnchorAwareOrderMixin`
- `_AnchorAwareV2StageBase`

其中：

- `_AnchorAwareOrderMixin` 负责根据 `anchor_idx` 生成新的 token 排列顺序
- `_AnchorAwareV2StageBase` 负责把这个顺序注入到每一轮 merge 里

### 3. 保持实验可比性

这 10 个版本都固定了同一套实验骨架：

- backbone: `bishemethod_v2stage_anchor16_litefirst` 风格
- `anchor_points = ["grid4x4"]`
- `merge_steps = [(192, 192), (144, 144), (96, 96), (48, 48)]`
- `target_total_tokens = 640`
- `task = mme`
- `limit = none`
- `attn = sdpa`

也就是说，这一轮只改 `anchor-aware matching` 的形状，不混入别的变量。

## 三种核心排序模式

### `alternate`

做法：

- `anchor, non-anchor, anchor, non-anchor, ...` 交替排

直觉：

- 最简单地避免 anchor 全落在同一 parity
- 让偶数位和奇数位都更容易同时包含 anchor

### `balanced_pairs`

做法：

- 尽量把 token 组织成 `(anchor, non-anchor)` 这样的二元组

直觉：

- 更明确地把 bipartite pair 做成“一个 anchor 对一个普通 token”
- 让 source/target 的对称性更强

### `block`

做法：

- 按块排列，比如 `g=2` 时是 `2 个 anchor + 2 个 non-anchor` 一组
- `g=3` 时是 `3 个 anchor + 3 个 non-anchor` 一组

直觉：

- 不再追求严格交替
- 保留局部的 anchor 团块，看看更强的局部结构保留是否有益

## 十个版本

## `v1`

- 策略名：`bishemethod_v2stage_anchor16_aware_v1`
- 改法：`alternate`
- 生效范围：只在第一轮 merge 之前重排
- 保护强度：`protected_penalty = 0.09`
- 目的：最小侵入版本，先修掉首轮最严重的 parity collapse

## `v2`

- 策略名：`bishemethod_v2stage_anchor16_aware_v2`
- 改法：`alternate`
- 生效范围：每一轮 merge 都重排
- 保护强度：`protected_penalty = 0.09`
- 目的：把 anchor-aware 划分做到底，测试“持续重排”是否比“只修第一轮”更强

## `v3`

- 策略名：`bishemethod_v2stage_anchor16_aware_v3`
- 改法：`balanced_pairs`
- 生效范围：只在第一轮生效
- 保护强度：`protected_penalty = 0.09`
- 目的：显式把 pair 组织成 `anchor + non-anchor`，比简单交替更贴近 bipartite matching 的结构

## `v4`

- 策略名：`bishemethod_v2stage_anchor16_aware_v4`
- 改法：`balanced_pairs`
- 生效范围：只在第一轮生效
- 保护强度：`protected_penalty = 0.05`
- 目的：在 `v3` 的基础上放松保护，测试“结构对齐”是否比“强保护”更重要

## `v5`

- 策略名：`bishemethod_v2stage_anchor16_aware_v5`
- 改法：`block`
- `group_size = 2`
- 生效范围：只在第一轮生效
- 保护强度：`protected_penalty = 0.09`
- 目的：允许更局部的 anchor 团块存在，避免过强的交替打散空间相关性

## `v6`

- 策略名：`bishemethod_v2stage_anchor16_aware_v6`
- 改法：`block`
- `group_size = 3`
- 生效范围：只在第一轮生效
- 保护强度：`protected_penalty = 0.09`
- 目的：进一步加强局部块结构，测试更粗粒度的 anchor grouping 是否更稳

## `v7`

- 策略名：`bishemethod_v2stage_anchor16_aware_v7`
- 改法：`alternate`
- 生效范围：只在第一轮生效
- 保护强度：`protected_penalty = 0.09`
- 附加改动：`pos_mix = 0.08`
- 目的：在 parity 修正之外，额外增强位置先验，看 anchor 的空间几何信息是否值得更强保留

## `v8`

- 策略名：`bishemethod_v2stage_anchor16_aware_v8`
- 改法：`alternate`
- 生效范围：只在第一轮生效
- 保护强度：`protected_penalty = 0.09`
- 附加改动：`pos_mix = 0.02`
- 目的：和 `v7` 形成对照，测试较弱位置先验是否反而让 Q/K 相似度主导得更合理

## `v9`

- 策略名：`bishemethod_v2stage_anchor16_aware_v9`
- 改法：`alternate`
- 生效范围：只在第一轮生效
- 保护强度：`protected_penalty = 0.09`
- 附加改动：`qk_mix = 0.30`
- 目的：增加 query 成分，让 matching 更偏“当前任务查询相关性”，而不只偏 key / feature 相似性

## `v10`

- 策略名：`bishemethod_v2stage_anchor16_aware_v10`
- 改法：`alternate`
- 生效范围：只在第一轮生效
- 保护强度：`protected_penalty = 0.03`
- 目的：测试最轻保护版本，看看 anchor-aware 排列本身是否已经足够，不需要太强的保护 bias

## 实验分组理解

如果按设计维度把这 10 个版本分组，可以这样看：

- 分组 A: 先解决 parity 问题
  - `v1`, `v2`, `v3`, `v5`, `v6`
- 分组 B: 看 protection 强弱
  - `v3` vs `v4`
  - `v1` vs `v10`
- 分组 C: 看 first-stage-only 还是 every-stage
  - `v1` vs `v2`
- 分组 D: 看位置先验强弱
  - `v7` vs `v8`
- 分组 E: 看 query-aware 程度
  - `v1` vs `v9`

## 当前运行方式

当前在 `worker 3832346` 上跑的是：

- 全量 `MME`
- 两张卡并发
- 10 个版本自动排队接力
- 当前队列脚本：
  - `scripts/queue_anchor_mme_variants.py`

队列日志目录：

- `/mlx_devbox/users/quyanyi/playground/AIM/logs/anchor_mme_queue_3832346_20260512_151856`

到我写这份文档的时候，已经至少完成：

- `v1`
- `v2`
- `v3`
- `v4`
- `v5`
- `v6`

后续版本会自动继续：

- `v7`
- `v8`
- `v9`
- `v10`

## 我最关心的比较

这一轮结果出来以后，我会优先看下面几组：

- `v1` vs `v2`
  - 判断是不是每一轮都做 anchor-aware 重排才有必要
- `v3` vs `v1`
  - 判断“显式 pair 平衡”是否比简单交替更好
- `v5/v6` vs `v1`
  - 判断块状结构保留是否比均匀打散更有利
- `v7/v8` vs `v1`
  - 判断位置编码强度是不是关键变量
- `v9` vs `v1`
  - 判断 query-aware matching 是否提升 MME
- `v10` vs `v1`
  - 判断 anchor-aware 的收益到底来自“重排”还是来自“强保护”

## 预期筛选方式

如果这 10 个版本里能跑出 2 到 3 个明显更强的版本，下一轮我建议这样继续：

- 先固定最优排序模式
- 再只在这个模式上扫 `protected_penalty`
- 然后把最优版本下探到 `320 / 160`
- 最后再回到 `GQA / MMMU / SQA / MME` 联合比较

这样能把搜索空间控制住，不会一下子把变量耦合得太多。
