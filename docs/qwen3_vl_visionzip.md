# Qwen3-VL 视觉 Token Reduction 适配说明与六基准结果

本文档给论文整理 / agent 交接使用，说明 AIM 中如何把 **VisionZip** 与 **bishe soft-merge** 适配到 **Qwen3-VL**，以及当前在 6 个 benchmark（不含 TextVQA）上的结果汇总。

---

## 1. 结论先行

- **Qwen3-VL 的视觉实现和 LLaVA 家族不一样**，不能直接把 LLaVA 上的 token reduction 逻辑原样搬过来。
- LLaVA 常见结构是：`vision_tower -> resampler/projector -> LLM`；而 **Qwen3-VL 是原生多模态模型**，视觉侧有自己的 `visual blocks + patch merger + deepstack + mRoPE`。
- 因此在 Qwen3-VL 上做 token reduction，除了“怎么选 token”之外，还必须同时处理：
  - 视觉塔内部的 token 流；
  - LLM 侧的 image placeholder 对齐；
  - `position_ids` / `rope_deltas` / 3D mRoPE；
  - `deepstack_visual_embeds` 的同步裁剪或同步 soft-merge。
- 当前 AIM 中：
  - **VisionZip** 在 Qwen3-VL 上走的是 **hard-prune 路径**；
  - **bishemethod_v2stage_anchor16_aware_v6** 走的是 **soft-merge 路径**。

---

## 2. 为什么 Qwen3-VL 不能按 LLaVA 的方式直接适配

### 2.1 LLaVA 家族常见路径

LLaVA 常见接法是外接视觉编码器，再把视觉特征投到 LLM 隐空间：

1. `vision_tower` 提取视觉特征；
2. `vision_resampler` / `mm_projector` 做尺寸或维度映射；
3. 再和文本 token 拼接送进 LLM。

这种结构下，token reduction 往往可以放在：

- vision tower 输出后；
- projector 前后；
- 或 LLaVA 的 `prepare_inputs_labels_for_multimodal()` 附近。

### 2.2 Qwen3-VL 的关键不同点

Qwen3-VL 不是“外接视觉塔”的 LLaVA 式结构，而是 **模型内部原生包含视觉主干**。它有几个适配上必须注意的点：

1. **视觉 patch embed 是 3D Conv**，不是常见 CLIP/SigLIP 的 2D patch embed。  
   这意味着视觉 token 的组织方式天然带有 `T/H/W` 结构。

2. **视觉 attention 是单个 `qkv` 线性层**，不是 `q_proj / k_proj / v_proj` 三分支。  
   因此 bishe 这类依赖 q/k metric 的方法，在 Qwen3-VL 上必须专门从 `qkv` 里拆 q/k，不能直接复用 LLaVA 代码。

3. **Qwen3-VL 内建 patch merger 和 deepstack**。  
   视觉 token 不是简单地“抽出来投影”就完了，而是还要进入 merger，并在语言模型早期层通过 deepstack 回注视觉信息。

4. **文本侧使用 image placeholder + 3D mRoPE 对齐视觉 token**。  
   一旦视觉 token 数变了，不仅要改视觉特征本身，还要同步改：
   - `input_ids`
   - `attention_mask`
   - `position_ids`
   - `rope_deltas`
   - image placeholder 对齐关系

所以，**Qwen3-VL 的适配难点不是 token 选择本身，而是 token 改完之后如何维护 Qwen3 原生多模态约束**。

---

## 3. 当前 AIM 里的适配方案

### 3.1 总体思路

AIM 没有重写一套 Qwen3-VL，而是采用了 **wrapper + monkey patch** 的方式：

1. 在 `lmms-eval` 的 `qwen3_vl` wrapper 中加载原生 `Qwen3VLForConditionalGeneration`；
2. 根据 `token_prune_strategy` 分发到不同 patch；
3. patch 原生 Qwen3-VL 的：
   - vision forward
   - image feature extraction
   - model forward
   - causal generation forward

这样做的好处是：

- 可以直接复用 HuggingFace Qwen3-VL；
- 改动集中在 AIM 自己的 patch 文件中；
- VisionZip / bishe 都能共用同一套 Qwen3 对齐逻辑。

### 3.2 入口文件

- `other_packages/lmms-eval/lmms_eval/models/qwen3_vl.py`
  - 负责解析 `token_prune_strategy` / `token_prune_config`
  - 负责根据 strategy 把 patch 应用到原生 Qwen3-VL

- `llava/qwen3_vl_visionzip.py`
  - Qwen3-VL token reduction 的核心 patch 文件
  - VisionZip hard-prune 与 bishe soft-merge 都在这里落地

- `llava/qwen3_vl_bishemethod_v2stage_anchor16_aware_v6.py`
  - bishe v2stage 在 Qwen3-VL 上的薄适配入口

---

## 4. VisionZip 在 Qwen3-VL 上是怎么适配的

### 4.1 当前实现语义

当前 AIM 里名义上叫 `VisionZip`，但在 Qwen3-VL 上实际走的是 **merged-token 级别的 hard-prune**：

1. 先取 Qwen3-VL 的 image features；
2. 在 merged token 空间里按 `dominant + contextual` 规则选保留位置；
3. 把 merged token 索引映射回 patch token 索引；
4. 在 Qwen3-VL 的视觉主干中做裁剪；
5. 再同步修正 LLM 侧的 placeholder / mask / mRoPE。

### 4.2 适配要点

- VisionZip 选择逻辑仍然保留“dominant + contextual”的配置接口；
- 但真正裁剪发生在 **Qwen3-VL 原生视觉前向路径** 中，而不是 LLaVA 式 projector 后；
- 裁剪完视觉 token 后，必须同步修改输入中的 image placeholder 和位置编码。

### 4.3 可引用的核心函数

- `apply_visionzip_to_qwen3_vl(...)`
- `_compute_visionzip_merged_keep_indices(...)`
- `_vision_forward(...)`
- `_adjust_inputs_for_dedup(...)`
- `_model_forward(...)`

---

## 5. bishe 在 Qwen3-VL 上是怎么适配的

### 5.1 当前实现语义

`bishemethod_v2stage_anchor16_aware_v6` 在 Qwen3-VL 上走的是 **soft-merge**，不是 hard-prune。

流程可以概括为：

1. 先从 Qwen3-VL visual blocks 中抽取 q/k metric；
2. 按 `spatial_merge_size` 把 patch token 池化到 merged token 粒度；
3. 使用 anchor-aware 的 bipartite merge 方案逐步 soft-merge；
4. 对主视觉特征和 deepstack 特征都做同样的 merge；
5. 再同步修改 LLM 侧的 placeholder 与位置编码。

### 5.2 为什么它不能直接复用 LLaVA 版 bishe

核心原因有两个：

1. **Qwen3 visual attention 只有单个 `qkv` 层**，而不是独立 `q_proj` / `k_proj`；
2. **Qwen3-VL 有 merger + deepstack + mRoPE**，soft-merge 后必须把这几部分一起维护好。

也就是说，Qwen3 版 bishe 不是简单地把 LLaVA 的 merge 函数搬过来，而是做了 **Qwen3 专项适配**。

### 5.3 可引用的核心函数

- `apply_bishemethod_v2stage_anchor16_aware_v6_to_qwen3_vl(...)`
- `_extract_qk_metric_from_qwen3_visual(...)`
- `_compute_bishe_soft_merged_outputs(...)`
- `_model_forward(...)`
- `_cg_forward(...)`

---

## 6. 多 benchmark 评测脚本与日志

当前六基准评测入口：

- `executable/runa_qwen3.sh`：单次入口，支持 bench alias 解析
- `executable/run_qwen3_visionzip_all.sh`：一次跑多个 benchmark
- `scripts/run_qwen3_vl_task.sh`：真正调用 lmms-eval 的脚本

当前固定使用的 6 个 benchmark：

- `gqa`
- `mme`
- `pope`
- `scienceqa_img`
- `mmmu_val`
- `mmbench_en_dev`

不含 `textvqa`。

日志命名已包含：

- benchmark 集合
- `limit`
- `token_prune_strategy`
- `token_prune_config`

便于后续追踪论文实验。

---

## 7. 当前六基准结果汇总

> 说明：
> - 64 档 VisionZip 对应历史配置 **d54/c10**，总保留 token 数为 64。
> - 原版 baseline 的 **GQA** 目前没有找到一条完全确认同口径的全量日志，因此暂记为空。

| 方案 | GQA | MMBench | MME cognition | MME perception | MMMU | POPE | ScienceQA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原版 baseline | — | 84.4502 | 638.2143 | 1717.3064 | 0.5344 | 0.8836 | 0.4725 |
| VisionZip-64（d54/c10） | 0.5860 | 81.6151 | 597.8571 | 1502.2706 | 0.5078 | 0.8684 | 0.4725 |
| bishe-64 | 0.5859 | 83.7629 | 589.2857 | 1615.9039 | 0.5033 | 0.8744 | 0.4809 |
| VisionZip-128（d128/c0） | 0.6045 | 83.5052 | 597.5000 | 1586.0025 | 0.5122 | 0.8768 | 0.4690 |
| bishe-128 | 0.6047 | 84.1065 | 616.4286 | 1687.8021 | 0.5111 | 0.8802 | 0.4834 |
| VisionZip-196（d196/c0） | 0.6172 | 84.4502 | 615.7143 | 1664.6415 | 0.5144 | 0.8814 | 0.4695 |
| bishe-196 | 0.6173 | 84.5361 | 628.9286 | 1682.9476 | 0.5256 | 0.8837 | 0.4720 |

### 7.1 结果解读

- **64 档**：bishe 在 MMBench、MME perception、POPE、ScienceQA 更强；VisionZip 在 MME cognition、MMMU 略强；GQA 几乎持平。
- **128 档**：bishe 在 7 项里有 6 项更优，仅 MMMU 略低 0.0011。
- **196 档**：bishe 在 7 项上全部优于 VisionZip。
- 从整体上看，**bishe 在中高保留率（128 / 196）上明显更稳**，尤其在 **MME perception** 和 **MMBench** 上优势更明显。

---

## 8. 建议在论文里怎么描述

可以用下面这段简化表述：

> We adapt visual token reduction to Qwen3-VL by patching the native HuggingFace Qwen3-VL forward stack, instead of inserting pruning only after an external vision tower as in LLaVA-style architectures. The key challenge is that Qwen3-VL uses a native visual backbone with 3D patch embedding, merged visual tokens, deepstack features, and multimodal 3D rotary position encoding (mRoPE). Therefore, token reduction must jointly modify visual token selection / merging, image placeholder alignment, attention masks, and position ids. Based on this design, we implement a hard-prune VisionZip path and a soft-merge bishe path for Qwen3-VL.

如果写中文版本，可以简化成：

> 与 LLaVA 家族常见的“外接 vision tower + projector”结构不同，Qwen3-VL 采用原生多模态视觉主干，并在视觉-语言对齐中使用 merged visual tokens、deepstack 和 3D mRoPE。因此，我们没有把 token reduction 简单插在 projector 后，而是直接 patch Qwen3-VL 原生 forward 栈，在视觉 token 选择/合并之外，同时维护 image placeholder、attention mask 与 position ids 的一致性。在此基础上，我们分别实现了 Qwen3-VL 上的 VisionZip hard-prune 路径与 bishe soft-merge 路径。

---

## 9. 关键代码位置（便于继续追）

- `other_packages/lmms-eval/lmms_eval/models/qwen3_vl.py`
  - wrapper，解析 strategy 并应用 patch
- `llava/qwen3_vl_visionzip.py`
  - Qwen3-VL token reduction 主实现
- `llava/qwen3_vl_bishemethod_v2stage_anchor16_aware_v6.py`
  - bishe v2stage 入口
- `executable/run_qwen3_visionzip_all.sh`
  - 多 benchmark 运行入口
- `scripts/run_qwen3_vl_task.sh`
  - lmms-eval 实际执行与日志目录命名

---

## 10. 数据来源

本文档中的关键评测结果来自以下日志：

- VisionZip-196：`/root/aim_qwen3_logs/visionzip_d196_c0_gqa+mme+pope+scienceqa+mmmu+mmbench_limitnone_runs_20260518_120507_summary.log`
- bishe-196：`/root/aim_qwen3_logs/bishemethod_v2stage_anchor16_aware_v6_bishe_target_keep-196_gqa+mme+pope+scienceqa+mmmu+mmbench_limitnone_runs_20260518_133200_summary.log`
- VisionZip-128：`/root/aim_qwen3_logs/visionzip_d128_c0_gqa+mme+pope+scienceqa+mmmu+mmbench_limitnone_runs_20260518_145854_summary.log`
- bishe-128：`/root/aim_qwen3_logs/bishemethod_v2stage_anchor16_aware_v6_bishe_target_keep-128_gqa+mme+pope+scienceqa+mmmu+mmbench_limitnone_runs_20260518_162216_summary.log`
- VisionZip-64：`/root/aim_qwen3_logs/run_full_d54_c10_20260516_123501/visionzip_runs_20260516_123501_summary.log`
- bishe-64：`/root/aim_qwen3_logs/bishe64_softmerge_full_runs_20260518_072710_summary.log`

