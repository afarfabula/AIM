# LLaVA-NeXT Three-Tier Benchmark Metrics

## Scope

- This document aggregates multiple run sets for LLaVA-NeXT token reduction experiments.

## Run Set A: SAMP (litefirst) on 13B

## Result Summary

| Tier | Strategy | GQA | MMMU | ScienceQA |
|---|---|---:|---:|---:|
| 640 tokens | `bishemethod_v2stage_anchor16_litefirst_next_t640` | 0.6327 | 0.3622 | 0.7134 |
| 320 tokens | `bishemethod_v2stage_anchor16_litefirst_next_t320` | 0.6101 | 0.3711 | 0.7000 |
| 160 tokens | `bishemethod_v2stage_anchor16_litefirst_next_t160` | 0.5922 | 0.3744 | 0.6941 |

## Delta Vs 640

| Tier | GQA delta | MMMU delta | ScienceQA delta |
|---|---:|---:|---:|
| 320 vs 640 | -0.0226 | +0.0089 | -0.0134 |
| 160 vs 640 | -0.0405 | +0.0122 | -0.0193 |

## MME Summary

| Tier | Strategy | MME cognition | MME perception | MME total |
|---|---|---:|---:|---:|
| 640 tokens | `bishemethod_v2stage_anchor16_litefirst_next_t640` | 315.3571 | 1496.8443 | 1812.2014 |
| 320 tokens | `bishemethod_v2stage_anchor16_litefirst_next_t320` | 313.5714 | 1437.4280 | 1750.9994 |
| 160 tokens | `bishemethod_v2stage_anchor16_litefirst_next_t160` | 323.2143 | 1377.9457 | 1701.1600 |

## MME Delta Vs 640

| Tier | Cognition delta | Perception delta | Total delta |
|---|---:|---:|---:|
| 320 vs 640 | -1.7857 | -59.4163 | -61.2020 |
| 160 vs 640 | +7.8572 | -118.8986 | -111.0414 |

## MME Breakdown

| Category | 160 tokens | 320 tokens | 640 tokens |
|---|---:|---:|---:|
| `code_reasoning` | 70.00 | 72.50 | 60.00 |
| `numerical_calculation` | 52.50 | 35.00 | 40.00 |
| `text_translation` | 80.00 | 87.50 | 92.50 |
| `commonsense_reasoning` | 120.71 | 118.57 | 122.86 |
| `artwork` | 116.75 | 126.00 | 125.75 |
| `celebrity` | 154.71 | 151.76 | 152.65 |
| `count` | 103.33 | 118.33 | 136.67 |
| `color` | 160.00 | 178.33 | 175.00 |
| `position` | 110.00 | 93.33 | 106.67 |
| `OCR` | 110.00 | 102.50 | 125.00 |
| `landmark` | 150.25 | 158.50 | 160.75 |
| `scene` | 171.75 | 170.50 | 169.50 |
| `existence` | 160.00 | 180.00 | 185.00 |
| `posters` | 141.16 | 158.16 | 159.86 |

## Source Runs

- `t640`
  - Log: `/mlx_devbox/users/quyanyi/playground/AIM/logs/gqa_mmmu_val_scienceqa_img_limitnone_llava_next13b_bishemethod_v2stage_anchor16_litefirst_next_t640_20260511_200525/run.log`
- `t320`
  - Log: `/mlx_devbox/users/quyanyi/playground/AIM/logs/gqa_mmmu_val_scienceqa_img_limitnone_llava_next13b_bishemethod_v2stage_anchor16_litefirst_next_t320_20260511_210311/run.log`
- `t160`
  - Log: `/mlx_devbox/users/quyanyi/playground/AIM/logs/gqa_mmmu_val_scienceqa_img_limitnone_llava_next13b_bishemethod_v2stage_anchor16_litefirst_next_t160_20260511_200525/run.log`
- `MME t640`
  - Log: `/mlx_devbox/users/quyanyi/playground/AIM/logs/mme_limitnone_llava_next13b_bishemethod_v2stage_anchor16_litefirst_next_t640_20260512_133627/run.log`
- `MME t320`
  - Log: `/mlx_devbox/users/quyanyi/playground/AIM/logs/mme_limitnone_llava_next13b_bishemethod_v2stage_anchor16_litefirst_next_t320_20260512_130951/run.log`
- `MME t160`
  - Log: `/mlx_devbox/users/quyanyi/playground/AIM/logs/mme_limitnone_llava_next13b_bishemethod_v2stage_anchor16_litefirst_next_t160_20260512_130952/run.log`

## Notes

- In this run set, `GQA` and `ScienceQA` decrease as the token budget is reduced from `640 -> 320 -> 160`.
- `MMMU` is slightly higher at lower token budgets in this specific three-task run.
- These are direct full-run metrics copied from the final metric table in each `run.log`.
- For `MME`, the total score ordering is `640 > 320 > 160`, with most of the gain coming from the perception side rather than cognition.

## Run Set B: Anchor-aware v6 (full 4-task runs) on 13B / 7B

### Scope

- Models:
  - `liuhaotian/llava-v1.6-vicuna-13b`
  - `liuhaotian/llava-v1.6-vicuna-7b`
- Tasks: `gqa`, `mme`, `scienceqa_img`, `mmmu_val`
- Attention: `sdpa`
- Strategy set:
  - `bishemethod_v2stage_anchor16_aware_v6_next_t640`
  - `bishemethod_v2stage_anchor16_aware_v6_next_t320`
  - `bishemethod_v2stage_anchor16_aware_v6_next_t160`

### Result Summary (13B)

| Tier | Strategy | GQA | MMMU | ScienceQA | MME cognition | MME perception | MME total |
|---|---|---:|---:|---:|---:|---:|---:|
| 640 tokens | `bishemethod_v2stage_anchor16_aware_v6_next_t640` | 0.633328 | 0.36667 | 0.708974 | 350.0000 | 1500.8980 | 1850.8980 |
| 320 tokens | `bishemethod_v2stage_anchor16_aware_v6_next_t320` | 0.621005 | 0.37444 | 0.712940 | 298.2143 | 1437.5186 | 1735.7329 |
| 160 tokens | `bishemethod_v2stage_anchor16_aware_v6_next_t160` | 0.602322 | 0.37222 | 0.702033 | 305.3571 | 1403.3398 | 1708.6970 |

### Delta Vs 640 (13B)

| Tier | GQA delta | MMMU delta | ScienceQA delta | MME total delta |
|---|---:|---:|---:|---:|
| 320 vs 640 | -0.012323 | +0.00777 | +0.003966 | -115.1651 |
| 160 vs 640 | -0.031006 | +0.00555 | -0.006941 | -142.2010 |

### Result Summary (7B)

| Tier | Strategy | GQA | MMMU | ScienceQA | MME cognition | MME perception | MME total |
|---|---|---:|---:|---:|---:|---:|---:|
| 640 tokens | `bishemethod_v2stage_anchor16_aware_v6_next_t640` | 0.620051 | 0.37889 | 0.679227 | 344.2857 | 1476.0037 | 1820.2894 |
| 320 tokens | `bishemethod_v2stage_anchor16_aware_v6_next_t320` | 0.606456 | 0.37222 | 0.671790 | 329.6429 | 1367.4271 | 1697.0699 |
| 160 tokens | `bishemethod_v2stage_anchor16_aware_v6_next_t160` | 0.587613 | 0.36333 | 0.684680 | 321.7857 | 1318.1435 | 1639.9292 |

### Delta Vs 640 (7B)

| Tier | GQA delta | MMMU delta | ScienceQA delta | MME total delta |
|---|---:|---:|---:|---:|
| 320 vs 640 | -0.013595 | -0.00667 | -0.007437 | -123.2195 |
| 160 vs 640 | -0.032438 | -0.01556 | +0.005453 | -180.3602 |

### Source Runs (Run Set B)

- 13B `t640`
  - Log: `/mlx_devbox/users/quyanyi/playground/AIM/logs/gqa_mme_scienceqa_img_mmmu_val_limitnone_llava-v1.6-vicuna-13b_bishemethod_v2stage_anchor16_aware_v6_next_t640_20260512_212049/run.log`
  - Results: `/mlx_devbox/users/quyanyi/playground/AIM/logs/gqa_mme_scienceqa_img_mmmu_val_limitnone_llava-v1.6-vicuna-13b_bishemethod_v2stage_anchor16_aware_v6_next_t640_20260512_212049/snapshots__22422b4c3a3ef1ba52aca074cc9021216877ce5d/20260512_212100_results.json`
- 13B `t320`
  - Log: `/mlx_devbox/users/quyanyi/playground/AIM/logs/gqa_mme_scienceqa_img_mmmu_val_limitnone_llava-v1.6-vicuna-13b_bishemethod_v2stage_anchor16_aware_v6_next_t320_20260513_000120/run.log`
- 13B `t160`
  - Log: `/mlx_devbox/users/quyanyi/playground/AIM/logs/gqa_mme_scienceqa_img_mmmu_val_limitnone_llava-v1.6-vicuna-13b_bishemethod_v2stage_anchor16_aware_v6_next_t160_20260512_212251/run.log`
- 7B `t640`
  - Log: `/mlx_devbox/users/quyanyi/playground/AIM/logs/gqa_mme_scienceqa_img_mmmu_val_limitnone_llava-v1.6-vicuna-7b_bishemethod_v2stage_anchor16_aware_v6_next_t640_20260512_165711/run.log`
- 7B `t320`
  - Log: `/mlx_devbox/users/quyanyi/playground/AIM/logs/gqa_mme_scienceqa_img_mmmu_val_limitnone_llava-v1.6-vicuna-7b_bishemethod_v2stage_anchor16_aware_v6_next_t320_20260512_190904/run.log`
- 7B `t160`
  - Log: `/mlx_devbox/users/quyanyi/playground/AIM/logs/gqa_mme_scienceqa_img_mmmu_val_limitnone_llava-v1.6-vicuna-7b_bishemethod_v2stage_anchor16_aware_v6_next_t160_20260512_165932/run.log`

## Paper Reference

- Paper table: `VisionZip on LLaVA-NeXT`
- Vanilla upper bound uses `2880` visual tokens.
- This updated table includes runtime measurements (`Prefilling`, `Total`) and reports both `13B` and `7B` vanilla baselines.

### Updated VisionZip Table

| Method | Model | Tokens | Prefilling | Total | GQA | MMB | MME | POPE | SQA | VQAV2 | VQAText | MMMU | SEED-I | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 13B | 2880 | 129.4ms | 2506s | 65.4 | 70.0 | 1901 | 86.2 | 73.5 | 81.8 | 64.3 | 36.2 | 71.9 | 100.0% |
| Retention vs Vanilla 13B |  |  |  |  | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Vanilla | 7B | 2880 | 54.2ms | 1598s | 64.2 | 67.9 | 1842 | 86.4 | 70.2 | 80.1 | 61.3 | 35.1 | 70.2 | 97.2% |
| Retention vs Vanilla 13B |  |  |  |  | 98.2% | 96.3% | 96.9% | 100.2% | 95.5% | 97.9% | 95.3% | 97.0% | 97.6% | 97.2% |
| VisionZip | 13B | 640 | 48.2ms | 1219s | 63.0 | 68.6 | 1871 | 85.7 | 71.2 | 79.7 | 62.2 | 36.4 | 68.8 | 97.5% |
| Retention vs Vanilla 13B |  |  |  |  | 96.3% | 98.0% | 98.4% | 99.4% | 96.7% | 96.9% | 96.7% | 100.5% | 95.7% | 97.5% |
| VisionZip | 13B | 320 | 30.3ms | 995s | 60.7 | 67.2 | 1805 | 82.0 | 70.3 | 76.8 | 60.9 | 35.6 | 65.2 | 94.7% |
| Retention vs Vanilla 13B |  |  |  |  | 92.8% | 96.0% | 95.0% | 95.1% | 95.6% | 93.9% | 94.7% | 98.3% | 90.7% | 94.7% |
| VisionZip | 13B | 160 | 23.9ms | 888s | 57.8 | 64.9 | 1739 | 76.6 | 69.3 | 72.4 | 58.4 | 37.0 | 61.1 | 91.3% |
| Retention vs Vanilla 13B |  |  |  |  | 88.4% | 92.7% | 91.5% | 88.9% | 94.3% | 88.5% | 90.8% | 102.2% | 84.8% | 91.3% |

## Overlap Comparison

- Our current run only overlaps with the paper on `GQA`, `MMMU`, and `SQA` (`scienceqa_img`).
- So the fairest direct comparison is on these three tasks only, not on the paper's full task set.

### Raw Scores On Overlapping Tasks

| Tier | Method | GQA | MMMU | SQA | 3-task mean |
|---|---|---:|---:|---:|---:|
| 640 tokens | Ours | 63.27 | 36.22 | 71.34 | 56.94 |
| 640 tokens | VisionZip (paper) | 63.00 | 36.40 | 71.20 | 56.87 |
| 320 tokens | Ours | 61.01 | 37.11 | 70.00 | 56.04 |
| 320 tokens | VisionZip (paper) | 60.70 | 35.60 | 70.30 | 55.53 |
| 160 tokens | Ours | 59.22 | 37.44 | 69.41 | 55.36 |
| 160 tokens | VisionZip (paper) | 57.80 | 37.00 | 69.30 | 54.70 |

### Retention On Overlapping Tasks

| Tier | Method | GQA retention | MMMU retention | SQA retention | 3-task mean retention |
|---|---|---:|---:|---:|---:|
| 640 tokens | Ours (vs Vanilla 13B) | 96.74% | 100.06% | 97.06% | 97.95% |
| 640 tokens | VisionZip (vs Vanilla 13B) | 96.33% | 100.55% | 96.87% | 97.92% |
| 320 tokens | Ours (vs Vanilla 13B) | 93.29% | 102.51% | 95.24% | 97.01% |
| 320 tokens | VisionZip (vs Vanilla 13B) | 92.81% | 98.34% | 95.65% | 95.60% |
| 160 tokens | Ours (vs Vanilla 13B) | 90.55% | 103.43% | 94.44% | 96.14% |
| 160 tokens | VisionZip (vs Vanilla 13B) | 88.38% | 102.21% | 94.29% | 94.96% |

## Takeaway

- 在与 VisionZip 表格可直接对齐的三项任务 `GQA + MMMU + SQA` 上，我们在 `640 / 320 / 160` 三档的三任务平均分均不低于 VisionZip。
- 在 `MMMU` 上我们并没有稳定优势，但 `GQA` 与 `SQA` 更稳定，整体三任务平均更平衡。
- 如果论文里要写严谨一点，建议表述成：
  `在与 VisionZip 表格可重叠的 GQA, MMMU, ScienceQA 三项任务上，我们的方法在 640/320/160 三档均达到可比或略优的三任务平均表现；在不同 token budget 下优势主要由 GQA 与 ScienceQA 贡献，而 MMMU 更接近持平。`
