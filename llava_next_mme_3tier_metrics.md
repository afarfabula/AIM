# LLaVA-NeXT MME Three-Tier Metrics

## Scope

- Model: `liuhaotian/llava-v1.6-vicuna-13b`
- Task: `mme`
- Attention: `sdpa`
- Strategy set:
  - `bishemethod_v2stage_anchor16_litefirst_next_t160`
  - `bishemethod_v2stage_anchor16_litefirst_next_t320`
  - `bishemethod_v2stage_anchor16_litefirst_next_t640`

## Summary

| Tier | Strategy | Cognition | Perception | Total |
|---|---|---:|---:|---:|
| 160 tokens | `bishemethod_v2stage_anchor16_litefirst_next_t160` | 323.2143 | 1377.9457 | 1701.1600 |
| 320 tokens | `bishemethod_v2stage_anchor16_litefirst_next_t320` | 313.5714 | 1437.4280 | 1750.9994 |
| 640 tokens | `bishemethod_v2stage_anchor16_litefirst_next_t640` | 315.3571 | 1496.8443 | 1812.2014 |

## Delta Vs 640

| Tier | Cognition delta | Perception delta | Total delta |
|---|---:|---:|---:|
| 320 vs 640 | -1.7857 | -59.4163 | -61.2020 |
| 160 vs 640 | +7.8572 | -118.8986 | -111.0414 |

## Category Breakdown

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

## Observations

- On total MME score, the ordering is `640 > 320 > 160`.
- `640` improves mainly through the perception-heavy categories, especially `count`, `OCR`, `existence`, and `posters`.
- `160` is not uniformly worse on every cognition sub-category; for example, it is higher than `640` on `code_reasoning` and `numerical_calculation`, but these gains do not offset the perception drop.
- `320` sits between `160` and `640` overall, and is very close to `640` on cognition score.

## Source Runs

- `t160`
  - Log: `/mlx_devbox/users/quyanyi/playground/AIM/logs/mme_limitnone_llava_next13b_bishemethod_v2stage_anchor16_litefirst_next_t160_20260512_130952/run.log`
- `t320`
  - Log: `/mlx_devbox/users/quyanyi/playground/AIM/logs/mme_limitnone_llava_next13b_bishemethod_v2stage_anchor16_litefirst_next_t320_20260512_130951/run.log`
- `t640`
  - Log: `/mlx_devbox/users/quyanyi/playground/AIM/logs/mme_limitnone_llava_next13b_bishemethod_v2stage_anchor16_litefirst_next_t640_20260512_133627/run.log`
