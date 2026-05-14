# LLaVA-1.5 9-Benchmark Anchor16 Retention

## Scope

- Model: `liuhaotian/llava-v1.5-7b`
- Backend: `sdpa`
- Parallelism: `virtual_dp`, `2 GPUs`
- Benchmarks: `gqa`, `pope`, `scienceqa_img`, `docvqa_val`, `mmmu_val`, `textvqa_val`, `chartqa`, `ai2d`, `vizwiz_vqa_val`
- Excluded benchmark: `mme` because its pairwise aggregation is incompatible with the current DP shard-local aggregation flow

## Source Logs

- Baseline (`strategy=none`):
  - `logs/virtual_dp_gqa__pope__scienceqa_img__docvqa_val__mmmu_val__textvqa_val__chartqa__ai2d__vizwiz_vqa_val_20260511_144613/shard_0/run.log`
- Token reduction (`strategy=bishemethod_v2stage_anchor16_litefirst`):
  - `logs/virtual_dp_gqa__pope__scienceqa_img__docvqa_val__mmmu_val__textvqa_val__chartqa__ai2d__vizwiz_vqa_val_20260511_161000/shard_0/run.log`

## Primary Metric Retention

Retention is computed as:

`retention = anchor16_metric / baseline_metric`

| Benchmark | Primary Metric | Baseline | Anchor16 | Retention |
|---|---|---:|---:|---:|
| `gqa` | `exact_match` | 0.6238 | 0.5828 | 93.43% |
| `pope` | `pope_accuracy` | 0.7913 | 0.8122 | 102.64% |
| `scienceqa_img` | `exact_match` | 0.6868 | 0.6759 | 98.41% |
| `docvqa_val` | `anls` | 0.2650 | 0.1821 | 68.72% |
| `mmmu_val` | `mmmu_acc` | 0.3667 | 0.3622 | 98.77% |
| `textvqa_val` | `exact_match` | 0.4708 | 0.3869 | 82.18% |
| `chartqa` | `relaxed_overall` | 0.2056 | 0.1632 | 79.38% |
| `ai2d` | `exact_match` | 0.5505 | 0.5324 | 96.71% |
| `vizwiz_vqa_val` | `exact_match` | 0.5412 | 0.5672 | 104.80% |

## Detailed Metrics

### `chartqa`

| Metric | Baseline | Anchor16 | Retention |
|---|---:|---:|---:|
| `relaxed_augmented_split` | 0.1536 | 0.1184 | 77.08% |
| `relaxed_human_split` | 0.2576 | 0.2080 | 80.75% |
| `relaxed_overall` | 0.2056 | 0.1632 | 79.38% |

### `pope`

| Metric | Baseline | Anchor16 | Retention |
|---|---:|---:|---:|
| `pope_accuracy` | 0.7913 | 0.8122 | 102.64% |
| `pope_f1_score` | 0.8835 | 0.8964 | 101.46% |
| `pope_precision` | 1.0000 | 1.0000 | 100.00% |
| `pope_recall` | 0.7913 | 0.8122 | 102.64% |
| `pope_yes_ratio` | 1.0000 | 1.0000 | 100.00% |

### `textvqa_val`

- `submission` is present in both logs as a generated artifact field, but it is not a numeric score and is therefore excluded from retention calculation.

## Quick Takeaways

- Best retention:
  - `vizwiz_vqa_val`: `104.80%`
  - `pope_accuracy`: `102.64%`
  - `mmmu_val`: `98.77%`
  - `scienceqa_img`: `98.41%`
  - `ai2d`: `96.71%`
- Moderate drop:
  - `gqa`: `93.43%`
  - `textvqa_val`: `82.18%`
  - `chartqa` overall: `79.38%`
- Largest drop in this 9-benchmark set:
  - `docvqa_val` `anls`: `68.72%`

## Notes

- These numbers are copied from the final metric tables printed in `shard_0/run.log` for the two completed DP runs.
- This document intentionally uses the log tables directly, matching the debugging workflow used in this session.
