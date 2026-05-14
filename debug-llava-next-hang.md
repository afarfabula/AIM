# Debug Session: llava-next-hang

Status: [OPEN]

## Symptom

- `llava-next` full eval appears stuck on `Model Responding`
- User reports no GPU utilization and no new log lines
- Example log under investigation:
  - `logs/gqa_mmmu_val_scienceqa_img_limitnone_llava_next13b_bishemethod_v2stage_anchor16_litefirst_next_t320_20260511_185555/run.log`

## Hypotheses

1. The process is still alive but blocked in CPU-side postprocessing or task metric code, so GPU util drops to zero while log progress stalls.
2. The process is alive but stuck on one hard sample inside `generate_until`, causing the progress bar to stop for a long time.
3. The process has exited or deadlocked, and the displayed progress line is just the last flushed TTY output.
4. The worker still has a running Python process, but CUDA context is idle because the code path is waiting on host-side I/O or synchronization.

## Evidence Log

- Local log file for `t320` stopped updating at `2026-05-11 19:34:17 +0800`
- Local log file for `t640` stopped updating at `2026-05-11 19:34:17 +0800`
- `t320` last visible progress: `9954/15495`
- `t640` last visible progress: `7514/15495`
- Both logs stop immediately after verbose per-sample token-pruning prints
- Worker GPU memory remains allocated on both GPUs, but `utilization.gpu = 0%`
- Worker still shows many `python -m lmms_eval` processes alive with long elapsed time and near-zero CPU

## Current Assessment

- Hypothesis 2 is weakened: two independent jobs stopping at the exact same timestamp is unlikely to be caused by two unrelated hard samples.
- Hypothesis 1 is partially weakened: there is no sign of continued CPU-side metric aggregation or log progress after the stop point.
- Hypothesis 3 and 4 are currently strongest: the processes appear alive but blocked, likely on a shared host-side resource such as output/logging or synchronization.

## Next Actions

- Inspect live process list on worker
- Inspect `nvidia-smi` utilization and running compute processes
- Tail the affected `run.log`
- Determine whether this is real hang, long-tail generation, or host-side blocking
