# Debug Session: six-bench-startup

- Status: OPEN
- Date: 2026-04-22
- Goal: 确认 `visionzipplus` 跑 6 个 benchmark 时，是否已经稳定进入 `Model Responding`，若没有，定位卡点

## User Symptom

- 用户要求“要看到稳定的 model respond 才算任务结束”
- 当前在单张 A100 上串行跑 `MMBench / TextVQA / DocVQA / POPE / GQA / ScienceQA`
- 现象是任务已启动，但尚未确认进入稳定推理阶段

## Scope

- 仅观察运行时行为
- 暂不修改业务逻辑
- 优先使用现有日志、GPU 利用率、任务输出目录判断状态

## Hypotheses

1. `mmbench_en_test` 正在下载/准备数据集，尚未到请求构建和推理阶段。
2. 模型已加载到 GPU，但进程阻塞在 task init 或 dataset prepare，导致 `GPU-Util` 低。
3. benchmark 实际已开始推理，但日志刷新不及时，需要从 `run.log` 和结果目录双重确认。
4. 任务在 `mmbench_en_test` 阶段失败退出，后续 5 个任务根本还没开始。
5. 代理可用，但某个特定 dataset repo 拉取过慢，导致看起来像“没跑起来”。

## Evidence Log

- 待补充

## Next Actions

1. 检查当前 benchmark 主进程状态
2. 检查 worker 3784970 的 GPU 利用率
3. 检查 `mmbench_en_test/run.log` 是否出现 `Model Responding`
4. 若没有，再判断是数据集阶段还是已失败退出
