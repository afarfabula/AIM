# Debug Session: qwen3-gqa-fpe

Status: [OPEN]

## Symptom

- `bash executable/runa_qwen3.sh --worker-id 3816716 --bench gqa --limit 100` can finish imports, model snapshot fetch, and checkpoint shard loading.
- It reaches `Running generate_until requests`.
- It then crashes with `Floating point exception (core dumped)` / exit code `136`.

## Expected

- Qwen3-VL should complete at least a small `gqa` run without SIGFPE after model loading.

## Scope

- Model: `Qwen/Qwen3-VL-30B-A3B-Instruct`
- Task: `gqa`
- Worker: `3816716`

## Falsifiable Hypotheses

1. `flash_attention_2` on this worker / driver stack is unstable for Qwen3-VL generation, and switching to `sdpa` should avoid the FPE.
2. The wrapper passes an unsafe dtype combination into generation, and explicitly setting model dtype should change or eliminate the failure mode.
3. The crash happens before the first generated token inside model forward, so a minimal single-sample/single-batch run with extra instrumentation should show the last successful stage before SIGFPE.
4. The invalid generation kwargs warning (`temperature`, `top_k`) correlates with an unsupported generation path for this wrapper, and removing/normalizing those kwargs should change behavior.
5. The worker environment is fine for model loading but fails on the first multimodal generate call, meaning preprocessing succeeds and the failure is inside generation kernels rather than dataset/model download.

## Evidence Collected

- User log shows successful shard loading and request building before `generate_until`.
- User log shows immediate SIGFPE at the lmms_eval invocation line after generation starts.
- User log also shows two strong precursor signals right before the crash:
  - `You are attempting to use Flash Attention 2 without specifying a torch dtype. This might lead to unexpected behaviour`
  - `The following generation flags are not valid and may be ignored: ['temperature', 'top_k']`

## Hypothesis Status

| ID | Hypothesis | Status | Evidence Summary |
|----|------------|--------|------------------|
| A | `flash_attention_2` is unstable here | INCONCLUSIVE | FPE still points near generate, but the pre-fix log also warned about unsafe FA2 setup. |
| B | Missing explicit dtype contributes to the crash | PARTIALLY CONFIRMED | Pre-fix log explicitly warned that FA2 was used without a specified dtype. |
| C | Crash happens at or after the first `model.generate(...)` | CONFIRMED | Pre-fix log completed shard loading and request building, then crashed immediately after `Running generate_until requests`. |
| D | Invalid generation kwargs contribute to the failing path | PARTIALLY CONFIRMED | Pre-fix log warned that `temperature/top_k` were invalid and may be ignored. |
| E | Download / preprocessing is not the root cause | CONFIRMED | Model fetch and shard loading completed before the crash. |

## Instrumentation Added

- Added runtime debug points in `lmms_eval/models/qwen3_vl.py` for:
  - batch entry and raw generation kwargs
  - prepared input tensor shapes / dtype / device
  - final generation kwargs immediately before `model.generate(...)`
  - successful return from `model.generate(...)`

## Minimal Fix Applied

- Qwen3-VL now sets explicit `dtype=bfloat16` by default when `attn_implementation=flash_attention_2`.
- Non-sampling generation no longer forces `temperature/top_p` into `model.generate(...)`.
- Qwen3 snapshot download remains on the hardened retry path added earlier.

## Next Evidence Step

- Re-run on worker `3816716` with `limit=1` and compare:
  - whether the previous FA2/dtype warning disappears
  - whether the process gets beyond the first generation step
