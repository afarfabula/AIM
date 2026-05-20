#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

PYTHON="${PYTHON:-$(command -v python3)}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
TASK="${TASK:-gqa}"
LIMIT="${LIMIT:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
MAX_PIXELS="${MAX_PIXELS:-2007040}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-/root/aim_qwen3_logs}"

slugify() {
  local s="$1"
  s="${s//\//_}"
  s="${s//,/+}"
  s="${s// /}"
  s="${s//:/-}"
  s="${s//;/_}"
  s="${s//|/_}"
  s="${s//~/_}"
  s="${s//=/-}"
  s="${s//(/}"
  s="${s//)/}"
  s="${s//[/}"
  s="${s//]/}"
  s="${s//\{/}"
  s="${s//\}/}"
  s="${s//__/_}"
  s="${s//__/_}"
  s="${s//+-/+}"
  printf '%s' "$s"
}

TASK_TAG="$(slugify "$TASK")"
if [ -n "${TOKEN_PRUNE_STRATEGY:-}" ]; then
  TOKEN_REDUCTION_TAG="$(slugify "$TOKEN_PRUNE_STRATEGY")"
  if [ -n "${TOKEN_PRUNE_CONFIG:-}" ]; then
    TOKEN_REDUCTION_TAG="${TOKEN_REDUCTION_TAG}_$(slugify "$TOKEN_PRUNE_CONFIG")"
  fi
else
  TOKEN_REDUCTION_TAG="baseline"
fi

OUT_DIR="${LOG_ROOT}/${TASK_TAG}_limit${LIMIT}_qwen3_vl_${TOKEN_REDUCTION_TAG}_${TS}"

# httpx (used by huggingface_hub) crashes on empty proxy env vars
# (`http_proxy=` -> `Invalid port: ':'`). Treat empty as unset.
for _v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; do
  if [ -n "${!_v+x}" ] && [ -z "${!_v}" ]; then
    unset "$_v"
  fi
done
unset _v

# All HF caches under /tmp.
CACHE_ROOT="${CACHE_ROOT:-/tmp/aim_qwen3_cache}"
export HF_HOME="$CACHE_ROOT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$CACHE_ROOT/hub}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_ROOT/datasets}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
unset TRANSFORMERS_CACHE
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
export QWEN3_MODEL_SNAPSHOT_MAX_WORKERS="${QWEN3_MODEL_SNAPSHOT_MAX_WORKERS:-4}"

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
mkdir -p "$TRITON_CACHE_DIR" "$OUT_DIR" "$CACHE_ROOT" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

MODEL_ARGS="pretrained=${MODEL},attn_implementation=${ATTN_IMPLEMENTATION},max_pixels=${MAX_PIXELS},min_pixels=${MIN_PIXELS}"

if [ -n "${TOKEN_PRUNE_STRATEGY:-}" ]; then
  MODEL_ARGS="${MODEL_ARGS},token_prune_strategy=${TOKEN_PRUNE_STRATEGY}"
fi
if [ -n "${TOKEN_PRUNE_CONFIG:-}" ]; then
  # token_prune_config 不能直接用 `,`，否则会被 lmms-eval 的 model_args 顶层解析拆开。
  # 推荐使用 shell-safe 的 `~` / `|` 作为外层分隔符（同时兼容历史 `;`）。
  MODEL_ARGS="${MODEL_ARGS},token_prune_config=${TOKEN_PRUNE_CONFIG}"
fi

echo "OUT_DIR=$OUT_DIR"
echo "PYTHON=$PYTHON"
echo "MODEL=$MODEL"
echo "TASK=$TASK"
echo "LIMIT=$LIMIT"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "TASK_TAG=$TASK_TAG"
echo "TOKEN_REDUCTION_TAG=$TOKEN_REDUCTION_TAG"
echo "MODEL_ARGS=$MODEL_ARGS"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

echo "---- Preflight: checking qwen3 imports ----"
"$PYTHON" - <<'PY'
import transformers
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, Qwen3VLProcessor

print("transformers=", transformers.__version__)
print("Qwen3VLForConditionalGeneration=OK", Qwen3VLForConditionalGeneration.__name__)
print("Qwen3VLMoeForConditionalGeneration=OK", Qwen3VLMoeForConditionalGeneration.__name__)
print("Qwen3VLProcessor=OK", Qwen3VLProcessor.__name__)
PY

if [ "$ATTN_IMPLEMENTATION" = "flash_attention_2" ]; then
  echo "---- Preflight: checking flash_attn ----"
  "$PYTHON" - <<'PY'
import sys
try:
    import flash_attn  # noqa: F401
    print("flash_attn=OK")
except Exception as e:
    print("flash_attn=FAIL:", repr(e))
    sys.exit(2)
PY
fi

# Pre-fetch model snapshot so the first run isn't blocked inside the eval driver.
echo "---- Pre-download model snapshot ($MODEL) ----"
"$PYTHON" - <<PY
import os
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="$MODEL",
    cache_dir=os.environ.get("HF_HUB_CACHE"),
    endpoint=os.environ.get("HF_ENDPOINT"),
    max_workers=int(os.environ.get("QWEN3_MODEL_SNAPSHOT_MAX_WORKERS", "4")),
    etag_timeout=60,
)
print("MODEL_PATH=", path)
PY

run_eval () {
  echo "---- Running lmms_eval (model=qwen3_vl, task=$TASK, limit=$LIMIT) ----"
  if [ "$LIMIT" = "0" ] || [ "$LIMIT" = "none" ]; then
    "$PYTHON" -m lmms_eval \
      --model qwen3_vl \
      --model_args "$MODEL_ARGS" \
      --tasks "$TASK" \
      --batch_size "$BATCH_SIZE" \
      --output_path "$OUT_DIR"
  else
    "$PYTHON" -m lmms_eval \
      --model qwen3_vl \
      --model_args "$MODEL_ARGS" \
      --tasks "$TASK" \
      --batch_size "$BATCH_SIZE" \
      --limit "$LIMIT" \
      --output_path "$OUT_DIR"
  fi
}

LOG_FILE="$OUT_DIR/run.log"
# Maintain a stable symlink to the latest run log under LOG_ROOT for easy tailing.
ln -sfn "$LOG_FILE" "$LOG_ROOT/last_run.log"
set +e
run_eval 2>&1 | tee "$LOG_FILE"
RC=${PIPESTATUS[0]}
set -e
if [ "$RC" -ne 0 ]; then
  echo "Run failed (exit=$RC). See $LOG_FILE"
  exit "$RC"
fi

RESULT_JSON="$(find "$OUT_DIR" -maxdepth 4 -type f -name '*_results.json' | sort | tail -n 1 || true)"
echo "RESULT_JSON=$RESULT_JSON"
