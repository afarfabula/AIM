#!/bin/bash
set -euo pipefail

cd /mlx_devbox/users/quyanyi/playground/AIM

PYTHON="${PYTHON:-/tmp/aim_qwen3_venv/bin/python}"
MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
TASK="${TASK:-gqa}"
LIMIT="${LIMIT:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
MAX_PIXELS="${MAX_PIXELS:-2007040}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="/mlx_devbox/users/quyanyi/playground/AIM/logs/${TASK}_limit${LIMIT}_qwen3_vl_${TS}"

CACHE_ROOT="${CACHE_ROOT:-/mlx_devbox/users/quyanyi/playground/AIM/hf_cache_shared}"
export HF_HOME="$CACHE_ROOT"
export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
export HF_ENDPOINT="${HF_ENDPOINT:-http://huggingface-proxy-sg.byted.org}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$CACHE_ROOT/hub}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"

export HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-http://sys-proxy-rd-relay.byted.org:3128}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-http://sys-proxy-rd-relay.byted.org:3128}}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"
export no_proxy="${no_proxy:-localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,127.0.0.1,127.0.0.0/8,169.254.0.0/16,100.64.0.0/10,172.16.0.0/12,192.168.0.0/16,10.0.0.0/8,::1,fe80::/10,fd00::/8}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="/tmp/triton_cache"
mkdir -p "$TRITON_CACHE_DIR" "$OUT_DIR"

MODEL_ARGS="pretrained=${MODEL},attn_implementation=${ATTN_IMPLEMENTATION},max_pixels=${MAX_PIXELS},min_pixels=${MIN_PIXELS}"

echo "OUT_DIR=$OUT_DIR"
echo "PYTHON=$PYTHON"
echo "MODEL=$MODEL"
echo "TASK=$TASK"
echo "LIMIT=$LIMIT"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "MODEL_ARGS=$MODEL_ARGS"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"

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
