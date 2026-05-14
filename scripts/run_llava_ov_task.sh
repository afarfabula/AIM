#!/bin/bash
set -euo pipefail

cd /mlx_devbox/users/quyanyi/playground/AIM

PYTHON="${PYTHON:-/tmp/aim_venv/bin/python}"
MODEL="${MODEL:-lmms-lab/llava-onevision-qwen2-7b-ov}"
TASK="${TASK:-gqa}"
STRATEGY="${STRATEGY:-bishemethod_v2stage_anchor16_litefirst}"
LIMIT="${LIMIT:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
VERBOSITY="${VERBOSITY:-INFO}"
VIDEO_DECODE_BACKEND="${VIDEO_DECODE_BACKEND:-}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
DEBUG_VIDEO_INPUTS="${DEBUG_VIDEO_INPUTS:-0}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MODEL_NAME="${MODEL_NAME:-llava_qwen}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="/mlx_devbox/users/quyanyi/playground/AIM/logs/${TASK}_limit${LIMIT}_${STRATEGY}_llava_ov_${TS}"

CACHE_ROOT="${CACHE_ROOT:-/mlx_devbox/users/quyanyi/playground/AIM/hf_cache_shared}"
export HF_ENDPOINT="${HF_ENDPOINT:-http://huggingface-proxy-sg.byted.org}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$CACHE_ROOT/hub}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"

USE_TMP_DATASETS="${USE_TMP_DATASETS:-0}"
if [[ "$TASK" == video* ]]; then
  USE_TMP_DATASETS=1
  # Prefer the internal HF endpoint for large video datasets.
  if [ -z "${HF_ENDPOINT:-}" ]; then
    export HF_ENDPOINT="http://huggingface.byted.org"
  fi
  export LMMS_EVAL_SNAPSHOT_MAX_WORKERS="${LMMS_EVAL_SNAPSHOT_MAX_WORKERS:-2}"
fi

if [ "$USE_TMP_DATASETS" = "1" ]; then
  TMP_HF_HOME="${TMP_HF_HOME:-/tmp/aim_hf_home}"
  export HF_HOME="$TMP_HF_HOME"
  export HF_DATASETS_CACHE="$TMP_HF_HOME/datasets"
else
  export HF_HOME="$CACHE_ROOT"
  export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
fi

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
export LMMS_EVAL_DEBUG_VIDEO_INPUTS="${LMMS_EVAL_DEBUG_VIDEO_INPUTS:-$DEBUG_VIDEO_INPUTS}"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="/tmp/triton_cache"
mkdir -p "$TRITON_CACHE_DIR" "$OUT_DIR" "$HF_DATASETS_CACHE"

MODEL_ARGS="pretrained=${MODEL},conv_template=${CONV_TEMPLATE},model_name=${MODEL_NAME},attn_implementation=${ATTN_IMPLEMENTATION},torch_dtype=${TORCH_DTYPE},token_prune_strategy=${STRATEGY}"

# Video tasks are very large and tend to exercise different kernels/paths. We've observed eager attention
# can crash with FPE during generation on some worker images; prefer sdpa unless explicitly overridden.
if [[ "$TASK" == video* ]] && [[ "$ATTN_IMPLEMENTATION" == "eager" ]]; then
  echo "WARN: video task + ATTN_IMPLEMENTATION=eager is unstable; switching eager -> sdpa"
  ATTN_IMPLEMENTATION="sdpa"
  MODEL_ARGS="pretrained=${MODEL},conv_template=${CONV_TEMPLATE},model_name=${MODEL_NAME},attn_implementation=${ATTN_IMPLEMENTATION},token_prune_strategy=${STRATEGY}"
fi

# Video decode: decord occasionally triggers native crashes on some workers. Prefer pyav by default.
if [[ "$TASK" == video* ]] && [[ -z "${VIDEO_DECODE_BACKEND}" ]]; then
  VIDEO_DECODE_BACKEND="pyav"
fi
if [[ "$TASK" == video* ]]; then
  MODEL_ARGS="${MODEL_ARGS},video_decode_backend=${VIDEO_DECODE_BACKEND},max_frames_num=${MAX_FRAMES_NUM}"
fi

# LLaVA-OneVision uses eager-attention path for its native AIM pruning inside Qwen2.
# When we explicitly enable our external token_prune_strategy, avoid stacking both
# mechanisms together because that path is unstable and can crash with FPE.
if [ "${STRATEGY}" != "none" ] && [ "${ATTN_IMPLEMENTATION}" = "eager" ]; then
  echo "WARN: token_prune_strategy=${STRATEGY} conflicts with OV native eager-pruning; switching ATTN_IMPLEMENTATION=eager -> sdpa"
  ATTN_IMPLEMENTATION="sdpa"
  MODEL_ARGS="pretrained=${MODEL},conv_template=${CONV_TEMPLATE},model_name=${MODEL_NAME},attn_implementation=${ATTN_IMPLEMENTATION},token_prune_strategy=${STRATEGY}"
fi

echo "OUT_DIR=$OUT_DIR"
echo "PYTHON=$PYTHON"
echo "MODEL=$MODEL"
echo "TASK=$TASK"
echo "STRATEGY=$STRATEGY"
echo "LIMIT=$LIMIT"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "VERBOSITY=$VERBOSITY"
echo "TORCH_DTYPE=$TORCH_DTYPE"
echo "LMMS_EVAL_DEBUG_VIDEO_INPUTS=$LMMS_EVAL_DEBUG_VIDEO_INPUTS"
if [[ "$TASK" == video* ]]; then
  echo "VIDEO_DECODE_BACKEND=$VIDEO_DECODE_BACKEND"
  echo "MAX_FRAMES_NUM=$MAX_FRAMES_NUM"
fi
echo "MODEL_ARGS=$MODEL_ARGS"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

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
  echo "---- Running lmms_eval (model=llava_onevision, task=$TASK, limit=$LIMIT, strategy=$STRATEGY) ----"
  if [ "$LIMIT" = "0" ] || [ "$LIMIT" = "none" ]; then
    "$PYTHON" -m lmms_eval \
      --model llava_onevision \
      --model_args "$MODEL_ARGS" \
      --tasks "$TASK" \
      --batch_size "$BATCH_SIZE" \
      --verbosity "$VERBOSITY" \
      --output_path "$OUT_DIR"
  else
    "$PYTHON" -m lmms_eval \
      --model llava_onevision \
      --model_args "$MODEL_ARGS" \
      --tasks "$TASK" \
      --batch_size "$BATCH_SIZE" \
      --limit "$LIMIT" \
      --verbosity "$VERBOSITY" \
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
