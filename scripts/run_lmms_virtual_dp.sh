#!/bin/bash
set -euo pipefail

cd /mlx_devbox/users/quyanyi/playground/AIM

PYTHON="${PYTHON:-/home/tiger/miniforge3/envs/aim/bin/python}"
TASKS="${TASKS:-gqa}"
MODEL="${MODEL:-llava}"
MODEL_ARGS="${MODEL_ARGS:-pretrained=liuhaotian/llava-v1.5-7b,conv_template=vicuna_v1,attn_implementation=eager,token_prune_strategy=bishemethod_v2stage}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT="${LIMIT:-200}"
VIRTUAL_WORLD_SIZE="${VIRTUAL_WORLD_SIZE:-2}"
VIRTUAL_DEVICE="${VIRTUAL_DEVICE:-cuda:0}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-/mlx_devbox/users/quyanyi/playground/AIM/logs/virtual_dp_${TASKS//,/__}_${TS}}"
CACHE_ROOT="${CACHE_ROOT:-/tmp/aim_hf_home}"

export http_proxy="${http_proxy:-http://sys-proxy-rd-relay.byted.org:3128}"
export https_proxy="${https_proxy:-http://sys-proxy-rd-relay.byted.org:3128}"
export HF_HOME="$CACHE_ROOT"
export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
export HF_HUB_CACHE="/mlx_devbox/users/quyanyi/playground/AIM"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
export LD_LIBRARY_PATH="/home/tiger/miniforge3/envs/aim/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$RUN_DIR" "$TRITON_CACHE_DIR"

echo "RUN_DIR=$RUN_DIR"
echo "TASKS=$TASKS"
echo "MODEL=$MODEL"
echo "MODEL_ARGS=$MODEL_ARGS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "LIMIT=$LIMIT"
echo "VIRTUAL_WORLD_SIZE=$VIRTUAL_WORLD_SIZE"
echo "VIRTUAL_DEVICE=$VIRTUAL_DEVICE"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"

pids=()
for ((rank=0; rank<VIRTUAL_WORLD_SIZE; rank++)); do
  shard_dir="$RUN_DIR/shard_${rank}"
  mkdir -p "$shard_dir"

  cmd=(
    "$PYTHON" -m lmms_eval
    --model "$MODEL"
    --model_args "$MODEL_ARGS"
    --tasks "$TASKS"
    --batch_size "$BATCH_SIZE"
    --output_path "$shard_dir"
    --log_samples
    --virtual_world_size "$VIRTUAL_WORLD_SIZE"
    --virtual_rank "$rank"
    --virtual_device "$VIRTUAL_DEVICE"
  )

  if [[ "$LIMIT" != "none" && "$LIMIT" != "0" ]]; then
    cmd+=(--limit "$LIMIT")
  fi

  echo "Launching shard $rank -> $shard_dir"
  "${cmd[@]}" >"$shard_dir/run.log" 2>&1 &
  pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "At least one shard failed. Check shard logs under $RUN_DIR."
  exit 1
fi

"$PYTHON" tools/merge_virtual_lmms_eval.py \
  --run_dir "$RUN_DIR" \
  --model "$MODEL"

echo "Virtual DP run finished: $RUN_DIR"
