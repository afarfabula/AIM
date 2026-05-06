#!/bin/bash
set -eo pipefail

cd /mlx_devbox/users/quyanyi/playground/AIM

export http_proxy="${http_proxy:-http://sys-proxy-rd-relay.byted.org:3128}"
export https_proxy="${https_proxy:-http://sys-proxy-rd-relay.byted.org:3128}"
export no_proxy="${no_proxy:-localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,127.0.0.1,127.0.0.0/8,169.254.0.0/16,100.64.0.0/10,172.16.0.0/12,192.168.0.0/16,10.0.0.0/8,::1,fe80::/10,fd00::/8}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"

CACHE_ROOT="${CACHE_ROOT:-/mlx_devbox/users/quyanyi/playground/AIM/hf_cache_shared}"
export HF_HOME="$CACHE_ROOT"
export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
export HF_HUB_CACHE="$CACHE_ROOT/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_ASSETS_CACHE="$CACHE_ROOT/assets"
export HF_MODULES_CACHE="$CACHE_ROOT/modules"
export XDG_CACHE_HOME="$CACHE_ROOT/.cache"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton_cache}"
mkdir -p "$TRITON_CACHE_DIR" "$HF_DATASETS_CACHE" "$HF_ASSETS_CACHE" "$HF_MODULES_CACHE"

PYTHON="${PYTHON:-/home/tiger/miniforge3/envs/aim/bin/python}"
MODEL_ARGS="${MODEL_ARGS:-pretrained=liuhaotian/llava-v1.5-7b,conv_template=vicuna_v1,attn_implementation=eager,token_prune_strategy=visionzipplus}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TASKS_CSV="${TASKS_CSV:-mmbench_en_test,textvqa_val,docvqa_val,pope,gqa,scienceqa_img}"
IFS=',' read -r -a TASKS <<< "$TASKS_CSV"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="/mlx_devbox/users/quyanyi/playground/AIM/logs/visionzipplus_six_benchmarks/${RUN_ID}"
mkdir -p "$RUN_ROOT"

echo "========================================="
echo "VisionZipPlus Six Benchmarks"
echo "========================================="
echo "RUN_ROOT: $RUN_ROOT"
echo "MODEL_ARGS: $MODEL_ARGS"
echo "TASKS_CSV: $TASKS_CSV"
echo "WORKER_HOSTNAME: $(hostname)"
echo "========================================="

run_task() {
  local task_name="$1"
  local task_dir="$RUN_ROOT/$task_name"
  local log_file="$task_dir/run.log"

  mkdir -p "$task_dir"

  echo ""
  echo "-----------------------------------------"
  echo "TASK: $task_name"
  echo "TASK_DIR: $task_dir"
  echo "LOG_FILE: $log_file"
  echo "-----------------------------------------"

  set +e
  "$PYTHON" -m lmms_eval \
    --model llava \
    --model_args "$MODEL_ARGS" \
    --tasks "$task_name" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$task_dir" \
    2>&1 | tee "$log_file"
  local rc=${PIPESTATUS[0]}
  set -e

  if [ "$rc" -ne 0 ]; then
    echo "❌ FAILED: $task_name (exit=$rc)"
    return "$rc"
  fi

  if grep -qE "Error during evaluation:|Tasks not found:|Traceback \\(most recent call last\\):|LocalTokenNotFoundError|RetryError\\[" "$log_file"; then
    echo "❌ FAILED: $task_name (error found in log)"
    return 1
  fi

  echo "✅ DONE: $task_name"
  find "$task_dir" -maxdepth 3 -type f -name "*_results.json" -print | tail -n 1 || true
}

for task in "${TASKS[@]}"; do
  run_task "$task"
done

echo ""
echo "========================================="
echo "All six benchmarks finished."
echo "RUN_ROOT: $RUN_ROOT"
echo "========================================="
