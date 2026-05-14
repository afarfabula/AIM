#!/bin/bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/AIM"
cd "$ROOT"

usage() {
  cat <<'EOF'
Easy entrypoint for common LLaVA evaluation runs on a worker.

Usage:
  bash executable/runa_llava.sh [options]

Common examples:
  bash executable/runa_llava.sh
  bash executable/runa_llava.sh --bench gqa --limit 100
  bash executable/runa_llava.sh --bench bench5 --strategy none
  bash executable/runa_llava.sh --model-preset llava15 --strategy bishemethod_v2stage_anchor16_litefirst_t64 --bench gqa --limit 50
  bash executable/runa_llava.sh --model-preset llava_ov --bench videomme --limit 1 --strategy none
  bash executable/runa_llava.sh --model /some/local/model --bench gqa --strategy none

Options:
  --model-preset <name>   Model family preset: llava15 | llava_ov
  --model <path_or_repo>  Override model path/repo directly
  --worker-id <id>        Login to a worker first, then run remotely
  --bench <name>          Benchmark alias:
                          gqa | mme | pope | scienceqa | mmmu | docvqa | seedbench
                          bench5 | full7 | videomme
  --tasks <csv>           Custom tasks csv for llava15, e.g. gqa,mme,pope
  --strategy <name>       Token reduction strategy
  --limit <n|none>        lmms-eval limit
  --batch-size <n>        Batch size
  --attn <name>           sdpa | eager | flash_attention_2 | auto
  --num-gpus <n>          For llava15 only: use virtual data parallel across N GPUs
  --bootstrap <0|1>       Auto create /tmp/aim_venv if missing (default: 1)
  --verbosity <level>     INFO | DEBUG | ...
  --video-decode-backend  For llava_ov video tasks, e.g. pyav
  --max-frames <n>        For llava_ov video tasks
  --torch-dtype <name>    For llava_ov, e.g. float16 | bfloat16
  --dry-run               Print resolved command only
  -h, --help              Show this help

Notes:
  - On llava15 runs, this script prefers the persistent local model directory:
    /mlx_devbox/users/quyanyi/playground/AIM/hf_cache_shared/local_models/liuhaotian__llava-v1.5-7b
  - On a fresh worker, leave --bootstrap 1 so /tmp/aim_venv is created automatically.
EOF
}

MODEL_PRESET="${MODEL_PRESET:-llava15}"
MODEL_OVERRIDE="${MODEL_OVERRIDE:-}"
WORKER_ID="${WORKER_ID:-}"
BENCH_ALIAS="${BENCH_ALIAS:-gqa}"
TASKS_CSV_OVERRIDE="${TASKS_CSV_OVERRIDE:-}"
STRATEGY="${STRATEGY:-bishemethod_v2stage_anchor16_litefirst}"
LIMIT="${LIMIT:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"
NUM_GPUS="${NUM_GPUS:-1}"
BOOTSTRAP="${BOOTSTRAP:-1}"
VERBOSITY="${VERBOSITY:-INFO}"
VIDEO_DECODE_BACKEND="${VIDEO_DECODE_BACKEND:-}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-preset)
      MODEL_PRESET="$2"; shift 2 ;;
    --model)
      MODEL_OVERRIDE="$2"; shift 2 ;;
    --worker-id)
      WORKER_ID="$2"; shift 2 ;;
    --bench)
      BENCH_ALIAS="$2"; shift 2 ;;
    --tasks)
      TASKS_CSV_OVERRIDE="$2"; shift 2 ;;
    --strategy)
      STRATEGY="$2"; shift 2 ;;
    --limit)
      LIMIT="$2"; shift 2 ;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    --attn)
      ATTN_IMPLEMENTATION="$2"; shift 2 ;;
    --num-gpus)
      NUM_GPUS="$2"; shift 2 ;;
    --bootstrap)
      BOOTSTRAP="$2"; shift 2 ;;
    --verbosity)
      VERBOSITY="$2"; shift 2 ;;
    --video-decode-backend)
      VIDEO_DECODE_BACKEND="$2"; shift 2 ;;
    --max-frames)
      MAX_FRAMES_NUM="$2"; shift 2 ;;
    --torch-dtype)
      TORCH_DTYPE="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -n "$WORKER_ID" && "${RUNA_LLAVA_REMOTE:-0}" != "1" ]]; then
  REMOTE_CMD=(
    env RUNA_LLAVA_REMOTE=1
    HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-}"
    HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
    bash executable/runa_llava.sh
    --model-preset "$MODEL_PRESET"
    --bench "$BENCH_ALIAS"
    --strategy "$STRATEGY"
    --limit "$LIMIT"
    --batch-size "$BATCH_SIZE"
    --attn "$ATTN_IMPLEMENTATION"
    --bootstrap "$BOOTSTRAP"
    --verbosity "$VERBOSITY"
    --max-frames "$MAX_FRAMES_NUM"
    --torch-dtype "$TORCH_DTYPE"
    --num-gpus "$NUM_GPUS"
  )
  if [[ -n "$MODEL_OVERRIDE" ]]; then
    REMOTE_CMD+=(--model "$MODEL_OVERRIDE")
  fi
  if [[ -n "$TASKS_CSV_OVERRIDE" ]]; then
    REMOTE_CMD+=(--tasks "$TASKS_CSV_OVERRIDE")
  fi
  if [[ -n "$VIDEO_DECODE_BACKEND" ]]; then
    REMOTE_CMD+=(--video-decode-backend "$VIDEO_DECODE_BACKEND")
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    REMOTE_CMD+=(--dry-run)
  fi

  quoted_remote_cmd="$(printf '%q ' "${REMOTE_CMD[@]}")"
  exec mlx worker login "$WORKER_ID" -- "cd $ROOT && $quoted_remote_cmd"
fi

CACHE_ROOT="${CACHE_ROOT:-$ROOT/hf_cache_shared}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface-proxy-sg.byted.org}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_ROOT/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$CACHE_ROOT/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
mkdir -p "$TRITON_CACHE_DIR"

PYTHON="${PYTHON:-/tmp/aim_venv/bin/python}"
LOCAL_LLAVA15_MODEL="${LOCAL_LLAVA15_MODEL:-$CACHE_ROOT/local_models/liuhaotian__llava-v1.5-7b}"

if [[ "$DRY_RUN" == "1" ]] && [[ ! -x "$PYTHON" ]]; then
  echo "[dry-run] $PYTHON is missing; a real run would bootstrap /tmp/aim_venv first."
elif [[ "$BOOTSTRAP" == "1" ]] && [[ ! -x "$PYTHON" ]]; then
  echo "[bootstrap] /tmp/aim_venv missing, creating it first..."
  bash "$ROOT/scripts/bootstrap_worker_venv.sh"
fi

if [[ "$DRY_RUN" != "1" ]] && [[ ! -x "$PYTHON" ]]; then
  echo "Python not found at $PYTHON" >&2
  echo "Set --bootstrap 1 or export PYTHON=/path/to/python" >&2
  exit 2
fi

resolve_llava15_model() {
  if [[ -n "$MODEL_OVERRIDE" ]]; then
    echo "$MODEL_OVERRIDE"
  elif [[ -d "$LOCAL_LLAVA15_MODEL" ]]; then
    echo "$LOCAL_LLAVA15_MODEL"
  else
    echo "liuhaotian/llava-v1.5-7b"
  fi
}

resolve_tasks_csv() {
  if [[ -n "$TASKS_CSV_OVERRIDE" ]]; then
    echo "$TASKS_CSV_OVERRIDE"
    return
  fi
  case "$BENCH_ALIAS" in
    gqa) echo "gqa" ;;
    mme) echo "mme" ;;
    pope) echo "pope" ;;
    scienceqa|scienceqa_img) echo "scienceqa_img" ;;
    mmmu|mmmu_val) echo "mmmu_val" ;;
    docvqa|docvqa_val) echo "docvqa_val" ;;
    seedbench) echo "seedbench" ;;
    bench5) echo "gqa,mme,pope,scienceqa_img,mmmu_val" ;;
    full7) echo "gqa,mme,pope,scienceqa_img,docvqa_val,mmmu_val,seedbench" ;;
    *)
      echo "Unsupported bench alias for llava15: $BENCH_ALIAS" >&2
      exit 2 ;;
  esac
}

if [[ "$MODEL_PRESET" == "llava15" ]]; then
  MODEL_NAME="$(resolve_llava15_model)"
  TASKS_CSV="$(resolve_tasks_csv)"
  if [[ "$ATTN_IMPLEMENTATION" == "auto" ]]; then
    ATTN_IMPLEMENTATION="sdpa"
  fi

  echo "=== runa_llava ==="
  echo "MODEL_PRESET=$MODEL_PRESET"
  echo "MODEL_NAME=$MODEL_NAME"
  echo "TASKS_CSV=$TASKS_CSV"
  echo "STRATEGY=$STRATEGY"
  echo "LIMIT=$LIMIT"
  echo "BATCH_SIZE=$BATCH_SIZE"
  echo "ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION"
  echo "NUM_GPUS=$NUM_GPUS"
  echo "PYTHON=$PYTHON"
  echo "HF_ENDPOINT=$HF_ENDPOINT"

  if [[ "$NUM_GPUS" -gt 1 ]]; then
    CMD=(bash "$ROOT/scripts/run_lmms_virtual_dp.sh")
  else
    CMD=(bash "$ROOT/scripts/run_5_benchmarks_one_strategy.sh")
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN: MODEL_NAME=\"$MODEL_NAME\" STRATEGY=\"$STRATEGY\" TASKS_CSV=\"$TASKS_CSV\" LIMIT=\"$LIMIT\" BATCH_SIZE=\"$BATCH_SIZE\" ATTN_IMPLEMENTATION=\"$ATTN_IMPLEMENTATION\" NUM_GPUS=\"$NUM_GPUS\" PYTHON=\"$PYTHON\" HF_ENDPOINT=\"$HF_ENDPOINT\" ${CMD[*]}"
    exit 0
  fi

  if [[ "$NUM_GPUS" -gt 1 ]]; then
    MODEL_ARGS="pretrained=${MODEL_NAME},conv_template=vicuna_v1,attn_implementation=${ATTN_IMPLEMENTATION},token_prune_strategy=${STRATEGY}"
    TASKS="$TASKS_CSV" \
    MODEL="llava" \
    MODEL_ARGS="$MODEL_ARGS" \
    LIMIT="$LIMIT" \
    BATCH_SIZE="$BATCH_SIZE" \
    VIRTUAL_WORLD_SIZE="$NUM_GPUS" \
    PYTHON="$PYTHON" \
    HF_ENDPOINT="$HF_ENDPOINT" \
    HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-}" \
    HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" \
    "${CMD[@]}"
  else
    MODEL_NAME="$MODEL_NAME" \
    STRATEGY="$STRATEGY" \
    TASKS_CSV="$TASKS_CSV" \
    LIMIT="$LIMIT" \
    BATCH_SIZE="$BATCH_SIZE" \
    ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" \
    PYTHON="$PYTHON" \
    HF_ENDPOINT="$HF_ENDPOINT" \
    "${CMD[@]}"
  fi
  exit 0
fi

if [[ "$MODEL_PRESET" == "llava_ov" ]]; then
  case "$BENCH_ALIAS" in
    gqa|mme|pope|scienceqa_img|videomme)
      TASK_NAME="$BENCH_ALIAS" ;;
    scienceqa)
      TASK_NAME="scienceqa_img" ;;
    *)
      echo "Unsupported bench alias for llava_ov: $BENCH_ALIAS" >&2
      echo "Use one task only, e.g. gqa or videomme" >&2
      exit 2 ;;
  esac

  if [[ -n "$MODEL_OVERRIDE" ]]; then
    MODEL_NAME_OVERRIDE="$MODEL_OVERRIDE"
  else
    MODEL_NAME_OVERRIDE="lmms-lab/llava-onevision-qwen2-7b-ov"
  fi
  if [[ "$ATTN_IMPLEMENTATION" == "auto" ]]; then
    ATTN_IMPLEMENTATION="sdpa"
  fi

  echo "=== runa_llava ==="
  echo "MODEL_PRESET=$MODEL_PRESET"
  echo "MODEL=$MODEL_NAME_OVERRIDE"
  echo "TASK=$TASK_NAME"
  echo "STRATEGY=$STRATEGY"
  echo "LIMIT=$LIMIT"
  echo "BATCH_SIZE=$BATCH_SIZE"
  echo "ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION"
  echo "TORCH_DTYPE=$TORCH_DTYPE"
  echo "VIDEO_DECODE_BACKEND=$VIDEO_DECODE_BACKEND"
  echo "MAX_FRAMES_NUM=$MAX_FRAMES_NUM"
  echo "PYTHON=$PYTHON"
  echo "HF_ENDPOINT=$HF_ENDPOINT"

  CMD=(
    bash "$ROOT/scripts/run_llava_ov_task.sh"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN: MODEL=\"$MODEL_NAME_OVERRIDE\" TASK=\"$TASK_NAME\" STRATEGY=\"$STRATEGY\" LIMIT=\"$LIMIT\" BATCH_SIZE=\"$BATCH_SIZE\" ATTN_IMPLEMENTATION=\"$ATTN_IMPLEMENTATION\" TORCH_DTYPE=\"$TORCH_DTYPE\" VIDEO_DECODE_BACKEND=\"$VIDEO_DECODE_BACKEND\" MAX_FRAMES_NUM=\"$MAX_FRAMES_NUM\" PYTHON=\"$PYTHON\" HF_ENDPOINT=\"$HF_ENDPOINT\" ${CMD[*]}"
    exit 0
  fi

  MODEL="$MODEL_NAME_OVERRIDE" \
  TASK="$TASK_NAME" \
  STRATEGY="$STRATEGY" \
  LIMIT="$LIMIT" \
  BATCH_SIZE="$BATCH_SIZE" \
  ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" \
  TORCH_DTYPE="$TORCH_DTYPE" \
  VIDEO_DECODE_BACKEND="$VIDEO_DECODE_BACKEND" \
  MAX_FRAMES_NUM="$MAX_FRAMES_NUM" \
  VERBOSITY="$VERBOSITY" \
  PYTHON="$PYTHON" \
  HF_ENDPOINT="$HF_ENDPOINT" \
  "${CMD[@]}"
  exit 0
fi

echo "Unsupported MODEL_PRESET: $MODEL_PRESET" >&2
echo "Supported presets: llava15, llava_ov" >&2
exit 2
