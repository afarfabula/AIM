#!/bin/bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/AIM"
cd "$ROOT"

usage() {
  cat <<'EOF'
Easy entrypoint for Qwen3-VL benchmark runs on a worker.

Usage:
  bash executable/runa_qwen3.sh [options]

Common examples:
  bash executable/runa_qwen3.sh
  bash executable/runa_qwen3.sh --bench gqa --limit 100
  bash executable/runa_qwen3.sh --bench mme --limit 20
  bash executable/runa_qwen3.sh --model Qwen/Qwen3-VL-8B-Instruct --bench gqa --limit 50
  bash executable/runa_qwen3.sh --bench scienceqa --attn sdpa
  bash executable/runa_qwen3.sh --worker-id 3816716 --bench gqa --limit 100

Options:
  --model <path_or_repo>  Override model path/repo
  --bench <name>          Task alias:
                          gqa | mme | pope | scienceqa | mmmu | docvqa | seedbench
  --limit <n|none>        lmms-eval limit
  --batch-size <n>        Batch size
  --attn <name>           flash_attention_2 | sdpa | eager | auto
  --worker-id <id>        If set, auto run this script on that worker
  --bootstrap <0|1>       Auto create /tmp/aim_qwen3_venv if missing (default: 1)
  --max-pixels <n>        Max pixels for processor
  --min-pixels <n>        Min pixels for processor
  --dry-run               Print resolved command only
  -h, --help              Show this help

Notes:
  - This script uses the dedicated Qwen3 worker venv:
    /tmp/aim_qwen3_venv
  - On a fresh worker, leave --bootstrap 1 so the venv is created automatically.
EOF
}

MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
BENCH_ALIAS="${BENCH_ALIAS:-gqa}"
LIMIT="${LIMIT:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"
BOOTSTRAP="${BOOTSTRAP:-1}"
MAX_PIXELS="${MAX_PIXELS:-2007040}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
WORKER_ID="${WORKER_ID:-}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"; shift 2 ;;
    --bench)
      BENCH_ALIAS="$2"; shift 2 ;;
    --limit)
      LIMIT="$2"; shift 2 ;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    --attn)
      ATTN_IMPLEMENTATION="$2"; shift 2 ;;
    --worker-id)
      WORKER_ID="$2"; shift 2 ;;
    --bootstrap)
      BOOTSTRAP="$2"; shift 2 ;;
    --max-pixels)
      MAX_PIXELS="$2"; shift 2 ;;
    --min-pixels)
      MIN_PIXELS="$2"; shift 2 ;;
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

resolve_task() {
  case "$BENCH_ALIAS" in
    gqa) echo "gqa" ;;
    mme) echo "mme" ;;
    pope) echo "pope" ;;
    scienceqa|scienceqa_img) echo "scienceqa_img" ;;
    mmmu|mmmu_val) echo "mmmu_val" ;;
    docvqa|docvqa_val) echo "docvqa_val" ;;
    seedbench) echo "seedbench" ;;
    *)
      echo "Unsupported qwen3 bench alias: $BENCH_ALIAS" >&2
      exit 2 ;;
  esac
}

TASK="$(resolve_task)"

CACHE_ROOT="${CACHE_ROOT:-$ROOT/hf_cache_shared}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface-proxy-sg.byted.org}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_ROOT/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$CACHE_ROOT/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
HF_ENDPOINT="$(printf '%s' "$HF_ENDPOINT" | tr -d '`' | xargs)"
export HF_ENDPOINT
unset TRANSFORMERS_CACHE
mkdir -p "$TRITON_CACHE_DIR"

PYTHON="${PYTHON:-/tmp/aim_qwen3_venv/bin/python}"
if [[ "$ATTN_IMPLEMENTATION" == "auto" ]]; then
  ATTN_IMPLEMENTATION="flash_attention_2"
fi

if [[ -n "$WORKER_ID" && "${RUNA_QWEN3_REMOTE:-0}" != "1" ]]; then
  REMOTE_CMD=(
    env RUNA_QWEN3_REMOTE=1
    bash executable/runa_qwen3.sh
    --model "$MODEL"
    --bench "$BENCH_ALIAS"
    --limit "$LIMIT"
    --batch-size "$BATCH_SIZE"
    --attn "$ATTN_IMPLEMENTATION"
    --bootstrap "$BOOTSTRAP"
    --max-pixels "$MAX_PIXELS"
    --min-pixels "$MIN_PIXELS"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    REMOTE_CMD+=(--dry-run)
  fi
  printf -v REMOTE_JOINED '%q ' "${REMOTE_CMD[@]}"
  exec mlx worker login "$WORKER_ID" -- "cd $ROOT && ${REMOTE_JOINED}"
fi

if [[ "$DRY_RUN" == "1" ]] && [[ ! -x "$PYTHON" ]]; then
  echo "[dry-run] $PYTHON is missing; a real run would bootstrap /tmp/aim_qwen3_venv first."
elif [[ "$BOOTSTRAP" == "1" ]] && [[ ! -x "$PYTHON" ]]; then
  echo "[bootstrap] /tmp/aim_qwen3_venv missing, creating it first..."
  bash "$ROOT/scripts/bootstrap_worker_qwen3_venv.sh"
fi

if [[ "$DRY_RUN" != "1" ]] && [[ ! -x "$PYTHON" ]]; then
  echo "Python not found at $PYTHON" >&2
  echo "Set --bootstrap 1 or export PYTHON=/path/to/python" >&2
  exit 2
fi

CUDA_OK=1
if [[ "$DRY_RUN" != "1" ]]; then
  if ! "$PYTHON" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    CUDA_OK=0
  fi
fi

if [[ "$CUDA_OK" == "0" ]]; then
  echo "当前环境没有可用 CUDA/GPU。" >&2
  echo "这个脚本用于 worker 上的 Qwen3-VL 推理。" >&2
  echo "请先执行 \`mlx worker login <worker_id>\` 再运行，或直接用:" >&2
  echo "  bash executable/runa_qwen3.sh --worker-id <worker_id>" >&2
  exit 2
fi

echo "=== runa_qwen3 ==="
echo "MODEL=$MODEL"
echo "TASK=$TASK"
echo "LIMIT=$LIMIT"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION"
echo "MAX_PIXELS=$MAX_PIXELS"
echo "MIN_PIXELS=$MIN_PIXELS"
echo "WORKER_ID=${WORKER_ID:-}"
echo "PYTHON=$PYTHON"
echo "HF_ENDPOINT=$HF_ENDPOINT"

CMD=(bash "$ROOT/scripts/run_qwen3_vl_task.sh")
if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN: MODEL=\"$MODEL\" TASK=\"$TASK\" LIMIT=\"$LIMIT\" BATCH_SIZE=\"$BATCH_SIZE\" ATTN_IMPLEMENTATION=\"$ATTN_IMPLEMENTATION\" MAX_PIXELS=\"$MAX_PIXELS\" MIN_PIXELS=\"$MIN_PIXELS\" PYTHON=\"$PYTHON\" HF_ENDPOINT=\"$HF_ENDPOINT\" ${CMD[*]}"
  exit 0
fi

MODEL="$MODEL" \
TASK="$TASK" \
LIMIT="$LIMIT" \
BATCH_SIZE="$BATCH_SIZE" \
ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" \
MAX_PIXELS="$MAX_PIXELS" \
MIN_PIXELS="$MIN_PIXELS" \
PYTHON="$PYTHON" \
HF_ENDPOINT="$HF_ENDPOINT" \
"${CMD[@]}"
