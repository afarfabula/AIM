#!/bin/bash
set -euo pipefail

# Resolve repo root from this script's location so it works on any machine.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Easy entrypoint for Qwen3-VL benchmark runs.

Usage:
  bash executable/runa_qwen3.sh [options]

Common examples:
  bash executable/runa_qwen3.sh
  bash executable/runa_qwen3.sh --bench gqa --limit 100
  bash executable/runa_qwen3.sh --bench mme --limit 20
  bash executable/runa_qwen3.sh --model Qwen/Qwen3-VL-8B-Instruct --bench gqa --limit 50
  bash executable/runa_qwen3.sh --bench scienceqa --attn sdpa

Options:
  --model <path_or_repo>  Override model path/repo (default: Qwen/Qwen3-VL-8B-Instruct)
  --bench <name|csv>      Task alias or comma-separated aliases:
                          gqa | mme | pope | scienceqa | mmmu | mmbench |
                          textvqa | docvqa | seedbench
  --limit <n|none>        lmms-eval limit
  --batch-size <n>        Batch size
  --attn <name>           flash_attention_2 | sdpa | eager | auto (default: sdpa)
  --max-pixels <n>        Max pixels for processor
  --min-pixels <n>        Min pixels for processor
  --dry-run               Print resolved command only
  -h, --help              Show this help

Notes:
  - HF cache (models + datasets) lives under /tmp/aim_qwen3_cache by default.
  - Models / datasets are auto-downloaded on first run.
EOF
}

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
BENCH_ALIAS="${BENCH_ALIAS:-gqa}"
LIMIT="${LIMIT:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
# H100 here doesn't have flash-attn installed; sdpa is the safe default.
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
MAX_PIXELS="${MAX_PIXELS:-2007040}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
TOKEN_PRUNE_STRATEGY="${TOKEN_PRUNE_STRATEGY:-}"
TOKEN_PRUNE_CONFIG="${TOKEN_PRUNE_CONFIG:-}"
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
    --max-pixels)
      MAX_PIXELS="$2"; shift 2 ;;
    --min-pixels)
      MIN_PIXELS="$2"; shift 2 ;;
    --token-prune-strategy)
      TOKEN_PRUNE_STRATEGY="$2"; shift 2 ;;
    --token-prune-config)
      TOKEN_PRUNE_CONFIG="$2"; shift 2 ;;
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

resolve_single_task() {
  case "$1" in
    gqa) echo "gqa" ;;
    mme) echo "mme" ;;
    pope) echo "pope" ;;
    scienceqa|scienceqa_img) echo "scienceqa_img" ;;
    mmmu|mmmu_val) echo "mmmu_val" ;;
    docvqa|docvqa_val) echo "docvqa_val" ;;
    seedbench) echo "seedbench" ;;
    mmbench|mmbench_en_dev) echo "mmbench_en_dev" ;;
    textvqa|textvqa_val) echo "textvqa_val" ;;
    *)
      echo "Unsupported qwen3 bench alias: $1" >&2
      exit 2 ;;
  esac
}

resolve_tasks() {
  local benches_csv="$1"
  local resolved=()
  local part=""
  IFS=',' read -r -a bench_parts <<< "$benches_csv"
  for part in "${bench_parts[@]}"; do
    part="${part// /}"
    [ -z "$part" ] && continue
    resolved+=("$(resolve_single_task "$part")")
  done
  (IFS=','; printf '%s' "${resolved[*]}")
}

TASK="$(resolve_tasks "$BENCH_ALIAS")"

# httpx (used by huggingface_hub) refuses to parse empty proxy strings such as
# `http_proxy=` and crashes with `Invalid port: ':'`. Drop empty values so they
# behave the same as unset.
for _v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; do
  if [[ -n "${!_v+x}" && -z "${!_v}" ]]; then
    unset "$_v"
  fi
done
unset _v

# All HF caches live under /tmp so models + datasets are downloaded there.
CACHE_ROOT="${CACHE_ROOT:-/tmp/aim_qwen3_cache}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$CACHE_ROOT/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_ROOT/datasets}"
# Use the public HF endpoint by default.
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
unset TRANSFORMERS_CACHE
mkdir -p "$CACHE_ROOT" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR"

PYTHON="${PYTHON:-$(command -v python3)}"

if [[ "$ATTN_IMPLEMENTATION" == "auto" ]]; then
  if "$PYTHON" -c 'import flash_attn' >/dev/null 2>&1; then
    ATTN_IMPLEMENTATION="flash_attention_2"
  else
    ATTN_IMPLEMENTATION="sdpa"
  fi
fi

if [[ "$DRY_RUN" != "1" ]]; then
  if ! "$PYTHON" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    echo "当前环境没有可用 CUDA/GPU." >&2
    exit 2
  fi
fi

echo "=== runa_qwen3 ==="
echo "MODEL=$MODEL"
echo "TASK=$TASK"
echo "LIMIT=$LIMIT"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION"
echo "MAX_PIXELS=$MAX_PIXELS"
echo "MIN_PIXELS=$MIN_PIXELS"
echo "PYTHON=$PYTHON"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "CACHE_ROOT=$CACHE_ROOT"

CMD=(bash "$ROOT/scripts/run_qwen3_vl_task.sh")
if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN: MODEL=\"$MODEL\" TASK=\"$TASK\" LIMIT=\"$LIMIT\" BATCH_SIZE=\"$BATCH_SIZE\" ATTN_IMPLEMENTATION=\"$ATTN_IMPLEMENTATION\" MAX_PIXELS=\"$MAX_PIXELS\" MIN_PIXELS=\"$MIN_PIXELS\" PYTHON=\"$PYTHON\" HF_ENDPOINT=\"$HF_ENDPOINT\" CACHE_ROOT=\"$CACHE_ROOT\" ${CMD[*]}"
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
CACHE_ROOT="$CACHE_ROOT" \
ROOT="$ROOT" \
TOKEN_PRUNE_STRATEGY="$TOKEN_PRUNE_STRATEGY" \
TOKEN_PRUNE_CONFIG="$TOKEN_PRUNE_CONFIG" \
"${CMD[@]}"
