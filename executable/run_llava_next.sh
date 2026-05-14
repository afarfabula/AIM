#!/bin/bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/AIM"
cd "$ROOT"

usage() {
  cat <<'EOF'
Easy entrypoint for original LLaVA-NeXT (v1.6) runs.

Usage:
  bash executable/run_llava_next.sh [options]

Common examples:
  bash executable/run_llava_next.sh
  bash executable/run_llava_next.sh --bench gqa --limit 100
  bash executable/run_llava_next.sh --tasks gqa,mme,pope,scienceqa_img
  bash executable/run_llava_next.sh --model llava-hf/llava-v1.6-vicuna-13b-hf --bench gqa --limit 20

Options:
  --model <path_or_repo>  Default: liuhaotian/llava-v1.6-vicuna-13b
  --bench <name>          gqa | mme | pope | scienceqa | mmmu | textvqa | mmb
  --tasks <csv>           Run a comma-separated lmms-eval task list directly
  --limit <n|none>        lmms-eval limit
  --batch-size <n>        Batch size (default: 1)
  --attn <name>           sdpa | eager | flash_attention_2 | auto
  --strategy <name>       Token prune strategy (default: none)
  --model-name <name>     Override LLaVA model family name passed into builder
  --bootstrap <0|1>       Auto create /tmp/aim_venv if missing (default: 1)
  --dry-run               Print resolved command only
  -h, --help              Show this help
EOF
}

MODEL="${MODEL:-liuhaotian/llava-v1.6-vicuna-13b}"
MODEL_NAME_ARG="${MODEL_NAME_ARG:-llava-v1.6-vicuna-13b}"
BENCH_ALIAS="${BENCH_ALIAS:-gqa}"
TASKS_CSV="${TASKS_CSV:-}"
LIMIT="${LIMIT:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"
TOKEN_PRUNE_STRATEGY="${TOKEN_PRUNE_STRATEGY:-none}"
BOOTSTRAP="${BOOTSTRAP:-1}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"; shift 2 ;;
    --bench)
      BENCH_ALIAS="$2"; shift 2 ;;
    --tasks)
      TASKS_CSV="$2"; shift 2 ;;
    --model-name)
      MODEL_NAME_ARG="$2"; shift 2 ;;
    --limit)
      LIMIT="$2"; shift 2 ;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    --attn)
      ATTN_IMPLEMENTATION="$2"; shift 2 ;;
    --strategy)
      TOKEN_PRUNE_STRATEGY="$2"; shift 2 ;;
    --bootstrap)
      BOOTSTRAP="$2"; shift 2 ;;
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
    textvqa|textvqa_val) echo "textvqa_val" ;;
    mmb|mmbench|mmbench_en_dev_nogpt) echo "mmbench_en_dev_nogpt" ;;
    *)
      echo "Unsupported bench alias for llava-next: $BENCH_ALIAS" >&2
      exit 2 ;;
  esac
}

if [[ -n "$TASKS_CSV" ]]; then
  TASK="$TASKS_CSV"
else
  TASK="$(resolve_task)"
fi

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
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
HF_ENDPOINT="$(printf '%s' "$HF_ENDPOINT" | tr -d '`' | xargs)"
export HF_ENDPOINT
unset TRANSFORMERS_CACHE
mkdir -p "$TRITON_CACHE_DIR"

# When HF_HOME is redirected to a shared cache, Hugging Face CLI logins saved
# under ~/.cache/huggingface/token are no longer discovered automatically.
# Rehydrate HF_TOKEN from common login locations and mirror the token file into
# the redirected HF_HOME so dataset loaders using token=True can still find it.
if [[ -z "${HF_TOKEN:-}" ]]; then
  for token_file in \
    "${HF_HOME}/token" \
    "${HOME:-}/.cache/huggingface/token" \
    "/home/tiger/.cache/huggingface/token"
  do
    if [[ -f "$token_file" ]]; then
      HF_TOKEN="$(tr -d '\r\n' < "$token_file")"
      export HF_TOKEN
      break
    fi
  done
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  mkdir -p "${HF_HOME}"
  printf '%s\n' "$HF_TOKEN" > "${HF_HOME}/token"
  chmod 600 "${HF_HOME}/token" || true
fi

PYTHON="${PYTHON:-/tmp/aim_venv/bin/python}"
if [[ "$ATTN_IMPLEMENTATION" == "auto" ]]; then
  ATTN_IMPLEMENTATION="sdpa"
fi

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

TS="$(date +%Y%m%d_%H%M%S)"
SAFE_TASK_TAG="$(printf '%s' "$TASK" | tr ',/' '__' | tr -cd '[:alnum:]_.-')"
SAFE_STRATEGY_TAG="$(printf '%s' "$TOKEN_PRUNE_STRATEGY" | tr ',/' '__' | tr -cd '[:alnum:]_.-')"
SAFE_MODEL_TAG="$(printf '%s' "$MODEL_NAME_ARG" | tr ',/' '__' | tr -cd '[:alnum:]_.-')"
OUT_DIR="$ROOT/logs/${SAFE_TASK_TAG}_limit${LIMIT}_${SAFE_MODEL_TAG}_${SAFE_STRATEGY_TAG}_${TS}"
mkdir -p "$OUT_DIR"

resolve_model_snapshot() {
  if [[ -d "$MODEL" ]]; then
    printf '%s\n' "$MODEL"
    return
  fi
  MODEL="$MODEL" "$PYTHON" - <<'PY'
import os
from huggingface_hub import snapshot_download

model = os.environ["MODEL"]
cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
endpoint = os.environ.get("HF_ENDPOINT")
token = os.environ.get("HF_TOKEN") or None

def do_download(**kwargs):
    return snapshot_download(
        repo_id=model,
        cache_dir=cache_dir,
        endpoint=endpoint,
        token=token,
        max_workers=1,
        etag_timeout=60,
        **kwargs,
    )

try:
    print(do_download(resume_download=True, force_download=False))
except Exception as e:
    if "Consistency check failed" not in str(e):
        raise
    print(do_download(resume_download=False, force_download=True))
PY
}

if [[ "$DRY_RUN" == "1" ]]; then
  RESOLVED_MODEL="$MODEL"
else
  RESOLVED_MODEL="$(resolve_model_snapshot)"
fi
MODEL_ARGS="pretrained=${RESOLVED_MODEL},model_name=${MODEL_NAME_ARG},conv_template=vicuna_v1,attn_implementation=${ATTN_IMPLEMENTATION},token_prune_strategy=${TOKEN_PRUNE_STRATEGY},truncate_context=False"

echo "=== run_llava_next ==="
echo "MODEL=$MODEL"
echo "RESOLVED_MODEL=$RESOLVED_MODEL"
echo "MODEL_NAME_ARG=$MODEL_NAME_ARG"
echo "TASK=$TASK"
echo "LIMIT=$LIMIT"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION"
echo "TOKEN_PRUNE_STRATEGY=$TOKEN_PRUNE_STRATEGY"
echo "MODEL_ARGS=$MODEL_ARGS"
echo "PYTHON=$PYTHON"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "OUT_DIR=$OUT_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN: $PYTHON -m lmms_eval --model llava --model_args \"$MODEL_ARGS\" --tasks \"$TASK\" --batch_size \"$BATCH_SIZE\" --limit \"$LIMIT\" --output_path \"$OUT_DIR\""
  exit 0
fi

set +e
if [[ "$LIMIT" == "none" || "$LIMIT" == "0" ]]; then
  "$PYTHON" -m lmms_eval \
    --model llava \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASK" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUT_DIR" 2>&1 | tee "$OUT_DIR/run.log"
  RC=${PIPESTATUS[0]}
else
  "$PYTHON" -m lmms_eval \
    --model llava \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASK" \
    --batch_size "$BATCH_SIZE" \
    --limit "$LIMIT" \
    --output_path "$OUT_DIR" 2>&1 | tee "$OUT_DIR/run.log"
  RC=${PIPESTATUS[0]}
fi
set -e

if [[ "$RC" -ne 0 ]]; then
  echo "Run failed (exit=$RC). See $OUT_DIR/run.log"
  exit "$RC"
fi

RESULT_JSON="$(find "$OUT_DIR" -maxdepth 4 -type f -name '*_results.json' | sort | tail -n 1 || true)"
echo "RESULT_JSON=$RESULT_JSON"
