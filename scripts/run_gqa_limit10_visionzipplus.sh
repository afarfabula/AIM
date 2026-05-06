#!/bin/bash
set -eo pipefail

# Runs GQA (limit=10) for visionzipplus on a GPU worker.
# Uses local HF hub cache under the repo to avoid re-downloading model/vision tower weights.

cd /mlx_devbox/users/quyanyi/playground/AIM

PYTHON="${PYTHON:-/home/tiger/miniforge3/envs/aim/bin/python}"
TASK="${TASK:-gqa}"                     # gqa | vqav2_val | textvqa_val | pope | ...
STRATEGY="${STRATEGY:-visionzipplus}"   # visionzip | visionzipplus
LIMIT="${LIMIT:-10}"                    # number of samples, or "none"/0 for full run
BATCH_SIZE="${BATCH_SIZE:-1}"           # lmms-eval batch size passed to model.generate
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}" # eager | sdpa | flash_attention_2
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="/mlx_devbox/users/quyanyi/playground/AIM/logs/${TASK}_limit${LIMIT}_${STRATEGY}_${TS}"

# Match run_llava15_benchmarks.sh cache layout for datasets/hub.
CACHE_ROOT="${CACHE_ROOT:-/mlx_devbox/users/quyanyi/playground/AIM/hf_cache_shared}"
export HF_HOME="$CACHE_ROOT"
export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"

# Force model/vision-tower weights to the local hub cache (fastest). You can override from env.
export HF_ENDPOINT="${HF_ENDPOINT:-http://huggingface-proxy-sg.byted.org}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$CACHE_ROOT/hub}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"

# Some datasets/configs use `token: True` in datasets.load_dataset(). huggingface_hub
# primarily reads HF_TOKEN; allow users to provide only HUGGINGFACE_HUB_TOKEN as well.
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-http://sys-proxy-rd-relay.byted.org:8118}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-http://sys-proxy-rd-relay.byted.org:8118}}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="/tmp/triton_cache"
mkdir -p "$TRITON_CACHE_DIR" "$OUT_DIR"

MODEL_ARGS="pretrained=liuhaotian/llava-v1.5-7b,conv_template=vicuna_v1,attn_implementation=${ATTN_IMPLEMENTATION},token_prune_strategy=${STRATEGY}"

echo "OUT_DIR=$OUT_DIR"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "MODEL_ARGS=$MODEL_ARGS"
echo "TASK=$TASK"
echo "STRATEGY=$STRATEGY"
echo "LIMIT=$LIMIT"
echo "BATCH_SIZE=$BATCH_SIZE"

run_eval () {
  # Build args late so STRATEGY can be overridden from env.
  MODEL_ARGS="pretrained=liuhaotian/llava-v1.5-7b,conv_template=vicuna_v1,attn_implementation=${ATTN_IMPLEMENTATION},token_prune_strategy=${STRATEGY}"

  echo "---- Running lmms_eval (task=$TASK, limit=$LIMIT, strategy=$STRATEGY) ----"
  if [ "$LIMIT" = "0" ] || [ "$LIMIT" = "none" ]; then
    "$PYTHON" -m lmms_eval \
      --model llava \
      --model_args "$MODEL_ARGS" \
      --tasks "$TASK" \
      --batch_size "$BATCH_SIZE" \
      --output_path "$OUT_DIR"
  else
    "$PYTHON" -m lmms_eval \
      --model llava \
      --model_args "$MODEL_ARGS" \
      --tasks "$TASK" \
      --batch_size "$BATCH_SIZE" \
      --limit "$LIMIT" \
      --output_path "$OUT_DIR"
  fi
}

extract_score () {
  local results_json
  results_json="$(find "$OUT_DIR" -maxdepth 4 -type f -name '*_results.json' | sort | tail -n 1 || true)"
  if [ -z "$results_json" ]; then
    echo "No results json found under $OUT_DIR"
    return 2
  fi
  echo "RESULT_JSON=$results_json"
  "$PYTHON" - "$results_json" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
# lmms-eval results format varies; handle common patterns.
gqa = data.get("gqa") or data.get("results", {}).get("gqa") or {}
def pick(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None
# Some results store metrics as "exact_match,none".
score = pick(gqa, ["exact_match", "accuracy", "acc", "score"])
key = None
if score is None:
    for k, v in gqa.items():
        if k.startswith("exact_match"):
            key, score = k, v
            break
print("GQA_SCORE_KEY=", key or ("exact_match" if "exact_match" in gqa else ("accuracy" if "accuracy" in gqa else ("acc" if "acc" in gqa else ("score" if "score" in gqa else "unknown")))))
print("GQA_SCORE=", score)
PY
}

LOG_FILE="$OUT_DIR/run.log"

# Preflight: verify flash-attn import when requested.
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

# Always run online for datasets, but still keep model weights local via HF_HUB_CACHE.
# Also cleanup any broken ".incomplete" dataset dirs from previous interrupted downloads.
export HF_DATASETS_OFFLINE=0 HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0

GQA_CACHE_ROOT="$HF_DATASETS_CACHE/lmms-lab___gqa"
if [ "$TASK" = "gqa" ] && [ -d "$GQA_CACHE_ROOT" ]; then
  echo "---- Cleaning incomplete dataset dirs under $GQA_CACHE_ROOT ----"
  find "$GQA_CACHE_ROOT" -maxdepth 6 -name "*.incomplete" -print -exec rm -rf {} + || true
fi

set +e
run_eval 2>&1 | tee "$LOG_FILE"
RC=${PIPESTATUS[0]}
set -e
if [ "$RC" -ne 0 ]; then
  echo "Run failed (exit=$RC). See $LOG_FILE"
  exit "$RC"
fi

echo "---- Extracting score ----"
extract_score
