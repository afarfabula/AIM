#!/bin/bash
set -euo pipefail

cd /mlx_devbox/users/quyanyi/playground/AIM

PYTHON="${PYTHON:-/tmp/aim_venv/bin/python}"
STRATEGY="${STRATEGY:-bishemethod_v2stage_a1_all_s1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
LIMIT="${LIMIT:-none}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="/mlx_devbox/users/quyanyi/playground/AIM/logs/bench5_${STRATEGY}_${TS}"
mkdir -p "$OUT_ROOT"

CACHE_ROOT="${CACHE_ROOT:-/mlx_devbox/users/quyanyi/playground/AIM/hf_cache_shared}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_ROOT/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$CACHE_ROOT/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"

export HF_ENDPOINT="${HF_ENDPOINT:-http://huggingface-proxy-sg.byted.org}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
mkdir -p "$TRITON_CACHE_DIR"

MODEL_NAME="${MODEL_NAME:-liuhaotian/llava-v1.5-7b}"
MODEL_ARGS="pretrained=${MODEL_NAME},conv_template=vicuna_v1,attn_implementation=${ATTN_IMPLEMENTATION},token_prune_strategy=${STRATEGY}"

TASKS=(
  gqa
  mme
  pope
  scienceqa_img
  mmmu_val
)

echo "OUT_ROOT=$OUT_ROOT"
echo "PYTHON=$PYTHON"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "MODEL_ARGS=$MODEL_ARGS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "LIMIT=$LIMIT"

run_one () {
  local task="$1"
  local out_dir="$OUT_ROOT/$task"
  mkdir -p "$out_dir"
  echo "==== RUN task=$task out=$out_dir ===="

  set +e
  if [ "$LIMIT" = "none" ] || [ "$LIMIT" = "0" ]; then
    "$PYTHON" -m lmms_eval \
      --model llava \
      --model_args "$MODEL_ARGS" \
      --tasks "$task" \
      --batch_size "$BATCH_SIZE" \
      --output_path "$out_dir" 2>&1 | tee "$out_dir/run.log"
    rc=${PIPESTATUS[0]}
  else
    "$PYTHON" -m lmms_eval \
      --model llava \
      --model_args "$MODEL_ARGS" \
      --tasks "$task" \
      --batch_size "$BATCH_SIZE" \
      --limit "$LIMIT" \
      --output_path "$out_dir" 2>&1 | tee "$out_dir/run.log"
    rc=${PIPESTATUS[0]}
  fi
  set -e

  echo "$rc" > "$out_dir/exit_code.txt"
  if [ "$rc" -ne 0 ]; then
    echo "==== FAIL task=$task exit=$rc ===="
  else
    echo "==== OK task=$task ===="
  fi
}

for t in "${TASKS[@]}"; do
  run_one "$t"
done

"$PYTHON" - "$OUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
tasks = ["gqa", "mme", "pope", "scienceqa_img", "mmmu_val"]

def find_results_json(task_dir: Path):
    cands = sorted(task_dir.rglob("*_results.json"))
    return cands[-1] if cands else None

def pick_prefix(d, prefix):
    if prefix in d:
        return prefix, d[prefix]
    for k, v in d.items():
        if k.startswith(prefix):
            return k, v
    return None, None

summary = {"root": str(root), "tasks": {}}
for task in tasks:
    tdir = root / task
    rec = {"exit_code": None, "results_json": None, "metrics": {}}
    ep = tdir / "exit_code.txt"
    if ep.exists():
        rec["exit_code"] = int(ep.read_text().strip() or "0")
    rj = find_results_json(tdir)
    if rj:
        rec["results_json"] = str(rj)
        data = json.loads(rj.read_text())
        blob = (data.get("results", {}) or {}).get(task, {})
        if task in ("gqa", "scienceqa_img"):
            k, v = pick_prefix(blob, "exact_match")
            rec["metrics"]["exact_match"] = {"key": k, "value": v}
        elif task == "mme":
            for p in ("mme_percetion_score", "mme_cognition_score"):
                k, v = pick_prefix(blob, p)
                rec["metrics"][p] = {"key": k, "value": v}
        elif task == "pope":
            for p in ("pope_accuracy", "pope_precision", "pope_recall", "pope_f1_score", "pope_yes_ratio"):
                k, v = pick_prefix(blob, p)
                rec["metrics"][p] = {"key": k, "value": v}
        elif task == "mmmu_val":
            k, v = pick_prefix(blob, "mmmu_acc")
            rec["metrics"]["mmmu_acc"] = {"key": k, "value": v}
    summary["tasks"][task] = rec

(root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print("SUMMARY_JSON=", root / "summary.json")
PY

echo "DONE=$OUT_ROOT"
