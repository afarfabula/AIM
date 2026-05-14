#!/bin/bash
set -euo pipefail

# Run 8 benchmarks via lmms-eval and write a single summary (json + tsv).
# Intended to be executed on a GPU worker via `mlx worker login ... -- env ... bash this_script`.
#
# Benchmarks (8, no GPT-judge):
#   gqa, mme, pope, scienceqa_img, vqav2_val, textvqa_val, mmmu_val, seedbench
#
# Notes:
# - Some datasets require HF token (token: True in yaml); set HF_TOKEN/HUGGINGFACE_HUB_TOKEN in env.

cd /mlx_devbox/users/quyanyi/playground/AIM

PYTHON="${PYTHON:-/tmp/aim_venv/bin/python}"
STRATEGY="${STRATEGY:-bishemethod_v2stage}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
LIMIT="${LIMIT:-none}"  # none|0 => full; or an int for debug

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="/mlx_devbox/users/quyanyi/playground/AIM/logs/bench11_${STRATEGY}_${TS}"
mkdir -p "$OUT_ROOT"

# Cache layout (worker local /tmp is strongly recommended for speed).
CACHE_ROOT="${CACHE_ROOT:-/mlx_devbox/users/quyanyi/playground/AIM/hf_cache_shared}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_ROOT/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$CACHE_ROOT/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"

export HF_ENDPOINT="${HF_ENDPOINT:-http://huggingface-proxy-sg.byted.org}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
mkdir -p "$TRITON_CACHE_DIR"

# Model config
LOCAL_LLAVA15_MODEL="${LOCAL_LLAVA15_MODEL:-$CACHE_ROOT/local_models/liuhaotian__llava-v1.5-7b}"
if [ -d "$LOCAL_LLAVA15_MODEL" ]; then
  DEFAULT_MODEL_NAME="$LOCAL_LLAVA15_MODEL"
else
  DEFAULT_MODEL_NAME="liuhaotian/llava-v1.5-7b"
fi
MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL_NAME}"
MODEL_ARGS="pretrained=${MODEL_NAME},conv_template=vicuna_v1,attn_implementation=${ATTN_IMPLEMENTATION},token_prune_strategy=${STRATEGY}"

echo "OUT_ROOT=$OUT_ROOT"
echo "PYTHON=$PYTHON"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "LOCAL_LLAVA15_MODEL=$LOCAL_LLAVA15_MODEL"
echo "MODEL_ARGS=$MODEL_ARGS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "LIMIT=$LIMIT"

if [ "$ATTN_IMPLEMENTATION" = "flash_attention_2" ]; then
  echo "---- Preflight: checking flash_attn ----"
  "$PYTHON" -c "import flash_attn; print('flash_attn=OK')"
fi

TASKS=(
  gqa
  mme
  pope
  scienceqa_img
  vqav2_val
  textvqa_val
  mmmu_val
  seedbench
)

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
    echo "==== FAIL task=$task exit=$rc (see $out_dir/run.log) ===="
  else
    echo "==== OK task=$task ===="
  fi
}

for t in "${TASKS[@]}"; do
  run_one "$t"
done

echo "---- Summarizing ----"
"$PYTHON" - "$OUT_ROOT" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])

TASKS = [
    "gqa",
    "mme",
    "pope",
    "scienceqa_img",
    "vqav2_val",
    "textvqa_val",
    "mmmu_val",
    "seedbench",
]

def find_results_json(task_dir: Path):
    # lmms-eval writes: <out>/<model_sanitized>/<date>_results.json
    candidates = sorted(task_dir.rglob("*_results.json"))
    return candidates[-1] if candidates else None

def pick_metric(d: dict, prefixes):
    # Metric keys often look like "exact_match,none" or "gpt_eval_score,none"
    for p in prefixes:
        if p in d:
            return p, d[p]
    for k, v in d.items():
        for p in prefixes:
            if k.startswith(p):
                return k, v
    return None, None

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

summary = {"root": str(root), "tasks": {}, "notes": []}

for task in TASKS:
    tdir = root / task
    rec = {"task": task, "exit_code": None, "results_json": None, "metrics": {}}
    exit_code_path = tdir / "exit_code.txt"
    if exit_code_path.exists():
        rec["exit_code"] = int(exit_code_path.read_text().strip() or "0")
    rj = find_results_json(tdir)
    if rj is None:
        rec["results_json"] = None
        summary["tasks"][task] = rec
        continue
    rec["results_json"] = str(rj)
    data = json.loads(rj.read_text())
    task_blob = (data.get("results", {}) or {}).get(task, {})

    if task in ("gqa", "vqav2_val", "textvqa_val", "scienceqa_img"):
        k, v = pick_metric(task_blob, ["exact_match"])
        rec["metrics"]["exact_match"] = {"key": k, "value": to_float(v)}
    elif task == "mmmu_val":
        k, v = pick_metric(task_blob, ["mmmu_acc"])
        rec["metrics"]["mmmu_acc"] = {"key": k, "value": to_float(v)}
    elif task == "mme":
        k1, v1 = pick_metric(task_blob, ["mme_percetion_score"])
        k2, v2 = pick_metric(task_blob, ["mme_cognition_score"])
        p = to_float(v1); c = to_float(v2)
        rec["metrics"]["mme_perception"] = {"key": k1, "value": p}
        rec["metrics"]["mme_cognition"] = {"key": k2, "value": c}
        rec["metrics"]["mme_total"] = {"value": (p + c) if (p is not None and c is not None) else None}
    elif task == "pope":
        for m in ["pope_accuracy", "pope_precision", "pope_recall", "pope_f1_score", "pope_yes_ratio"]:
            k, v = pick_metric(task_blob, [m])
            rec["metrics"][m] = {"key": k, "value": to_float(v)}
    elif task == "seedbench":
        for m in ["seed_all", "seed_image", "seed_video"]:
            k, v = pick_metric(task_blob, [m])
            rec["metrics"][m] = {"key": k, "value": to_float(v)}
    else:
        # Fallback: dump first 10 scalar metrics
        for k, v in list(task_blob.items())[:10]:
            rec["metrics"][k] = {"value": v}

    summary["tasks"][task] = rec

out_json = root / "summary.json"
out_tsv = root / "summary.tsv"
out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

# Flat TSV: one row per task with a single "primary" score where possible
def primary_score(task, rec):
    m = rec.get("metrics", {})
    if task in ("gqa", "vqav2_val", "textvqa_val", "scienceqa_img"):
        return m.get("exact_match", {}).get("value")
    if task == "mmmu_val":
        return m.get("mmmu_acc", {}).get("value")
    if task == "mme":
        return m.get("mme_total", {}).get("value")
    if task == "seedbench":
        return m.get("seed_all", {}).get("value")
    if task == "pope":
        return m.get("pope_f1_score", {}).get("value")
    return None

lines = ["task\tprimary_score\texit_code\tresults_json"]
for task in TASKS:
    rec = summary["tasks"][task]
    lines.append(f"{task}\t{primary_score(task, rec)}\t{rec.get('exit_code')}\t{rec.get('results_json')}")
out_tsv.write_text("\n".join(lines) + "\n")

print("WROTE", out_json)
print("WROTE", out_tsv)
PY

echo "DONE: $OUT_ROOT"
echo "SUMMARY_JSON=$OUT_ROOT/summary.json"
echo "SUMMARY_TSV=$OUT_ROOT/summary.tsv"
