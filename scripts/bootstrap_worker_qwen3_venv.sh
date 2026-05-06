#!/bin/bash
set -euo pipefail

# Create a dedicated Qwen3-VL worker venv without touching the vendored transformers.
# Reuse system site-packages so worker torch / flash-attn stay available.

cd /mlx_devbox/users/quyanyi/playground/AIM

VENV_DIR="${VENV_DIR:-/tmp/aim_qwen3_venv}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==4.57.1}"
ACCELERATE_SPEC="${ACCELERATE_SPEC:-accelerate>=0.29.1}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "[venv] create $VENV_DIR (reuse system site-packages)"
  python3 -m venv --system-site-packages "$VENV_DIR"
fi

PY="$VENV_DIR/bin/python"

echo "[venv] python=$($PY -c 'import sys; print(sys.executable)')"
$PY -m pip install -U pip setuptools wheel

echo "[install] qwen3 runtime deps"
$PY -m pip install -U "$TRANSFORMERS_SPEC" "$ACCELERATE_SPEC" "qwen-vl-utils[decord]" hf_transfer

echo "[install] local editable packages (no vendored transformers)"
$PY -m pip install -e . --no-deps
$PY -m pip install -e other_packages/lmms-eval

# Some llava modules import ipdb unconditionally in this repo.
$PY -m pip install -U ipdb

echo "[check] qwen3 transformers import"
$PY - <<'PY'
import transformers
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, Qwen3VLProcessor

print("transformers=", transformers.__version__)
print("Qwen3VLForConditionalGeneration=OK", Qwen3VLForConditionalGeneration.__name__)
print("Qwen3VLMoeForConditionalGeneration=OK", Qwen3VLMoeForConditionalGeneration.__name__)
print("Qwen3VLProcessor=OK", Qwen3VLProcessor.__name__)
PY

echo "[check] lmms_eval qwen3 wrapper import"
$PY - <<'PY'
from lmms_eval.models import get_model

model_cls = get_model("qwen3_vl")
print("lmms_eval_qwen3_vl=OK", model_cls.__name__)
PY

echo "[check] flash_attn import (from system site-packages)"
$PY -c "import flash_attn; print('flash_attn=OK')"

echo "DONE_VENV=$VENV_DIR"
