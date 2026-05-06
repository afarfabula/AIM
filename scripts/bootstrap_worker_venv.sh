#!/bin/bash
set -euo pipefail

# Create a lightweight venv on worker without conda.
# Reuse system site-packages so we keep the existing torch + flash-attn.

cd /mlx_devbox/users/quyanyi/playground/AIM

VENV_DIR="${VENV_DIR:-/tmp/aim_venv}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "[venv] create $VENV_DIR (reuse system site-packages)"
  python3 -m venv --system-site-packages "$VENV_DIR"
fi

PY="$VENV_DIR/bin/python"

echo "[venv] python=$($PY -c 'import sys; print(sys.executable)')"
$PY -m pip install -U pip setuptools wheel

echo "[install] editable packages (AIM + customized deps)"
$PY -m pip install -e .
$PY -m pip install -e other_packages/transformers
$PY -m pip install -e other_packages/lmms-eval
$PY -m pip install -e other_packages/qwen-vl-utils
 # Some llava modules import ipdb unconditionally in this repo.
 $PY -m pip install -U ipdb

echo "[check] lmms_eval import"
$PY -c "import lmms_eval; print('lmms_eval=OK', getattr(lmms_eval,'__version__',None))"

echo "[check] flash_attn import (from system site-packages)"
$PY -c "import flash_attn; print('flash_attn=OK')"

echo "DONE_VENV=$VENV_DIR"
