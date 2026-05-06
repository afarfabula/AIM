#!/bin/bash
set -euo pipefail

PY="${PYTHON:-/tmp/aim_venv/bin/python}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="/mlx_devbox/users/quyanyi/playground/AIM/logs/debug_generate_batch_${TS}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"

{
  echo "PY=$PY"
  $PY -V

  echo "=== batch=1 sdpa ==="
  $PY /mlx_devbox/users/quyanyi/playground/AIM/scripts/debug_llava_generate_batch.py --batch 1 --attn-implementation sdpa

  echo "=== batch=2 sdpa ==="
  $PY /mlx_devbox/users/quyanyi/playground/AIM/scripts/debug_llava_generate_batch.py --batch 2 --attn-implementation sdpa

  echo "DONE_LOG=$LOG"
} 2>&1 | tee "$LOG"
