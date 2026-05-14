#!/bin/bash
set -euo pipefail

cd /mlx_devbox/users/quyanyi/playground/AIM

echo "=== python stack ==="
echo "PY=$(command -v python3 || true)"
python3 -V || true

python3 - <<'PY'
import sys

print("exe=", sys.executable)
for mod in ["torch", "flash_attn", "lmms_eval", "llava"]:
    try:
        __import__(mod)
        print(f"{mod}=OK")
    except Exception as e:
        print(f"{mod}=FAIL {e!r}")
PY
