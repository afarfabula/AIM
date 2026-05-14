#!/bin/bash
set -euo pipefail

echo "=== find python envs ==="
for p in \
  /home/tiger/miniforge3/envs/aim/bin/python \
  /opt/conda/envs/aim/bin/python \
  /tmp/aim_venv/bin/python \
  /mlx_devbox/users/quyanyi/playground/AIM/.venv/bin/python \
  /usr/local/bin/python3
do
  if [ -x "$p" ]; then
    echo "FOUND=$p"
    "$p" -V || true
    "$p" - <<'PY'
mods = ["torch", "flash_attn", "lmms_eval", "llava", "ipdb"]
for m in mods:
    try:
        __import__(m)
        print(f"{m}=OK")
    except Exception as e:
        print(f"{m}=FAIL {e!r}")
PY
    echo "---"
  fi
done
