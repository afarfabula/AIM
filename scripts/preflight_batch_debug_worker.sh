#!/bin/bash
set -euo pipefail

echo "worker_ok"
echo "python3=$(command -v python3 || true)"
python3 -V || true

python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count", torch.cuda.device_count())
PY

python3 - <<'PY'
try:
    import flash_attn  # noqa: F401
    print("flash_attn=OK")
except Exception as e:
    print("flash_attn=FAIL", repr(e))
PY

if [ -x /tmp/aim_venv/bin/python ]; then
  echo "HAS_VENV=/tmp/aim_venv"
  /tmp/aim_venv/bin/python -c "import lmms_eval; print('lmms_eval=OK')"
else
  echo "NO_VENV"
fi

