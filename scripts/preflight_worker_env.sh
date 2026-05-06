#!/bin/bash
set -euo pipefail

echo "=== preflight ==="
echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "PY=$(command -v python3 || true)"
python3 -V || true

echo "--- torch ---"
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count", torch.cuda.device_count())
PY

echo "--- flash_attn ---"
python3 - <<'PY'
try:
    import flash_attn  # noqa: F401
    print("flash_attn=OK")
except Exception as e:
    print("flash_attn=FAIL", repr(e))
PY

echo "--- env (selected) ---"
env | egrep '^(HTTP_PROXY|HTTPS_PROXY|http_proxy|https_proxy|HF_ENDPOINT|HF_HOME|HF_DATASETS_CACHE|HF_HUB_CACHE|TRANSFORMERS_CACHE|HUGGINGFACE_HUB_CACHE|TRITON_CACHE_DIR|ATTN_IMPLEMENTATION|TASK|STRATEGY|LIMIT|BATCH_SIZE)=' || true

echo "=== end preflight ==="

