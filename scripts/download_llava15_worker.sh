#!/bin/bash
set -euo pipefail

# Pre-download LLaVA-1.5 weights to local /tmp cache on the worker.
# Uses HF mirror + proxy for faster and more stable download.

REPO_ID="${REPO_ID:-liuhaotian/llava-v1.5-7b}"
HF_ENDPOINT="${HF_ENDPOINT:-http://huggingface-proxy-sg.byted.org}"

export http_proxy="${http_proxy:-http://sys-proxy-rd-relay.byted.org:3128}"
export https_proxy="${https_proxy:-http://sys-proxy-rd-relay.byted.org:3128}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"

export no_proxy="${no_proxy:-localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,127.0.0.1,127.0.0.0/8,169.254.0.0/16,100.64.0.0/10,172.16.0.0/12,192.168.0.0/16,10.0.0.0/8,::1,fe80::/10,fd00::/8}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"

export HF_HOME="${HF_HOME:-/tmp/aim_hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

PYTHON="${PYTHON:-python3}"

mkdir -p "$HF_HUB_CACHE"

echo "REPO_ID=$REPO_ID"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "PYTHON=$PYTHON"

# Prefer huggingface-cli if available (supports resume nicely).
if "$PYTHON" -m huggingface_hub.cli download --help >/dev/null 2>&1; then
  echo "---- downloading via huggingface-cli ----"
  HF_ENDPOINT="$HF_ENDPOINT" "$PYTHON" -m huggingface_hub.cli download \
    --resume-download \
    --local-dir "$HF_HUB_CACHE/__llava15_localdir__" \
    --local-dir-use-symlinks False \
    "$REPO_ID"
else
  echo "---- downloading via snapshot_download ----"
  HF_ENDPOINT="$HF_ENDPOINT" "$PYTHON" - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["REPO_ID"]
cache_dir = os.environ["HF_HUB_CACHE"]

path = snapshot_download(
    repo_id=repo_id,
    cache_dir=cache_dir,
    resume_download=True,
    local_files_only=False,
)
print("DOWNLOADED_TO=", path)
PY
fi

echo "---- cache summary ----"
du -sh "$HF_HUB_CACHE" || true
