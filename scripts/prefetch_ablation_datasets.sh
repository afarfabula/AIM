#!/bin/bash
set -euo pipefail

# Prefetch datasets needed for ablations into a shared cache directory (does not require a worker).
# Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN in env for token-gated datasets (MME).

export http_proxy="${http_proxy:-http://sys-proxy-rd-relay.byted.org:3128}"
export https_proxy="${https_proxy:-http://sys-proxy-rd-relay.byted.org:3128}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"

export no_proxy="${no_proxy:-localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,127.0.0.1,127.0.0.0/8,169.254.0.0/16,100.64.0.0/10,172.16.0.0/12,192.168.0.0/16,10.0.0.0/8,::1,fe80::/10,fd00::/8}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"

export HF_ENDPOINT="${HF_ENDPOINT:-http://huggingface-proxy-sg.byted.org}"

CACHE_ROOT="${CACHE_ROOT:-/mlx_devbox/users/quyanyi/playground/AIM/hf_cache_shared}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_ROOT/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$CACHE_ROOT/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"

mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

python3 /mlx_devbox/users/quyanyi/playground/AIM/scripts/prefetch_ablation_datasets.py

echo "CACHE_ROOT=$CACHE_ROOT"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"

