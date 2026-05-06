#!/bin/bash
set -euo pipefail

echo "== cache roots (du -sh) =="
du -sh /tmp/aim_hf_home /tmp/aim_hf_home/datasets /tmp/aim_hf_home/hub /tmp/triton_cache 2>/dev/null || true
echo

echo "== datasets (top-level) =="
ls -lah /tmp/aim_hf_home/datasets 2>/dev/null | head -n 50 || true
echo

echo "== datasets size (top 30) =="
du -sh /tmp/aim_hf_home/datasets/* 2>/dev/null | sort -h | tail -n 30 || true
echo

echo "== hub models (top-level) =="
ls -lah /tmp/aim_hf_home/hub 2>/dev/null | head -n 80 || true
echo

echo "== hub models size (top 30) =="
du -sh /tmp/aim_hf_home/hub/models--* 2>/dev/null | sort -h | tail -n 30 || true
echo

echo "== triton cache size breakdown (top 30) =="
du -sh /tmp/triton_cache/* 2>/dev/null | sort -h | tail -n 30 || true

