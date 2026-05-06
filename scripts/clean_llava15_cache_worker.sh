#!/bin/bash
set -euo pipefail

TARGET="/tmp/aim_hf_home/hub/models--liuhaotian--llava-v1.5-7b"
echo "removing: $TARGET"
rm -rf -- "$TARGET"
echo "done"
ls -lah /tmp/aim_hf_home/hub | head -n 30 || true

