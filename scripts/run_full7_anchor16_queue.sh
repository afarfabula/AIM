#!/bin/bash
set -euo pipefail

cd /mlx_devbox/users/quyanyi/playground/AIM

ONE_SH="${ONE_SH:-/mlx_devbox/users/quyanyi/playground/AIM/scripts/run_full7_benchmarks_one_strategy.sh}"
QUEUE_NAME="${QUEUE_NAME:-full7_anchor16}"
TS="$(date +%Y%m%d_%H%M%S)"
QUEUE_ROOT="${QUEUE_ROOT:-/mlx_devbox/users/quyanyi/playground/AIM/logs/ablation_queue_${QUEUE_NAME}_${TS}}"
mkdir -p "$QUEUE_ROOT"

STRATEGIES=(
  bishemethod_v2stage_anchor16_litefirst
  bishemethod_v2stage_anchor16_litefirst_t192
)

LABELS=(
  "anchor16 token=96"
  "anchor16 token=192"
)

echo "QUEUE_ROOT=$QUEUE_ROOT"
echo "ONE_SH=$ONE_SH"

for i in "${!STRATEGIES[@]}"; do
  idx=$(printf "%02d" $((i + 1)))
  strategy="${STRATEGIES[$i]}"
  label="${LABELS[$i]}"
  out_root="$QUEUE_ROOT/${idx}_${strategy}"
  mkdir -p "$out_root"

  printf '%s\t%s\t%s\n' "$idx" "$strategy" "$label" >> "$QUEUE_ROOT/manifest.tsv"
  echo "==== QUEUE RUN idx=$idx strategy=$strategy label=$label out=$out_root ===="

  set +e
  STRATEGY="$strategy" OUT_ROOT="$out_root" "$ONE_SH" 2>&1 | tee "$out_root/queue_run.log"
  rc=${PIPESTATUS[0]}
  set -e

  echo "$rc" > "$out_root/queue_exit_code.txt"
  if [ "$rc" -ne 0 ]; then
    echo "==== QUEUE FAIL idx=$idx strategy=$strategy exit=$rc ===="
  else
    echo "==== QUEUE OK idx=$idx strategy=$strategy ===="
  fi
done

python3 - "$QUEUE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

queue_root = Path(sys.argv[1])
rows = []
for entry in sorted(p for p in queue_root.iterdir() if p.is_dir()):
    summary_path = entry / "summary.json"
    queue_exit = entry / "queue_exit_code.txt"
    row = {
        "strategy_dir": entry.name,
        "strategy": entry.name.split("_", 1)[1] if "_" in entry.name else entry.name,
        "queue_exit_code": int(queue_exit.read_text().strip()) if queue_exit.exists() else None,
        "summary_json": str(summary_path) if summary_path.exists() else None,
    }
    rows.append(row)

summary = {
    "queue_root": str(queue_root),
    "count": len(rows),
    "rows": rows,
}
(queue_root / "queue_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"QUEUE_SUMMARY_JSON={queue_root / 'queue_summary.json'}")
PY

echo "QUEUE_DONE=$QUEUE_ROOT"
