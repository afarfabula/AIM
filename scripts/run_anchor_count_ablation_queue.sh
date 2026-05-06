#!/bin/bash
set -euo pipefail

cd /mlx_devbox/users/quyanyi/playground/AIM

ONE_SH="${ONE_SH:-/mlx_devbox/users/quyanyi/playground/AIM/scripts/run_5_benchmarks_one_strategy.sh}"
QUEUE_NAME="${QUEUE_NAME:-anchor_count_ablation}"
TS="$(date +%Y%m%d_%H%M%S)"
QUEUE_ROOT="${QUEUE_ROOT:-/mlx_devbox/users/quyanyi/playground/AIM/logs/ablation_queue_${QUEUE_NAME}_${TS}}"
mkdir -p "$QUEUE_ROOT"

STRATEGIES=(
  bishemethod_v2stage_anchor9_litefirst
  bishemethod_v2stage_anchor36_litefirst
  bishemethod_v2stage_anchor49_litefirst
)

LABELS=(
  "A2 9-anchor"
  "A2 36-anchor"
  "A2 49-anchor"
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

BASELINES = {
    "gqa": 61.9,
    "mme_total": 1862.0,
    "pope": 85.9,
    "scienceqa_img": 69.5,
    "mmmu_val": 36.3,
}

def pct(x):
    if x is None:
        return None
    x = float(x)
    return x * 100.0 if x <= 1.0 else x

def safe_ret(value, baseline):
    if value is None:
        return None
    return float(value) / float(baseline) * 100.0

rows = []
for entry in sorted(p for p in queue_root.iterdir() if p.is_dir()):
    summary_path = entry / "summary.json"
    queue_exit = entry / "queue_exit_code.txt"
    row = {
        "strategy_dir": entry.name,
        "strategy": entry.name.split("_", 1)[1] if "_" in entry.name else entry.name,
        "queue_exit_code": int(queue_exit.read_text().strip()) if queue_exit.exists() else None,
        "summary_json": str(summary_path) if summary_path.exists() else None,
        "metrics": {},
        "avg_retention": None,
    }
    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        tasks = data.get("tasks", {})
        gqa = pct((((tasks.get("gqa") or {}).get("metrics") or {}).get("exact_match") or {}).get("value"))
        pope = pct((((tasks.get("pope") or {}).get("metrics") or {}).get("pope_f1_score") or {}).get("value"))
        sqa = pct((((tasks.get("scienceqa_img") or {}).get("metrics") or {}).get("exact_match") or {}).get("value"))
        mmmu = pct((((tasks.get("mmmu_val") or {}).get("metrics") or {}).get("mmmu_acc") or {}).get("value"))
        mme_metrics = ((tasks.get("mme") or {}).get("metrics") or {})
        mme_p = ((mme_metrics.get("mme_percetion_score") or {}).get("value"))
        mme_c = ((mme_metrics.get("mme_cognition_score") or {}).get("value"))
        mme_total = None
        if mme_p is not None or mme_c is not None:
            mme_total = float(mme_p or 0.0) + float(mme_c or 0.0)

        row["metrics"] = {
            "gqa": gqa,
            "mme_total": mme_total,
            "pope_f1": pope,
            "scienceqa": sqa,
            "mmmu": mmmu,
        }

        rets = [
            safe_ret(gqa, BASELINES["gqa"]),
            safe_ret(mme_total, BASELINES["mme_total"]),
            safe_ret(pope, BASELINES["pope"]),
            safe_ret(sqa, BASELINES["scienceqa_img"]),
            safe_ret(mmmu, BASELINES["mmmu_val"]),
        ]
        rets = [x for x in rets if x is not None]
        if len(rets) == 5:
            row["avg_retention"] = sum(rets) / len(rets)

    rows.append(row)

summary = {
    "queue_root": str(queue_root),
    "count": len(rows),
    "rows": rows,
}
(queue_root / "queue_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

with (queue_root / "queue_summary.tsv").open("w", encoding="utf-8") as f:
    f.write("strategy\tqueue_exit_code\tgqa\tmme_total\tpope_f1\tscienceqa\tmmmu\tavg_retention\n")
    for row in rows:
        m = row["metrics"]
        f.write(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                row["strategy"],
                row["queue_exit_code"],
                "" if m.get("gqa") is None else f"{m['gqa']:.4f}",
                "" if m.get("mme_total") is None else f"{m['mme_total']:.4f}",
                "" if m.get("pope_f1") is None else f"{m['pope_f1']:.4f}",
                "" if m.get("scienceqa") is None else f"{m['scienceqa']:.4f}",
                "" if m.get("mmmu") is None else f"{m['mmmu']:.4f}",
                "" if row["avg_retention"] is None else f"{row['avg_retention']:.4f}",
            )
        )

print(f"QUEUE_SUMMARY_JSON={queue_root / 'queue_summary.json'}")
print(f"QUEUE_SUMMARY_TSV={queue_root / 'queue_summary.tsv'}")
PY

echo "QUEUE_DONE=$QUEUE_ROOT"
