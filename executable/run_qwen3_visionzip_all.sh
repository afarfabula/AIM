#!/bin/bash
# 用 Qwen3-VL token reduction 跑全部 benchmark 的便捷入口。
# - 默认 strategy=visionzip
# - visionzip 默认 dominant=64, contextual=16
# - 默认 limit=none（全量）
# - 所有 summary / per-bench stdout 日志写到 LOG_ROOT（默认 /root/aim_qwen3_logs）
# - 真正 lmms-eval 的 run.log 仍由 scripts/run_qwen3_vl_task.sh 写到 OUT_DIR/run.log
#
# 用法：
#   bash executable/run_qwen3_visionzip_all.sh
#   bash executable/run_qwen3_visionzip_all.sh --limit 10
#   bash executable/run_qwen3_visionzip_all.sh --dominant 64 --contextual 16
#   bash executable/run_qwen3_visionzip_all.sh --strategy bishemethod_v2stage_anchor16_aware_v6 --token-prune-config "bishe_target_keep:64"
#   bash executable/run_qwen3_visionzip_all.sh --benches "gqa,pope"
#   bash executable/run_qwen3_visionzip_all.sh --foreground
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LIMIT="${LIMIT:-none}"
DOMINANT="${DOMINANT:-64}"
CONTEXTUAL="${CONTEXTUAL:-16}"
STRATEGY="${STRATEGY:-visionzip}"
TOKEN_PRUNE_CONFIG="${TOKEN_PRUNE_CONFIG:-}"
RUN_TAG="${RUN_TAG:-}"
BENCHES_CSV="${BENCHES:-gqa,mme,pope,scienceqa,mmmu,mmbench,textvqa}"
LOG_ROOT="${LOG_ROOT:-/root/aim_qwen3_logs}"
FOREGROUND="${FOREGROUND:-0}"
# 内部使用：__INNER=1 表示已经在 nohup 出来的子进程中真正执行
__INNER="${__INNER:-0}"

usage() {
  cat <<'EOF'
Run all (or selected) Qwen3-VL benchmarks with token reduction.

Options:
  --limit <n|none>          lmms-eval limit (default: none = full run)
  --strategy <name>         token_prune_strategy (default: visionzip)
  --token-prune-config <s>  raw token_prune_config string; if omitted and
                            strategy=visionzip, use dominant/contextual
  --dominant <n>            VisionZip dominant tokens (default: 64)
  --contextual <n>          VisionZip contextual tokens (default: 16)
  --run-tag <name>          prefix used in summary/log filenames
  --benches <csv>           Comma-separated bench aliases (default:
                            gqa,mme,pope,scienceqa,mmmu,mmbench,textvqa)
  --log-root <dir>          Where to write summary + per-bench stdout logs
                            (default: /root/aim_qwen3_logs)
  --foreground              Run in foreground (default: nohup background)
  -h, --help                Show this help
EOF
}

slugify() {
  local s="$1"
  s="${s//\//_}"
  s="${s//,/+}"
  s="${s// /}"
  s="${s//:/-}"
  s="${s//;/_}"
  s="${s//|/_}"
  s="${s//~/_}"
  s="${s//=/-}"
  s="${s//(/}"
  s="${s//)/}"
  s="${s//[/}"
  s="${s//]/}"
  s="${s//\{/}"
  s="${s//\}/}"
  s="${s//__/_}"
  s="${s//__/_}"
  s="${s//+-/+}"
  printf '%s' "$s"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT="$2"; shift 2 ;;
    --strategy) STRATEGY="$2"; shift 2 ;;
    --token-prune-config) TOKEN_PRUNE_CONFIG="$2"; shift 2 ;;
    --dominant) DOMINANT="$2"; shift 2 ;;
    --contextual) CONTEXTUAL="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --benches) BENCHES_CSV="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --foreground) FOREGROUND=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$LOG_ROOT"

# HF token / 网络相关默认（与之前 smoke test 保持一致，可被外部覆盖）
export HF_TOKEN="${HF_TOKEN:-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}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export LOG_ROOT

IFS=',' read -r -a BENCHES <<< "$BENCHES_CSV"

if [ -z "$TOKEN_PRUNE_CONFIG" ] && [ "$STRATEGY" = "visionzip" ]; then
  TOKEN_PRUNE_CONFIG="dominant:${DOMINANT}~contextual:${CONTEXTUAL}"
fi

if [ -z "$RUN_TAG" ]; then
  RUN_TAG="$STRATEGY"
  if [ -n "$TOKEN_PRUNE_CONFIG" ]; then
    if [ "$STRATEGY" = "visionzip" ] && [[ "$TOKEN_PRUNE_CONFIG" =~ dominant:([0-9]+).*contextual:([0-9]+) ]]; then
      RUN_TAG="${RUN_TAG}_d${BASH_REMATCH[1]}_c${BASH_REMATCH[2]}"
    else
      RUN_TAG="${RUN_TAG}_$(slugify "$TOKEN_PRUNE_CONFIG")"
    fi
  elif [ "$STRATEGY" = "visionzip" ]; then
    RUN_TAG="${RUN_TAG}_d${DOMINANT}_c${CONTEXTUAL}"
  fi
fi

BENCH_TAG="$(slugify "$BENCHES_CSV")"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${SUMMARY:-$LOG_ROOT/${RUN_TAG}_${BENCH_TAG}_limit${LIMIT}_runs_${RUN_TS}_summary.log}"

run_all() {
  echo "[run_qwen3_visionzip_all] strategy=$STRATEGY token_prune_config=$TOKEN_PRUNE_CONFIG limit=$LIMIT benches=${BENCHES[*]}" \
    | tee -a "$SUMMARY"
  echo "[run_qwen3_visionzip_all] LOG_ROOT=$LOG_ROOT  SUMMARY=$SUMMARY" | tee -a "$SUMMARY"
  TS="$(date +%Y%m%d_%H%M%S)"
  LOG="$LOG_ROOT/${RUN_TAG}_${BENCH_TAG}_limit${LIMIT}_${TS}.log"
  echo "==== [$BENCHES_CSV] start $(date) -> $LOG" | tee -a "$SUMMARY"
  bash "$ROOT/executable/runa_qwen3.sh" \
    --bench "$BENCHES_CSV" \
    --limit "$LIMIT" \
    --token-prune-strategy "$STRATEGY" \
    --token-prune-config "$TOKEN_PRUNE_CONFIG" \
    > "$LOG" 2>&1
  RC=$?
  echo "==== [$BENCHES_CSV] done rc=$RC $(date)" | tee -a "$SUMMARY"
  tail -n 80 "$LOG" | tee -a "$SUMMARY"
  echo "----" | tee -a "$SUMMARY"
  echo "ALL_DONE $(date)" | tee -a "$SUMMARY"
}

if [ "$FOREGROUND" = "1" ] || [ "$__INNER" = "1" ]; then
  ln -sfn "$SUMMARY" "$LOG_ROOT/visionzip_last_summary.log"
  run_all
  exit 0
fi

# 后台模式：用 nohup 重新调起自己（FOREGROUND=1 走上面分支）
NOHUP_OUT="$LOG_ROOT/${RUN_TAG}_${BENCH_TAG}_limit${LIMIT}_runs_${RUN_TS}_nohup.out"
ln -sfn "$SUMMARY" "$LOG_ROOT/visionzip_last_summary.log"
ln -sfn "$NOHUP_OUT" "$LOG_ROOT/visionzip_last_nohup.out"
echo "[run_qwen3_visionzip_all] launching in background"
echo "  SUMMARY = $SUMMARY"
echo "  NOHUP   = $NOHUP_OUT"
SUMMARY="$SUMMARY" \
LIMIT="$LIMIT" \
DOMINANT="$DOMINANT" \
CONTEXTUAL="$CONTEXTUAL" \
STRATEGY="$STRATEGY" \
TOKEN_PRUNE_CONFIG="$TOKEN_PRUNE_CONFIG" \
RUN_TAG="$RUN_TAG" \
BENCHES="$BENCHES_CSV" \
LOG_ROOT="$LOG_ROOT" \
__INNER=1 \
nohup bash "$0" --foreground > "$NOHUP_OUT" 2>&1 &
echo "PID=$!"
