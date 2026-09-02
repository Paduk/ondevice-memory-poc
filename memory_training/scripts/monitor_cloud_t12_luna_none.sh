#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-cloud-ondevice-prompt-t12-luna-none-smoke-20260831
LOG="$ROOT/monitor-5min.log"
INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-300}"
DURATION_SECONDS="${MONITOR_DURATION_SECONDS:-0}"
STARTED_AT="$(date +%s)"
mkdir -p "$ROOT"

while true; do
  completed=0
  methods=(
    summary_compact600_filtered
    patch_filtered
    delta_v3_filtered
    patch_filtered_semantic5
    delta_v3_filtered_semantic5
  )
  {
    date --iso-8601=seconds
    for method in "${methods[@]}"; do
      progress="$ROOT/$method/progress.json"
      if [[ -f "$progress" ]]; then
        jq -c --arg method "$method" \
          '{method:$method,status,phase,item,items,progress,updated_at}' "$progress"
      else
        printf '{"method":"%s","status":"INITIALIZING"}\n' "$method"
      fi
      if [[ -f "$ROOT/$method/summary.json" ]]; then
        completed=$((completed + 1))
      fi
    done
  } >>"$LOG"
  if [[ "$completed" -eq "${#methods[@]}" ]]; then
    break
  fi
  elapsed=$(($(date +%s) - STARTED_AT))
  if [[ "$DURATION_SECONDS" -gt 0 && "$elapsed" -ge "$DURATION_SECONDS" ]]; then
    printf '%s monitoring stopped after %ss\n' "$(date --iso-8601=seconds)" \
      "$DURATION_SECONDS" >>"$LOG"
    break
  fi
  sleep "$INTERVAL_SECONDS"
done
