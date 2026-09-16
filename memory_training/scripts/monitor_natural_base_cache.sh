#!/usr/bin/env bash
set -euo pipefail

interval_seconds="${1:-900}"
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
benchmark_root="${workspace}/benchmarks"
granite_root="${benchmark_root}/granite4-350m-natural-s86-s90-cache-once-20260907-v1"
qwen_root="${benchmark_root}/qwen35-2b-natural-s86-s90-cache-once-20260907-v1"
aggregate_log="${benchmark_root}/stress-tradeoff-aggregate-20260907.log"
result_json=/home/hj153lee/PalmClaw/docs/engineering/results/stress-tradeoff-four-models.json
monitor_log="${benchmark_root}/natural-base-cache-monitor-20260907.log"

while true; do
  granite_done=0
  qwen_done=0
  if [[ -d "${granite_root}/cache" ]]; then
    granite_done=$(find "${granite_root}/cache" -name summary.json | wc -l)
  fi
  if [[ -d "${qwen_root}/cache" ]]; then
    qwen_done=$(find "${qwen_root}/cache" -name summary.json | wc -l)
  fi
  current=$(ps -eo cmd | sed -n 's/.*benchmark_hf_prefix_cache --model \([^ ]*\) --method \([^ ]*\).*/\1:\2/p' | paste -sd, -)
  runner=stopped
  if tmux has-session -t natural-base-cache-20260907 2>/dev/null \
    || tmux has-session -t qwen2b-natural-base-cache-20260907 2>/dev/null; then
    runner=running
  fi
  aggregate=waiting
  [[ -s "$aggregate_log" ]] && aggregate=finished-or-failed
  [[ -s "$result_json" ]] && aggregate=complete
  printf '[%s] runner=%s granite=%s/5 qwen=%s/5 current=%s aggregate=%s\n' \
    "$(date -Is)" "$runner" "$granite_done" "$qwen_done" "${current:-none}" "$aggregate" \
    | tee -a "$monitor_log"

  if [[ "$aggregate" == complete ]]; then
    exit 0
  fi
  if [[ "$runner" == stopped && "$aggregate" == finished-or-failed ]]; then
    exit 1
  fi
  sleep "$interval_seconds"
done
