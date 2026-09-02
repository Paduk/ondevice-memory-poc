#!/usr/bin/env bash
set -uo pipefail

readonly REPO_ROOT="/home/hj153lee/PalmClaw"
readonly SCENARIO_RUNNER="$REPO_ROOT/ubuntu/evaluation/experiment-scripts/run_vehiclemembench_v2_hybrid_scenario_controller.sh"
readonly RUN_ROOT="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-hybrid/hybrid-scale-s21-s25-state-evolution-r1"
readonly LOG_ROOT="$RUN_ROOT/logs"
readonly STATUS_ROOT="$RUN_ROOT/status"
readonly MAX_PARALLEL=5
readonly MAX_SCENARIO_ATTEMPTS=2

mkdir -p "$LOG_ROOT" "$STATUS_ROOT"

run_scenario() {
  local scenario_index="$1"
  local scenario_tag
  local attempt
  local log_path
  local status_path

  scenario_tag="$(printf '%02d' "$scenario_index")"
  log_path="$LOG_ROOT/scenario-${scenario_tag}.log"
  status_path="$STATUS_ROOT/scenario-${scenario_tag}.status"

  for ((attempt = 1; attempt <= MAX_SCENARIO_ATTEMPTS; attempt += 1)); do
    printf '[%s] S%s attempt %d/%d started\n' \
      "$(date -u +%FT%TZ)" "$scenario_tag" "$attempt" \
      "$MAX_SCENARIO_ATTEMPTS" >>"$log_path"
    if "$SCENARIO_RUNNER" "$scenario_index" >>"$log_path" 2>&1; then
      printf 'COMPLETED\t%s\tattempt=%d\n' \
        "$(date -u +%FT%TZ)" "$attempt" >"$status_path"
      printf '[%s] S%s completed\n' "$(date -u +%FT%TZ)" "$scenario_tag"
      return 0
    fi
    printf '[%s] S%s attempt %d failed; resuming from checkpoints\n' \
      "$(date -u +%FT%TZ)" "$scenario_tag" "$attempt" >>"$log_path"
  done

  printf 'FAILED\t%s\tattempts=%d\n' \
    "$(date -u +%FT%TZ)" "$MAX_SCENARIO_ATTEMPTS" >"$status_path"
  printf '[%s] S%s failed after %d attempts\n' \
    "$(date -u +%FT%TZ)" "$scenario_tag" "$MAX_SCENARIO_ATTEMPTS"
  return 1
}

failed=0
active=0
if (($#)); then
  scenario_indexes=("$@")
else
  scenario_indexes=($(seq 21 25))
fi
for scenario_index in "${scenario_indexes[@]}"; do
  if ((scenario_index < 21 || scenario_index > 25)); then
    printf 'scenario index must be in [21, 25]: %s\n' "$scenario_index" >&2
    exit 2
  fi
done
printf '[%s] S21-S25 state-evolution controller started (parallel=%d scenarios=%s)\n' \
  "$(date -u +%FT%TZ)" "$MAX_PARALLEL" "${scenario_indexes[*]}"

for scenario_index in "${scenario_indexes[@]}"; do
  run_scenario "$scenario_index" &
  active=$((active + 1))
done

while ((active > 0)); do
  if ! wait -n; then
    failed=1
  fi
  active=$((active - 1))
done

completed_count="$(find "$STATUS_ROOT" -type f -name 'scenario-*.status' -exec grep -l '^COMPLETED' {} + 2>/dev/null | wc -l)"
failed_count="$(find "$STATUS_ROOT" -type f -name 'scenario-*.status' -exec grep -l '^FAILED' {} + 2>/dev/null | wc -l)"
printf '[%s] Controller finished: completed=%s failed=%s\n' \
  "$(date -u +%FT%TZ)" "$completed_count" "$failed_count"

exit "$failed"
