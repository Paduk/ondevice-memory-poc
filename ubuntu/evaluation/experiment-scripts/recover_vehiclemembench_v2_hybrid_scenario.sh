#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 1 ]] || [[ "$1" -lt 21 ]] || [[ "$1" -gt 100 ]]; then
  echo "usage: $0 SCENARIO_INDEX (21..100)" >&2
  exit 2
fi

readonly SCENARIO_INDEX="$1"
readonly SCENARIO_TAG="$(printf '%02d' "$SCENARIO_INDEX")"
readonly RUNNER="/home/hj153lee/PalmClaw/ubuntu/evaluation/experiment-scripts/run_vehiclemembench_v2_hybrid_scenario_controller.sh"
if ((SCENARIO_INDEX <= 25)); then
  readonly RUN_RANGE="s21-s25"
elif ((SCENARIO_INDEX <= 50)); then
  readonly RUN_RANGE="s26-s50"
else
  readonly RUN_RANGE="s51-s100"
fi
readonly RUN_ROOT="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-hybrid/hybrid-scale-${RUN_RANGE}-state-evolution-r1"
readonly LOG_PATH="$RUN_ROOT/logs/scenario-${SCENARIO_TAG}.log"
readonly STATUS_PATH="$RUN_ROOT/status/scenario-${SCENARIO_TAG}.status"

printf 'RUNNING_RECOVERY\t%s\n' "$(date -u +%FT%TZ)" >"$STATUS_PATH"

for attempt in 1 2 3; do
  printf '[%s] S%s recovery attempt %d/3 started\n' \
    "$(date -u +%FT%TZ)" "$SCENARIO_TAG" "$attempt" >>"$LOG_PATH"
  if "$RUNNER" "$SCENARIO_INDEX" >>"$LOG_PATH" 2>&1; then
    printf 'COMPLETED\t%s\trecovery_attempt=%d\n' \
      "$(date -u +%FT%TZ)" "$attempt" >"$STATUS_PATH"
    exit 0
  fi
done

printf 'FAILED\t%s\trecovery_attempts=3\n' \
  "$(date -u +%FT%TZ)" >"$STATUS_PATH"
exit 1
