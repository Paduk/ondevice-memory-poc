#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/hj153lee/PalmClaw"
runner="$repo_root/ubuntu/evaluation/experiment-scripts/run_vehiclemembench_v2_temporal_scenario.sh"
control_root="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-temporal/controller-r1"
max_parallel="${TEMPORAL_MAX_PARALLEL:-5}"
max_scenario_attempts="${TEMPORAL_MAX_SCENARIO_ATTEMPTS:-3}"
start_index="${TEMPORAL_START_INDEX:-1}"
end_index="${TEMPORAL_END_INDEX:-20}"

if (( max_parallel < 1 || max_scenario_attempts < 1 || start_index < 1 || end_index > 20 || start_index > end_index )); then
  echo "invalid controller range or parallelism" >&2
  exit 2
fi

mkdir -p "$control_root/logs" "$control_root/status"

run_scenario() {
  local index="$1"
  local tag
  tag="$(printf '%02d' "$index")"
  local log="$control_root/logs/temporal-t${tag}.log"
  local status="$control_root/status/temporal-t${tag}.status"
  printf 'RUNNING\t%s\n' "$(date -u +%FT%TZ)" > "$status"
  : > "$log"
  local attempt code
  for attempt in $(seq 1 "$max_scenario_attempts"); do
    printf '[%s] T%s controller attempt %s/%s\n' \
      "$(date -u +%FT%TZ)" "$tag" "$attempt" "$max_scenario_attempts" \
      >> "$log"
    if "$runner" "$index" >> "$log" 2>&1; then
      printf 'COMPLETED\t%s\tattempt=%s\n' \
        "$(date -u +%FT%TZ)" "$attempt" > "$status"
      return 0
    else
      code="$?"
    fi
    printf '[%s] T%s attempt %s failed exit=%s; resuming\n' \
      "$(date -u +%FT%TZ)" "$tag" "$attempt" "$code" >> "$log"
  done
  printf 'FAILED\t%s\texit=%s\tattempts=%s\n' \
    "$(date -u +%FT%TZ)" "$code" "$max_scenario_attempts" > "$status"
  return "$code"
}

if (( $# > 0 )); then
  scenario_indexes=("$@")
else
  mapfile -t scenario_indexes < <(seq "$start_index" "$end_index")
fi
for index in "${scenario_indexes[@]}"; do
  if (( index < 1 || index > 20 )); then
    echo "scenario index must be in [1, 20]: $index" >&2
    exit 2
  fi
done

active=0
failed=0
for index in "${scenario_indexes[@]}"; do
  run_scenario "$index" &
  active=$((active + 1))
  if (( active >= max_parallel )); then
    if ! wait -n; then
      failed=1
    fi
    active=$((active - 1))
  fi
done

while (( active > 0 )); do
  if ! wait -n; then
    failed=1
  fi
  active=$((active - 1))
done

if (( failed != 0 )); then
  echo "One or more Temporal scenarios failed; inspect $control_root/status" >&2
  exit 1
fi

echo "All Temporal scenarios completed."
