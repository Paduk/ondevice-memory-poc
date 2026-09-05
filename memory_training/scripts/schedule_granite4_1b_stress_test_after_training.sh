#!/usr/bin/env bash
set -euo pipefail

run_tag="${1:-20260904-v1}"
shift || true
training_pids=("$@")

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
granite_output="${workspace}/benchmarks/granite4-1b-stress-test-once-${run_tag}"
scheduler_log="${granite_output}/scheduler.log"
runner="${repo_root}/memory_training/scripts/run_granite4_1b_stress_test_once.sh"

run_dirs=(
  "${workspace}/runs/granite4-1b-delta_v3_compact_k2-multitask-noop5-uniform-depth-e4-b2-trainseed45-r1"
  "${workspace}/runs/granite4-1b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b2-trainseed45-r1"
  "${workspace}/runs/granite4-1b-delta_v3_compact_k10-multitask-noop5-uniform-depth-e4-b2-trainseed45-r1"
)

mkdir -p "$granite_output"
exec >>"$scheduler_log" 2>&1
echo "[$(date -Is)] scheduler started; training_pids=${training_pids[*]:-none}"

training_processes_alive() {
  local pid
  for pid in "${training_pids[@]}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

granite_ready() {
  local run_dir status selection checkpoint
  for run_dir in "${run_dirs[@]}"; do
    status="${run_dir}/status.json"
    selection="${run_dir}/eval-fixed-best-checkpoint.json"
    [[ -s "$status" ]] || return 1
    [[ "$(jq -r '.state // .final_state' "$status")" == COMPLETED ]] || return 1
    [[ -s "$selection" ]] || return 1
    checkpoint=$(jq -er '.winner.checkpoint' "$selection") || return 1
    [[ -d "${checkpoint}/adapter" ]] || return 1
  done
}

last_note=""
while true; do
  granite_state=waiting
  granite_ready && granite_state=ready

  note="granite=${granite_state}"
  if [[ "$note" != "$last_note" ]]; then
    echo "[$(date -Is)] ${note}"
    last_note="$note"
  fi

  if [[ "$granite_state" == ready ]]; then
    break
  fi

  if ! training_processes_alive && [[ "$granite_state" != ready ]]; then
    echo "[$(date -Is)] ERROR: Granite pipelines exited without all COMPLETED selections."
    exit 3
  fi
  sleep 60
done

echo "[$(date -Is)] prerequisites ready; launching Granite stress Test."
"$runner" "$run_tag"
echo "[$(date -Is)] scheduler completed."
