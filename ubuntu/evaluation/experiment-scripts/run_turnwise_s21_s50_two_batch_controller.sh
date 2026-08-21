#!/usr/bin/env bash
set -uo pipefail

readonly ARTIFACT_ROOT="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench"
readonly CONTROL_ROOT="${ARTIFACT_ROOT}/turnwise-summary-combined-s21-s50-controller-20260815"
readonly SUMMARY_SCRIPT="/home/hj153lee/PalmClaw/ubuntu/evaluation/experiment-scripts/run_turnwise_summary_fresh_single.sh"
readonly COMBINED_SCRIPT="/home/hj153lee/PalmClaw/ubuntu/evaluation/experiment-scripts/run_turnwise_combined_fresh_single.sh"

mkdir -p "${CONTROL_ROOT}"
exec > >(tee -a "${CONTROL_ROOT}/controller.log") 2>&1
echo "$$" > "${CONTROL_ROOT}/controller.pid"

experiment_root() {
  local method="$1"
  local scenario="$2"
  printf '%s/turnwise-%s-fresh-r1-s%s-20260815' \
    "${ARTIFACT_ROOT}" "${method}" "${scenario}"
}

terminal_count() {
  local start="$1"
  local end="$2"
  local count=0
  local scenario method root
  for scenario in $(seq "${start}" "${end}"); do
    for method in summary combined; do
      root="$(experiment_root "${method}" "${scenario}")"
      if [[ -f "${root}/COMPLETED" || -f "${root}/FAILED" ]]; then
        count=$((count + 1))
      fi
    done
  done
  printf '%s' "${count}"
}

failed_count() {
  local start="$1"
  local end="$2"
  local count=0
  local scenario method root
  for scenario in $(seq "${start}" "${end}"); do
    for method in summary combined; do
      root="$(experiment_root "${method}" "${scenario}")"
      [[ -f "${root}/FAILED" ]] && count=$((count + 1))
    done
  done
  printf '%s' "${count}"
}

wait_for_batch() {
  local start="$1"
  local end="$2"
  local label="$3"
  local done_count
  while true; do
    done_count="$(terminal_count "${start}" "${end}")"
    echo "[$(date --iso-8601=seconds)] ${label}: ${done_count}/30 terminal"
    [[ "${done_count}" -eq 30 ]] && break
    sleep 60
  done
}

cleanup_sessions() {
  local start="$1"
  local end="$2"
  local scenario method session
  for scenario in $(seq "${start}" "${end}"); do
    for method in summary combined; do
      session="tw-${method}-r1-s${scenario}"
      if tmux has-session -t "${session}" 2>/dev/null; then
        tmux kill-session -t "${session}"
      fi
    done
  done
}

start_batch_two() {
  local scenario method root
  for scenario in $(seq 36 50); do
    for method in summary combined; do
      root="$(experiment_root "${method}" "${scenario}")"
      if [[ -e "${root}" ]]; then
        echo "Batch 2 target already exists: ${root}" >&2
        touch "${CONTROL_ROOT}/CONTROLLER_FAILED"
        return 1
      fi
    done
  done

  for scenario in $(seq 36 50); do
    tmux new-session -d -s "tw-summary-r1-s${scenario}" \
      "/bin/bash -lc '${SUMMARY_SCRIPT} ${scenario} 1; run_status=\$?; echo RUN_STATUS=\$run_status; exec bash'"
    tmux new-session -d -s "tw-combined-r1-s${scenario}" \
      "/bin/bash -lc '${COMBINED_SCRIPT} ${scenario} 1; run_status=\$?; echo RUN_STATUS=\$run_status; exec bash'"
  done
  touch "${CONTROL_ROOT}/BATCH2_STARTED"
  echo "[$(date --iso-8601=seconds)] Batch 2 S36-S50 started"
}

echo "[$(date --iso-8601=seconds)] Controller started"
wait_for_batch 21 35 "Batch 1 S21-S35"
echo "[$(date --iso-8601=seconds)] Batch 1 terminal; failures=$(failed_count 21 35)"
sleep 15
cleanup_sessions 21 35

if ! start_batch_two; then
  exit 1
fi

wait_for_batch 36 50 "Batch 2 S36-S50"
echo "[$(date --iso-8601=seconds)] Batch 2 terminal; failures=$(failed_count 36 50)"
sleep 15
cleanup_sessions 36 50
touch "${CONTROL_ROOT}/ALL_TERMINAL"
echo "[$(date --iso-8601=seconds)] All S21-S50 runs reached terminal state"
