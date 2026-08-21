#!/usr/bin/env bash
set -uo pipefail

readonly ARTIFACT_ROOT="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench"
readonly CONTROL_ROOT="${ARTIFACT_ROOT}/turnwise-patch-soft30-s1-s50-2x-controller-20260816"
readonly RUN_SCRIPT="/home/hj153lee/PalmClaw/ubuntu/evaluation/experiment-scripts/run_turnwise_patch_soft30_fresh_single.sh"

mkdir -p "${CONTROL_ROOT}"
exec > >(tee -a "${CONTROL_ROOT}/controller.log") 2>&1
echo "$$" > "${CONTROL_ROOT}/controller.pid"

experiment_root() {
  local repetition="$1" scenario="$2"
  printf '%s/turnwise-patch-soft30-fresh-r%s-s%s-20260816' \
    "${ARTIFACT_ROOT}" "${repetition}" "${scenario}"
}

terminal_count() {
  local repetition="$1" start="$2" end="$3" count=0 scenario root
  for scenario in $(seq "${start}" "${end}"); do
    root="$(experiment_root "${repetition}" "${scenario}")"
    if [[ -f "${root}/COMPLETED" || -f "${root}/FAILED" ]]; then
      count=$((count + 1))
    fi
  done
  printf '%s' "${count}"
}

failed_count() {
  local repetition="$1" start="$2" end="$3" count=0 scenario root
  for scenario in $(seq "${start}" "${end}"); do
    root="$(experiment_root "${repetition}" "${scenario}")"
    [[ -f "${root}/FAILED" ]] && count=$((count + 1))
  done
  printf '%s' "${count}"
}

start_batch() {
  local repetition="$1" start="$2" end="$3" scenario root session
  for scenario in $(seq "${start}" "${end}"); do
    root="$(experiment_root "${repetition}" "${scenario}")"
    if [[ -e "${root}" ]]; then
      echo "Batch target already exists: ${root}" >&2
      touch "${CONTROL_ROOT}/CONTROLLER_FAILED"
      return 1
    fi
  done
  for scenario in $(seq "${start}" "${end}"); do
    session="tw-soft30-r${repetition}-s${scenario}"
    tmux new-session -d -s "${session}" \
      "/bin/bash -lc '${RUN_SCRIPT} ${scenario} ${repetition}; run_status=\$?; echo RUN_STATUS=\$run_status; exec bash'"
  done
  echo "[$(date --iso-8601=seconds)] R${repetition} S${start}-S${end} started"
}

wait_for_batch() {
  local repetition="$1" start="$2" end="$3" done_count
  while true; do
    done_count="$(terminal_count "${repetition}" "${start}" "${end}")"
    echo "[$(date --iso-8601=seconds)] R${repetition} S${start}-S${end}: ${done_count}/25 terminal"
    [[ "${done_count}" -eq 25 ]] && break
    sleep 60
  done
}

cleanup_sessions() {
  local repetition="$1" start="$2" end="$3" scenario session
  for scenario in $(seq "${start}" "${end}"); do
    session="tw-soft30-r${repetition}-s${scenario}"
    tmux has-session -t "${session}" 2>/dev/null && tmux kill-session -t "${session}"
  done
}

run_batch() {
  local repetition="$1" start="$2" end="$3"
  start_batch "${repetition}" "${start}" "${end}" || return 1
  wait_for_batch "${repetition}" "${start}" "${end}"
  echo "[$(date --iso-8601=seconds)] R${repetition} S${start}-S${end} terminal; failures=$(failed_count "${repetition}" "${start}" "${end}")"
  sleep 15
  cleanup_sessions "${repetition}" "${start}" "${end}"
}

echo "[$(date --iso-8601=seconds)] Pure soft-30 Patch 2x controller started"
run_batch 1 1 25 || exit 1
touch "${CONTROL_ROOT}/BATCH1_TERMINAL"
run_batch 1 26 50 || exit 1
touch "${CONTROL_ROOT}/BATCH2_TERMINAL"
run_batch 2 1 25 || exit 1
touch "${CONTROL_ROOT}/BATCH3_TERMINAL"
run_batch 2 26 50 || exit 1
touch "${CONTROL_ROOT}/BATCH4_TERMINAL"
touch "${CONTROL_ROOT}/ALL_TERMINAL"
echo "[$(date --iso-8601=seconds)] All pure soft-30 Patch S1-S50 2x runs reached terminal state"
