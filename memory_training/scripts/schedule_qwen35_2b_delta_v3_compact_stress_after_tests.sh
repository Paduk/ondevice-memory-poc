#!/usr/bin/env bash
set -euo pipefail

run_tag="${1:-20260905-v1}"
k2_pipeline_pid="${2:-}"
k5_pipeline_pid="${3:-}"
k10_pipeline_pid="${4:-}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
output_root="${workspace}/benchmarks/qwen35-2b-stress-test-once-${run_tag}"
runner="${repo_root}/memory_training/scripts/run_qwen35_2b_delta_v3_compact_k_stress_test_once.sh"
scheduler_log="${output_root}/scheduler.log"

mkdir -p "${output_root}/logs"
exec >>"${scheduler_log}" 2>&1
echo "[$(date -Is)] parallel scheduler started: GPU0=k2 GPU1=k5 GPU7=k10"

pipeline_ready() {
  local interval=$1
  local method="delta_v3_compact_k${interval}"
  local run_id="qwen35-2b-${method}-multitask-noop5-uniform-depth-e4-b2-trainseed45-evalfixed-noop5-r1"
  local run_dir="${workspace}/runs/${run_id}"
  local status_path="${run_dir}/status.json"
  local selection_path="${run_dir}/eval-fixed-best-checkpoint.json"
  local checkpoint test_summary

  [[ -s "${status_path}" ]] || return 1
  [[ "$(jq -r '.state // .final_state' "${status_path}")" == COMPLETED ]] || return 1
  [[ -s "${selection_path}" ]] || return 1
  checkpoint=$(jq -er '.winner.checkpoint' "${selection_path}") || return 1
  [[ -d "${checkpoint}/adapter" ]] || return 1
  test_summary=$(jq -er '.external_test' "${status_path}") || return 1
  [[ -s "${test_summary}" ]] || return 1
  jq -e \
    '.complete == true and (.completed_scenarios | length) == 24 and (.expected_scenarios | length) == 24' \
    "${test_summary}" >/dev/null
}

wait_and_run() {
  local gpu=$1
  local interval=$2
  local pipeline_pid=$3
  local method_log="${output_root}/logs/k${interval}.log"

  echo "[$(date -Is)] GPU=${gpu} k=${interval} waiting for training, fixed Validation, and 24/24 fixed Test"
  while ! pipeline_ready "${interval}"; do
    if [[ -n "${pipeline_pid}" ]] && ! kill -0 "${pipeline_pid}" 2>/dev/null; then
      echo "[$(date -Is)] ERROR: GPU=${gpu} k=${interval} pipeline PID ${pipeline_pid} exited before prerequisites completed"
      return 3
    fi
    sleep 60
  done

  echo "[$(date -Is)] GPU=${gpu} k=${interval} prerequisites ready; launching stress Test"
  "${runner}" "${gpu}" "${interval}" "${run_tag}" >"${method_log}" 2>&1
  echo "[$(date -Is)] GPU=${gpu} k=${interval} stress Test completed"
}

wait_and_run 0 2 "${k2_pipeline_pid}" &
k2_scheduler_pid=$!
wait_and_run 1 5 "${k5_pipeline_pid}" &
k5_scheduler_pid=$!
wait_and_run 7 10 "${k10_pipeline_pid}" &
k10_scheduler_pid=$!

status=0
for scheduler_pid in "${k2_scheduler_pid}" "${k5_scheduler_pid}" "${k10_scheduler_pid}"; do
  if ! wait "${scheduler_pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "[$(date -Is)] ERROR: one or more compact stress workers failed"
  exit "${status}"
fi

echo "[$(date -Is)] parallel scheduler completed all compact methods"
