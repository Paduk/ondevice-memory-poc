#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/hj153lee/PalmClaw"
workspace="/mnt/data/hj153lee/PalmClaw/on-device-memory-training"
python_bin="/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python"
test_script="${repo_root}/memory_training/scripts/run_qwen35_4b_grouped_v1_test.sh"
gpu=3
log_path="${workspace}/grouped-summary-patch-test-gpu3-scheduler.log"

exec > >(tee -a "${log_path}") 2>&1

wait_for_training() {
  local method="$1"
  local run_id="qwen35-4b-${method}-multitask-noop5-grouped-v2-v1-10-r1"
  local status_path="${workspace}/runs/${run_id}/status.json"
  local checkpoint="${workspace}/runs/${run_id}/checkpoints/epoch-04/adapter"

  echo "[$(date -Is)] Waiting for ${run_id}."
  while true; do
    local state="MISSING"
    if [[ -f "${status_path}" ]]; then
      state="$(${python_bin} -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state", "UNKNOWN"))' "${status_path}")"
    fi
    case "${state}" in
      COMPLETED)
        while pgrep -f "memory_training.train.*--run-id ${run_id}" >/dev/null; do
          sleep 10
        done
        if [[ ! -d "${checkpoint}" ]]; then
          echo "[$(date -Is)] Missing final checkpoint: ${checkpoint}" >&2
          exit 1
        fi
        echo "[$(date -Is)] ${run_id} is ready."
        return
        ;;
      FAILED)
        echo "[$(date -Is)] ${run_id} failed; scheduler stopped." >&2
        exit 1
        ;;
    esac
    sleep 30
  done
}

for method in summary patch; do
  wait_for_training "${method}"
  echo "[$(date -Is)] Starting ${method} Test on GPU ${gpu}."
  "${test_script}" "${method}" "${gpu}" 04
  echo "[$(date -Is)] Completed ${method} Test."
done

echo "[$(date -Is)] All grouped Summary/Patch Tests completed."
