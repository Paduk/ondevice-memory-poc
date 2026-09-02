#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
patch_run_id=qwen35-2b-patch-multitask-noop5-grouped-v2-v1-10-e5-b4-r1
patch_run_root="${workspace}/runs/${patch_run_id}"
v2_progress="${patch_run_root}/test-hf-closed-loop-s86-s100-t12-t20-best-epoch-04/progress.json"
v1_progress="${patch_run_root}/test-hf-v1-s1-s50-best-epoch-04/progress.json"
runner="${repo_root}/memory_training/scripts/run_qwen35_2b_summary.sh"
gpu=3
log_path="${workspace}/qwen35-2b-summary-e4-after-patch-tests-gpu3-scheduler.log"

exec > >(tee -a "${log_path}") 2>&1
echo "[$(date -Is)] Waiting for 2B Patch V2 and V1 Tests to complete."

read_status() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    "${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", "UNKNOWN"))' "${path}"
  else
    echo MISSING
  fi
}

while true; do
  v2_state="$(read_status "${v2_progress}")"
  v1_state="$(read_status "${v1_progress}")"
  if [[ "${v2_state}" == COMPLETED && "${v1_state}" == COMPLETED ]]; then
    while pgrep -f '[m]emory_training.evaluate_hf_.*qwen35-2b-patch-multitask' >/dev/null; do
      sleep 10
    done
    echo "[$(date -Is)] Patch Tests completed; starting 2B Summary E4 on GPU ${gpu}."
    exec bash "${runner}" "${gpu}" 4
  fi
  if [[ "${v2_state}" == FAILED || "${v1_state}" == FAILED ]]; then
    echo "[$(date -Is)] Patch Test failed; Summary training will not start." >&2
    exit 1
  fi
  sleep 30
done
