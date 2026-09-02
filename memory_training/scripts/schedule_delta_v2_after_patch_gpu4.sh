#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/hj153lee/PalmClaw"
workspace="/mnt/data/hj153lee/PalmClaw/on-device-memory-training"
python_bin="/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python"
train_script="${repo_root}/memory_training/scripts/run_qwen35_4b_grouped_v1.sh"
patch_run_id="qwen35-4b-patch-multitask-noop5-grouped-v2-v1-10-r1"
delta_run_id="qwen35-4b-delta-v2-multitask-noop5-grouped-v2-v1-10-r1"
status_path="${workspace}/runs/${patch_run_id}/status.json"
log_path="${workspace}/delta-v2-after-patch-gpu4-scheduler.log"
gpu=4

exec > >(tee -a "${log_path}") 2>&1
echo "[$(date -Is)] Waiting for ${patch_run_id}."

while true; do
  state="MISSING"
  if [[ -f "${status_path}" ]]; then
    state="$(${python_bin} -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state", "UNKNOWN"))' "${status_path}")"
  fi
  case "${state}" in
    COMPLETED)
      while pgrep -f "memory_training.train.*--run-id ${patch_run_id}" >/dev/null; do
        sleep 10
      done
      if [[ -e "${workspace}/runs/${delta_run_id}" ]]; then
        echo "[$(date -Is)] Refusing to overwrite existing Delta-v2 run: ${delta_run_id}" >&2
        exit 1
      fi
      echo "[$(date -Is)] Patch completed. Starting ${delta_run_id} on GPU ${gpu}."
      exec "${train_script}" delta_v2 "${gpu}"
      ;;
    FAILED)
      echo "[$(date -Is)] Patch failed; Delta-v2 will not start." >&2
      exit 1
      ;;
  esac
  sleep 30
done
