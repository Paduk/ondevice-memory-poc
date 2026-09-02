#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-6}"
poll_seconds="${2:-30}"

repo_root=/home/hj153lee/PalmClaw
train_runner="${repo_root}/memory_training/scripts/run_granite4_350m_method_train_first.sh"
validation_runner="${repo_root}/memory_training/scripts/run_checkpoint_6full_validation_pairwise.sh"
run_id=granite4-350m-delta-v3-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1

cd "${repo_root}"
export PYTHONUNBUFFERED=1

idle_checks=0
echo "[$(date -Is)] Waiting for GPU ${gpu} to have no compute processes."
while (( idle_checks < 2 )); do
  compute_pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true)"
  if [[ -z "${compute_pids}" ]]; then
    idle_checks=$((idle_checks + 1))
    echo "[$(date -Is)] GPU ${gpu} idle check ${idle_checks}/2."
  else
    idle_checks=0
    echo "[$(date -Is)] GPU ${gpu} is occupied by PID(s): $(echo "${compute_pids}" | paste -sd, -)"
  fi
  if (( idle_checks < 2 )); then
    sleep "${poll_seconds}"
  fi
done

echo "[$(date -Is)] Starting Granite4 350M Delta-V3 training on GPU ${gpu}."
bash "${train_runner}" "${gpu}" delta_v3 4 45 r1

echo "[$(date -Is)] Starting Delta-V3 Epoch 1-4 Validation on GPU ${gpu}."
bash "${validation_runner}" "${gpu}" "${run_id}" 1 4

echo "[$(date -Is)] Completed Granite4 350M Delta-V3 training and Validation."
