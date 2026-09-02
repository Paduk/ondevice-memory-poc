#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
train_runner="${repo_root}/memory_training/scripts/run_granite4_350m_method_train_first.sh"
validation_runner="${repo_root}/memory_training/scripts/run_checkpoint_6full_validation_pairwise.sh"
gpu=3

seed45_run=granite4-350m-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1
seed45_result="${workspace}/runs/${seed45_run}/validation-6full-pairwise-summary.json"
seed46_run=granite4-350m-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed46-r1

cd "${repo_root}"
export PYTHONUNBUFFERED=1

echo "[$(date -Is)] Waiting for Summary Seed 45 Validation on GPU ${gpu}."
while [[ ! -s "${seed45_result}" ]]; do
  if ! tmux has-session -t granite4-350m-summary-validation-gpu3 2>/dev/null \
      && ! pgrep -f "[v]alidate_hf_checkpoint_v2.*${seed45_run}" >/dev/null; then
    echo "[$(date -Is)] Summary Seed 45 ended without a final result; aborting." >&2
    exit 1
  fi
  sleep 30
done

echo "[$(date -Is)] Starting Summary Seed 46 training on GPU ${gpu}."
bash "${train_runner}" "${gpu}" summary 4 46 r1
echo "[$(date -Is)] Starting Summary Seed 46 Epoch 1-4 Validation."
bash "${validation_runner}" "${gpu}" "${seed46_run}" 1 4
echo "[$(date -Is)] Completed Summary Seed 46 training and Validation."
