#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
train_runner="${repo_root}/memory_training/scripts/run_granite4_350m_method_train_first.sh"
validation_runner="${repo_root}/memory_training/scripts/run_checkpoint_6full_validation_pairwise.sh"
gpu=3
epochs=4

summary_run=granite4-350m-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1
summary_result="${workspace}/runs/${summary_run}/validation-6full-pairwise-summary.json"

cd "${repo_root}"
export PYTHONUNBUFFERED=1

echo "[$(date -Is)] Waiting for the GPU ${gpu} Summary Validation to finish."
while [[ ! -s "${summary_result}" ]]; do
  if ! tmux has-session -t granite4-350m-summary-validation-gpu3 2>/dev/null \
      && ! pgrep -f "[v]alidate_hf_checkpoint_v2.*${summary_run}" >/dev/null; then
    echo "[$(date -Is)] Summary Validation ended without a final result; aborting." >&2
    exit 1
  fi
  sleep 30
done

for training_seed in 45 46; do
  run_id="granite4-350m-delta-v3-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed${training_seed}-r1"
  echo "[$(date -Is)] Starting Delta-V3 Seed ${training_seed} training on GPU ${gpu}."
  bash "${train_runner}" "${gpu}" delta_v3 "${epochs}" "${training_seed}" r1
  echo "[$(date -Is)] Starting Delta-V3 Seed ${training_seed} Epoch 1-4 Validation."
  bash "${validation_runner}" "${gpu}" "${run_id}" 1 4
done

summary_seed=46
summary_run_id="granite4-350m-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed${summary_seed}-r1"
echo "[$(date -Is)] Starting the additional Summary Seed ${summary_seed} training on GPU ${gpu}."
bash "${train_runner}" "${gpu}" summary "${epochs}" "${summary_seed}" r1
echo "[$(date -Is)] Starting Summary Seed ${summary_seed} Epoch 1-4 Validation."
bash "${validation_runner}" "${gpu}" "${summary_run_id}" 1 4

echo "[$(date -Is)] Completed both Delta-V3 runs and the additional Summary run."
