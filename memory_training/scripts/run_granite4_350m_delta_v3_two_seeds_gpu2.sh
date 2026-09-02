#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
train_runner="${repo_root}/memory_training/scripts/run_granite4_350m_method_train_first.sh"
validation_runner="${repo_root}/memory_training/scripts/run_checkpoint_6full_validation_pairwise.sh"
gpu=2

cd "${repo_root}"
export PYTHONUNBUFFERED=1

for training_seed in 45 46; do
  run_id="granite4-350m-delta-v3-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed${training_seed}-r1"
  echo "[$(date -Is)] Starting Delta-V3 Seed ${training_seed} training on GPU ${gpu}."
  bash "${train_runner}" "${gpu}" delta_v3 4 "${training_seed}" r1
  echo "[$(date -Is)] Starting Delta-V3 Seed ${training_seed} Epoch 1-4 Validation."
  bash "${validation_runner}" "${gpu}" "${run_id}" 1 4
done

echo "[$(date -Is)] Completed both Granite4 350M Delta-V3 runs."
