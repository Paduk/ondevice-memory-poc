#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
train_runner="${repo_root}/memory_training/scripts/run_granite4_350m_patch_train_first.sh"
validation_runner="${repo_root}/memory_training/scripts/run_checkpoint_6full_validation_pairwise.sh"
gpu=3
epochs=4
training_seed=47
run_tag=r1
run_id="granite4-350m-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed47-r1"

cd "${repo_root}"
echo "[$(date -Is)] Starting 4-Epoch train-first run ${run_id} on GPU ${gpu}."
bash "${train_runner}" "${gpu}" "${epochs}" "${training_seed}" "${run_tag}"
echo "[$(date -Is)] Training completed; starting pairwise checkpoint Validation."
exec bash "${validation_runner}" "${gpu}" "${run_id}" 1 4
