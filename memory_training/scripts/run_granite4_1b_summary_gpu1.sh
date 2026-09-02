#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
train_runner="${repo_root}/memory_training/scripts/run_granite4_1b_method_train_first.sh"
validation_runner="${repo_root}/memory_training/scripts/run_checkpoint_6full_validation_pairwise.sh"
gpu=1
run_id=granite4-1b-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1

cd "${repo_root}"
export PYTHONUNBUFFERED=1

echo "[$(date -Is)] Starting Granite4 1B Summary training on GPU ${gpu}."
bash "${train_runner}" "${gpu}" summary 4 45 r1
echo "[$(date -Is)] Starting Summary Epoch 1-4 Validation."
bash "${validation_runner}" "${gpu}" "${run_id}" 1 4
echo "[$(date -Is)] Completed Granite4 1B Summary training and Validation."
