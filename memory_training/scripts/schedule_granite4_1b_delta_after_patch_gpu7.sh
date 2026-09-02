#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
train_runner="${repo_root}/memory_training/scripts/run_granite4_1b_method_train_first.sh"
validation_runner="${repo_root}/memory_training/scripts/run_checkpoint_6full_validation_pairwise.sh"
gpu=7

patch_run=granite4-1b-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed46-r1
patch_result="${workspace}/runs/${patch_run}/validation-6full-pairwise-summary.json"
delta_run=granite4-1b-delta-v3-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1

cd "${repo_root}"
export PYTHONUNBUFFERED=1

echo "[$(date -Is)] Waiting for Patch Seed 46 Validation on GPU ${gpu}."
while [[ ! -s "${patch_result}" ]]; do
  if ! tmux has-session -t granite4-1b-patch-two-seeds-gpu7 2>/dev/null \
      && ! pgrep -f "[m]emory_training.*${patch_run}" >/dev/null; then
    echo "[$(date -Is)] Patch chain ended without the Seed 46 result; aborting." >&2
    exit 1
  fi
  sleep 30
done

echo "[$(date -Is)] Starting Delta-V3 training on GPU ${gpu}."
bash "${train_runner}" "${gpu}" delta_v3 4 45 r1
echo "[$(date -Is)] Starting Delta-V3 Epoch 1-4 Validation."
bash "${validation_runner}" "${gpu}" "${delta_run}" 1 4
echo "[$(date -Is)] Completed Granite4 1B Delta-V3 training and Validation."
