#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
train_runner="${repo_root}/memory_training/scripts/run_granite4_1b_method_train_first.sh"
validation_runner="${repo_root}/memory_training/scripts/run_checkpoint_6full_validation_pairwise.sh"
gpu=7
epochs=4
training_seed=45

patch_run=granite4-1b-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed46-r1
patch_result="${workspace}/runs/${patch_run}/validation-6full-pairwise-summary.json"

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

for method in summary delta_v3; do
  method_slug="${method//_/-}"
  run_id="granite4-1b-${method_slug}-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed${training_seed}-r1"
  echo "[$(date -Is)] Starting ${method} training on GPU ${gpu}."
  bash "${train_runner}" "${gpu}" "${method}" "${epochs}" "${training_seed}" r1
  echo "[$(date -Is)] Starting ${method} Epoch 1-4 Validation."
  bash "${validation_runner}" "${gpu}" "${run_id}" 1 4
done

echo "[$(date -Is)] Completed Granite4 1B Summary and Delta-V3 runs."
