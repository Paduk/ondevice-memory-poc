#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
train_runner="${repo_root}/memory_training/scripts/run_granite4_350m_method_train_first.sh"
validation_runner="${repo_root}/memory_training/scripts/run_checkpoint_6full_validation_pairwise.sh"
gpu=3
epochs=4
training_seed=45
patch_run=granite4-350m-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed47-r1
patch_summary="${workspace}/runs/${patch_run}/validation-6full-pairwise-summary.json"

cd "${repo_root}"
export PYTHONUNBUFFERED=1

echo "[$(date -Is)] Waiting for Seed 47 Patch Validation on GPU ${gpu}."
while [[ ! -s "${patch_summary}" ]]; do
  if ! tmux list-sessions -F '#{session_name}' 2>/dev/null \
          | grep -q '^granite4-350m-seed47' \
      && ! pgrep -f '[v]alidate_hf_checkpoint_v2.*granite4-350m-patch.*trainseed47-r1' >/dev/null; then
    echo "[$(date -Is)] Seed 47 Validation stopped without a summary; aborting follow-up jobs." >&2
    exit 1
  fi
  sleep 30
done

for method in summary delta_v3; do
  method_slug="${method//_/-}"
  run_id="granite4-350m-${method_slug}-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1"
  echo "[$(date -Is)] Starting ${method} training on GPU ${gpu}."
  bash "${train_runner}" "${gpu}" "${method}" "${epochs}" "${training_seed}" r1
  echo "[$(date -Is)] Starting ${method} Epoch 1-4 Validation."
  bash "${validation_runner}" "${gpu}" "${run_id}" 1 4
done

echo "[$(date -Is)] Completed Granite 350M Summary and Delta-V3 runs."
