#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"
method="${2:?Method is required: summary or delta_v3}"
prerequisite_tmux="${3:?Prerequisite tmux session is required}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
train_runner="${repo_root}/memory_training/scripts/run_granite4_1b_method_train_first.sh"
validation_runner="${repo_root}/memory_training/scripts/run_checkpoint_6full_validation_pairwise.sh"

case "${method}" in
  summary) method_slug=summary ;;
  delta_v3) method_slug=delta-v3 ;;
  *) echo "Unsupported method: ${method}" >&2; exit 2 ;;
esac

seed45_run="granite4-1b-${method_slug}-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1"
seed45_result="${workspace}/runs/${seed45_run}/validation-6full-pairwise-summary.json"
seed46_run="granite4-1b-${method_slug}-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed46-r1"

cd "${repo_root}"
export PYTHONUNBUFFERED=1

echo "[$(date -Is)] Waiting for ${method} Seed 45 Validation on GPU ${gpu}."
while [[ ! -s "${seed45_result}" ]]; do
  if ! tmux has-session -t "${prerequisite_tmux}" 2>/dev/null \
      && ! pgrep -f "[m]emory_training.*${seed45_run}" >/dev/null; then
    echo "[$(date -Is)] Seed 45 chain ended without its final result; aborting." >&2
    exit 1
  fi
  sleep 30
done

echo "[$(date -Is)] Starting ${method} Seed 46 training on GPU ${gpu}."
bash "${train_runner}" "${gpu}" "${method}" 4 46 r1
echo "[$(date -Is)] Starting ${method} Seed 46 Epoch 1-4 Validation."
bash "${validation_runner}" "${gpu}" "${seed46_run}" 1 4
echo "[$(date -Is)] Completed ${method} Seed 46 training and Validation."
