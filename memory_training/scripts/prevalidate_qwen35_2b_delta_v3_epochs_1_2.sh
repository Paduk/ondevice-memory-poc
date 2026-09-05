#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-5}"
repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
eval_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-v2-v1-10-eval-fixed-noop5-seed45-v1
run_id=qwen35-2b-delta-v3-multitask-noop5-grouped-v2-v1-10-e4-b2-trainseed45-evalfixed-noop5-r1
run_dir="${workspace}/runs/${run_id}"

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

validate_epoch() {
  local epoch="$1"
  local output="${run_dir}/eval-fixed-v2-validation-epoch-${epoch}.json"
  local log="${run_dir}/eval-fixed-v2-validation-epoch-${epoch}.log"
  if [[ -s "${output}" ]]; then
    echo "epoch ${epoch} already complete; skipping"
    return
  fi
  "${python_bin}" -m memory_training.validate_hf_checkpoint_v2 \
    --run-id "${run_id}" --checkpoint "${run_dir}/checkpoints/epoch-${epoch}" \
    --output "${output}" --workspace "${workspace}" \
    --data-root "${eval_root}" --catalog-path "${eval_root}/catalog.sqlite" \
    --validation-scenarios 81 82 83 84 85 111 \
    --full-scenarios 81 82 83 84 85 111 --closed-loop-only \
    --scenario-batch-size 3 --quiz-batch-size 16 --no-status-update \
    >"${log}" 2>&1
}

echo "[$(date -Is)] Starting epoch 01 and 02 fixed validation on GPU ${gpu}."
validate_epoch 01 & p1=$!
validate_epoch 02 & p2=$!
failed=0
wait "${p1}" || failed=1
wait "${p2}" || failed=1
if (( failed )); then
  echo "[$(date -Is)] One or more validations failed." >&2
  exit 1
fi
echo "[$(date -Is)] COMPLETED epoch 01 and 02 fixed validation."
