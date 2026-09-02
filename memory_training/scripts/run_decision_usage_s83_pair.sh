#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/hj153lee/PalmClaw"
runs_root="/mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs"
python_bin="/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python"
gpu="${1:-4}"

export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "${repo_root}"

run_one() {
  local method="$1"
  local run_id="qwen35-4b-${method}-multitask-noop5-r1"
  local run_root="${runs_root}/${run_id}"
  local checkpoint="${run_root}/checkpoints/epoch-04"
  local output="${run_root}/test-hf-decision-usage-s83-epoch04"

  echo "[$(date -Is)] Starting ${method} S83 decision-usage evaluation on GPU ${gpu}."
  "${python_bin}" -m memory_training.evaluate_hf_closed_loop \
    --model qwen3.5-4b \
    --method "${method}" \
    --checkpoint "${checkpoint}" \
    --output-dir "${output}" \
    --scenarios 83 \
    --max-length 2048 \
    --max-new-tokens 768 \
    --quiz-max-new-tokens 256 \
    --scenario-batch-size 1 \
    --quiz-batch-size 16
  echo "[$(date -Is)] Completed ${method} S83 decision-usage evaluation."
}

run_one patch
run_one summary
echo "[$(date -Is)] Both S83 decision-usage evaluations completed."
