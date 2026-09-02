#!/usr/bin/env bash
set -euo pipefail

source /mnt/data/miniconda3/bin/activate \
  /mnt/data/hj153lee/conda-envs/palmclaw-memory-sft

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/home/hj153lee/PalmClaw
export TOKENIZERS_PARALLELISM=false

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/hybrid-s1-s100-date-batch-v1
catalog_path="$workspace/cache/dataset/date-batch-v1.sqlite"
log_root="$workspace/logs"
mkdir -p "$log_root"
cd "$repo_root"

common_args=(
  --model qwen3.5-4b
  --method summary_batch
  --workspace "$workspace"
  --data-root "$data_root"
  --catalog-path "$catalog_path"
  --epochs 2
  --batch-size 2
  --quiz-batch-size 2
  --gradient-accumulation 8
  --learning-rate 2e-4
  --max-length 8192
  --max-new-tokens 768
  --seed 42
  --noop-per-update 5
  --adjacent-noop-fraction 0.30
  --trajectory-fraction 0.20
  --multitask
  --quiz-total-passes 2
  --gradient-checkpointing
  --log-steps 5
  --eval-steps 0
  --save-steps 500
)

python -m memory_training.train \
  "${common_args[@]}" \
  --run-id qwen35-4b-summary-batch-multitask-e2-canary-v2 \
  --num-workers 0 \
  --max-train-steps 1 \
  --eval-max-rows 128 \
  --eval-max-batches 1 \
  --skip-generation-validation \
  --skip-quiz-validation \
  2>&1 | tee "$log_root/qwen35-4b-summary-batch-multitask-e2-canary-v2.log"

python -m memory_training.train \
  "${common_args[@]}" \
  --run-id qwen35-4b-summary-batch-multitask-e2-r1 \
  --num-workers 4 \
  --prefetch-factor 2 \
  --eval-max-rows 128 \
  --eval-max-batches 128 \
  --one-step-max-rows 50 \
  --quiz-validation-rows-per-scenario 2 \
  --closed-loop-full-scenarios 83 \
  --closed-loop-sparse-scenarios 82 85 \
  --closed-loop-sparse-noop-keep-fraction 0.10 \
  2>&1 | tee "$log_root/qwen35-4b-summary-batch-multitask-e2-r1.log"
