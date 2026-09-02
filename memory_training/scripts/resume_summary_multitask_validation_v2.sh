#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 RUN_ID GPU SEED NOOP_PER_UPDATE CHECKPOINT" >&2
  exit 2
fi

run_id=$1
gpu=$2
seed=$3
noop_per_update=$4
checkpoint=$5

source /mnt/data/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/data/hj153lee/conda-envs/palmclaw-memory-sft
cd /home/hj153lee/PalmClaw

export CUDA_VISIBLE_DEVICES="$gpu"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

exec python -m memory_training.train \
  --model qwen3.5-4b \
  --method summary \
  --run-id "$run_id" \
  --epochs 4 \
  --batch-size 2 \
  --quiz-batch-size 2 \
  --gradient-accumulation 8 \
  --learning-rate 2e-4 \
  --max-length 2048 \
  --seed "$seed" \
  --noop-per-update "$noop_per_update" \
  --adjacent-noop-fraction 0.30 \
  --trajectory-fraction 0.20 \
  --multitask \
  --quiz-total-passes 2 \
  --num-workers 4 \
  --prefetch-factor 2 \
  --gradient-checkpointing \
  --log-steps 5 \
  --eval-steps 0 \
  --save-steps 500 \
  --closed-loop-full-scenarios 83 84 \
  --closed-loop-sparse-scenarios 82 85 \
  --closed-loop-sparse-noop-keep-fraction 0.10 \
  --closed-loop-quiz \
  --gold-quiz-validation-rows 50 \
  --quiz-validation-seed 42 \
  --resume "$checkpoint"
