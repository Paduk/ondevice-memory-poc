#!/usr/bin/env bash
set -euo pipefail

MEM0_TWO_STAGE_GPU="${MEM0_TWO_STAGE_GPU:-4}"
export CUDA_VISIBLE_DEVICES="$MEM0_TWO_STAGE_GPU"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="/home/hj153lee/PalmClaw${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON=/mnt/nvme2/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
DATA_ROOT=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/mem0-style-two-stage-training-v1
WORKSPACE=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
RUN_ID=granite4-1b-mem0-two-stage-multitask-extract1to5-managerall-e4-b8-trainseed45-r1

exec "$PYTHON" -m memory_training.train \
  --model granite4-1b \
  --method mem0_two_stage \
  --run-id "$RUN_ID" \
  --workspace "$WORKSPACE" \
  --data-root "$DATA_ROOT" \
  --catalog-path "$DATA_ROOT/catalog.sqlite" \
  --epochs 4 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --gradient-accumulation 2 \
  --learning-rate 2e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --max-length 2048 \
  --max-new-tokens 768 \
  --multitask \
  --quiz-total-passes 2 \
  --quiz-batch-size 4 \
  --skip-quiz-validation \
  --quiz-validation-rows-per-scenario 4 \
  --gold-quiz-validation-rows 0 \
  --noop-per-update 5 \
  --adjacent-noop-fraction 0.3 \
  --trajectory-fraction 0.2 \
  --train-scenarios \
    15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
    31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 \
    47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 \
    63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 \
    79 80 101 102 103 104 105 106 107 108 109 110 \
    202 205 206 214 217 223 231 232 233 236 \
  --validation-scenarios 81 82 83 84 85 111 \
  --num-workers 4 \
  --prefetch-factor 2 \
  --log-steps 5 \
  --save-steps 500 \
  --eval-steps 0 \
  --eval-max-rows 256 \
  --eval-max-batches 1 \
  --skip-generation-validation \
  --quiz-validation-batch-size 8 \
  --seed 45 \
  --training-seed 45
