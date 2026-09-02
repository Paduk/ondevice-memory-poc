#!/usr/bin/env bash
set -euo pipefail

variant="${1:?variant must be summary-r1, patch-r2, or temporal-patch-r1}"
gpu="${2:?GPU index is required}"

repo_root="/home/hj153lee/PalmClaw"
workspace="/mnt/data/hj153lee/PalmClaw/on-device-memory-training"
python_bin="/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python"
data_parent="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training"

case "${variant}" in
  summary-r1)
    method="summary"
    run_id="qwen35-4b-summary-multitask-noop5-st-t1t10-r1"
    data_root="${data_parent}/hybrid-s1-s100-plus-temporal-t1-t20-patch-t1t10-v2"
    ;;
  patch-r2)
    method="patch"
    run_id="qwen35-4b-patch-multitask-noop5-st-t1t10-r2"
    data_root="${data_parent}/hybrid-s1-s100-plus-temporal-t1-t20-patch-t1t10-v2"
    ;;
  temporal-patch-r1)
    method="temporal_patch"
    run_id="qwen35-4b-temporal-patch-multitask-noop5-st-t1t10-r1"
    data_root="${data_parent}/hybrid-s1-s100-plus-temporal-t1-t20-temporal-patch-t1t10-terra-audited-v2"
    ;;
  *)
    echo "Unknown variant: ${variant}" >&2
    exit 2
    ;;
esac

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

train_scenarios=($(seq 15 80) $(seq 101 110))

exec "${python_bin}" -m memory_training.train \
  --model qwen3.5-4b \
  --method "${method}" \
  --run-id "${run_id}" \
  --workspace "${workspace}" \
  --data-root "${data_root}" \
  --catalog-path "${data_root}/catalog.sqlite" \
  --epochs 4 \
  --batch-size 2 \
  --eval-batch-size 1 \
  --gradient-accumulation 8 \
  --learning-rate 0.0002 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --max-length 2048 \
  --max-new-tokens 768 \
  --multitask \
  --quiz-total-passes 2 \
  --quiz-batch-size 2 \
  --quiz-validation-rows-per-scenario 10 \
  --gold-quiz-validation-rows 50 \
  --quiz-validation-seed 42 \
  --quiz-validation-max-new-tokens 256 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --gradient-checkpointing \
  --seed 45 \
  --noop-per-update 5 \
  --adjacent-noop-fraction 0.30 \
  --trajectory-fraction 0.20 \
  --train-scenarios "${train_scenarios[@]}" \
  --validation-scenarios 81 82 83 84 85 111 \
  --num-workers 4 \
  --prefetch-factor 2 \
  --pin-memory \
  --log-steps 5 \
  --save-steps 500 \
  --eval-steps 0 \
  --eval-max-rows 512 \
  --eval-adjacent-noop-fraction 0.50 \
  --closed-loop-full-scenarios 83 84 111 \
  --closed-loop-sparse-scenarios 82 85 \
  --closed-loop-quiz \
  --closed-loop-sparse-noop-keep-fraction 0.10 \
  --false-update-threshold 0.20
