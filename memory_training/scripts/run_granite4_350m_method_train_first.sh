#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"
method="${2:?Method is required: summary or delta_v3}"
epochs="${3:-4}"
training_seed="${4:-45}"
run_tag="${5:-r1}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2

case "${method}" in
  summary)
    method_slug=summary
    method_args=()
    ;;
  delta_v3)
    method_slug=delta-v3
    method_args=(--delta-v3-noop-pending-weights 40 30 15 10 5)
    ;;
  *)
    echo "Unsupported method: ${method}" >&2
    exit 2
    ;;
esac

run_id="granite4-350m-${method_slug}-multitask-noop5-trainfirst-grouped-v2-v1-10-e${epochs}-b8-trainseed${training_seed}-${run_tag}"

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

train_scenarios=(
  $(seq 15 80)
  $(seq 101 110)
  202 205 206 214 217 223 231 232 233 236
)

exec "${python_bin}" -m memory_training.train \
  --model granite4-350m \
  --method "${method}" \
  --run-id "${run_id}" \
  --workspace "${workspace}" \
  --data-root "${data_root}" \
  --catalog-path "${data_root}/catalog.sqlite" \
  --epochs "${epochs}" \
  --batch-size 8 \
  --eval-batch-size 16 \
  --gradient-accumulation 2 \
  --learning-rate 0.0002 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --max-length 2048 \
  --max-new-tokens 768 \
  --multitask \
  --quiz-total-passes 2 \
  --quiz-batch-size 8 \
  --quiz-validation-rows-per-scenario 4 \
  --gold-quiz-validation-rows 0 \
  --quiz-validation-seed 42 \
  --quiz-validation-max-new-tokens 256 \
  --quiz-validation-batch-size 16 \
  --skip-generation-validation \
  --skip-quiz-validation \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --gradient-checkpointing \
  --seed 45 \
  --training-seed "${training_seed}" \
  --noop-per-update 5 \
  "${method_args[@]}" \
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
  --eval-max-rows 100 \
  --eval-max-batches 1 \
  --eval-adjacent-noop-fraction 0.50 \
  --false-update-threshold 0.20
