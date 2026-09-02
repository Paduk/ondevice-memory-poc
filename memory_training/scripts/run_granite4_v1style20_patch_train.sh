#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"
model="${2:?Model key is required (granite4-350m or granite4-1b)}"
epochs="${3:-4}"
training_seed="${4:-45}"
run_tag="${5:-r1}"

case "${model}" in
  granite4-350m) batch_size=16 ;;
  granite4-1b) batch_size=12 ;;
  *) echo "Unsupported model: ${model}" >&2; exit 2 ;;
esac

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-plus-v1-style-20-v1
gradient_accumulation=1
run_id="${model}-patch-multitask-noop5-v1style20-grouped-v2-v1-10-e${epochs}-b${batch_size}-trainseed${training_seed}-${run_tag}"

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

train_scenarios=(
  $(seq 15 80)
  $(seq 101 110)
  202 205 206 214 217 223 231 232 233 236
  $(seq 301 320)
)

echo "[$(date -Is)] Training ${run_id} on GPU ${gpu}."
echo "[$(date -Is)] Physical batch=${batch_size}, accumulation=${gradient_accumulation}, effective batch=16."

exec "${python_bin}" -m memory_training.train \
  --model "${model}" \
  --method patch \
  --run-id "${run_id}" \
  --workspace "${workspace}" \
  --data-root "${data_root}" \
  --catalog-path "${data_root}/catalog.sqlite" \
  --epochs "${epochs}" \
  --batch-size "${batch_size}" \
  --eval-batch-size 16 \
  --gradient-accumulation "${gradient_accumulation}" \
  --learning-rate 0.0002 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --max-length 2048 \
  --max-new-tokens 768 \
  --multitask \
  --quiz-total-passes 2 \
  --quiz-batch-size "${batch_size}" \
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
