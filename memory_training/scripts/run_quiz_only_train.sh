#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?GPU index required}
model=${2:?Model key required}

case "${model}" in
  granite4-350m)
    model_slug=granite4-350m
    quiz_batch_size=8
    accumulation=2
    ;;
  granite4-1b)
    model_slug=granite4-1b
    quiz_batch_size=4
    accumulation=2
    ;;
  qwen3.5-0.8b)
    model_slug=qwen35-0.8b
    quiz_batch_size=8
    accumulation=2
    ;;
  qwen3.5-2b)
    model_slug=qwen35-2b
    quiz_batch_size=4
    accumulation=4
    ;;
  llama3.2-1b)
    model_slug=llama3.2-1b
    quiz_batch_size=8
    accumulation=2
    ;;
  llama3.2-3b)
    model_slug=llama3.2-3b
    quiz_batch_size=4
    accumulation=4
    ;;
  *)
    echo "Unsupported Quiz-only model: ${model}" >&2
    exit 2
    ;;
esac

repo=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-s1-s100-plus-temporal-t1-t20-v2
run_id=${model_slug}-quiz-only-e4-qpasses2-qb${quiz_batch_size}-trainseed45-r1
run_dir=${workspace}/runs/${run_id}

export CUDA_VISIBLE_DEVICES="${gpu}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${repo}:${repo}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${run_dir}"
exec > >(tee -a "${run_dir}/pipeline.log") 2>&1
cd "${repo}"

echo "[$(date -Is)] Starting ${model} Quiz-only SFT on GPU ${gpu}."
"${python}" -m memory_training.train \
  --model "${model}" --method patch --quiz-only --run-id "${run_id}" \
  --workspace "${workspace}" --data-root "${data_root}" \
  --catalog-path "${data_root}/catalog.sqlite" \
  --epochs 4 --batch-size "${quiz_batch_size}" --quiz-batch-size "${quiz_batch_size}" \
  --eval-batch-size 8 --gradient-accumulation "${accumulation}" \
  --learning-rate 2e-4 --weight-decay 0.01 --warmup-ratio 0.03 \
  --max-length 2048 --quiz-total-passes 2 \
  --quiz-validation-rows-per-scenario 4 --gold-quiz-validation-rows 0 \
  --quiz-validation-seed 42 --quiz-validation-max-new-tokens 256 \
  --quiz-validation-batch-size 8 \
  --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 \
  --gradient-checkpointing --seed 45 --training-seed 45 \
  --num-workers 4 --prefetch-factor 2 --pin-memory \
  --log-steps 5 --save-steps 0 --eval-steps 0 \
  --skip-generation-validation

echo "[$(date -Is)] COMPLETED ${model} Quiz-only SFT and epoch Validation."
