#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index required}"
repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
train_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2
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
mkdir -p "${run_dir}"
exec > >(tee -a "${run_dir}/pipeline.log") 2>&1

train_scenarios=($(seq 15 80) $(seq 101 110) 202 205 206 214 217 223 231 232 233 236)
validation_scenarios=(81 82 83 84 85 111)

if [[ ! -d "${run_dir}/checkpoints/epoch-04/adapter" ]]; then
  echo "[$(date -Is)] Starting 2B Delta-v3 training on GPU ${gpu}."
  "${python_bin}" -m memory_training.train \
    --model qwen3.5-2b --method delta_v3 --run-id "${run_id}" \
    --workspace "${workspace}" --data-root "${train_root}" --catalog-path "${train_root}/catalog.sqlite" \
    --epochs 4 --batch-size 2 --eval-batch-size 8 --gradient-accumulation 8 \
    --learning-rate 0.0002 --weight-decay 0.01 --warmup-ratio 0.03 \
    --max-length 2048 --max-new-tokens 768 --multitask \
    --quiz-total-passes 2 --quiz-batch-size 4 --quiz-validation-rows-per-scenario 4 \
    --gold-quiz-validation-rows 0 --quiz-validation-seed 42 \
    --quiz-validation-max-new-tokens 256 --quiz-validation-batch-size 8 \
    --skip-generation-validation --skip-quiz-validation \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 --gradient-checkpointing \
    --seed 45 --training-seed 45 --noop-per-update 5 \
    --delta-v3-noop-pending-weights 40 30 15 10 5 \
    --adjacent-noop-fraction 0.30 --trajectory-fraction 0.20 \
    --train-scenarios "${train_scenarios[@]}" --validation-scenarios "${validation_scenarios[@]}" \
    --num-workers 4 --prefetch-factor 2 --pin-memory --log-steps 5 --save-steps 500 \
    --eval-steps 0 --eval-max-rows 100 --eval-max-batches 1 \
    --eval-adjacent-noop-fraction 0.50 --false-update-threshold 0.20
else
  echo "[$(date -Is)] Training already complete; reusing checkpoints."
fi

exec /home/hj153lee/PalmClaw/memory_training/scripts/run_qwen35_2b_fixed_eval.sh "${gpu}" delta_v3
