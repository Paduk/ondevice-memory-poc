#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?GPU index required}
model=${2:?Model required: llama3.2-1b or llama3.2-3b}

case "${model}" in
  llama3.2-1b)
    model_slug=llama3.2-1b
    batch_size=8
    accumulation=2
    quiz_batch_size=8
    test_batch_size=16
    ;;
  llama3.2-3b)
    model_slug=llama3.2-3b
    batch_size=4
    accumulation=4
    quiz_batch_size=4
    test_batch_size=8
    ;;
  *)
    echo "Unsupported model: ${model}" >&2
    exit 2
    ;;
esac

repo=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
train_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/mem0-style-one-pass-training-s21t10-no-v1-v2
eval_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/mem0-style-one-pass-eval-fixed-noop5-seed45-v1
run_id=${model_slug}-mem0-one-pass-multitask-noop5-e4-b${batch_size}-trainseed45-r1
run_dir=${workspace}/runs/${run_id}

export CUDA_VISIBLE_DEVICES="${gpu}" HF_HOME="${workspace}/cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PALMCLAW_MEM0_RETRIEVAL_DEVICE=cuda
export PYTHONPATH="${repo}:${repo}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

train_scenarios=(
  $(seq 21 80) $(seq 101 110)
)
validation_scenarios=(81 82 83 84 85 111)
test_scenarios=($(seq 86 100) $(seq 112 120))

mkdir -p "${run_dir}"
exec > >(tee -a "${run_dir}/pipeline.log") 2>&1
cd "${repo}"

final_checkpoint=${run_dir}/checkpoints/epoch-04
if [[ ! -d "${final_checkpoint}/adapter" ]]; then
  echo "[$(date -Is)] Starting ${model} Mem0 one-pass training on GPU ${gpu}."
  "${python}" -m memory_training.train \
    --model "${model}" --method mem0_one_pass --run-id "${run_id}" \
    --workspace "${workspace}" --data-root "${train_root}" \
    --catalog-path "${train_root}/catalog.sqlite" \
    --epochs 4 --batch-size "${batch_size}" --eval-batch-size 8 \
    --gradient-accumulation "${accumulation}" --learning-rate 2e-4 \
    --weight-decay 0.01 --warmup-ratio 0.03 \
    --max-length 2048 --max-new-tokens 768 \
    --multitask --quiz-total-passes 2 --quiz-batch-size "${quiz_batch_size}" \
    --skip-quiz-validation --quiz-validation-rows-per-scenario 4 \
    --gold-quiz-validation-rows 0 --quiz-validation-seed 42 \
    --quiz-validation-max-new-tokens 256 --quiz-validation-batch-size 8 \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 \
    --gradient-checkpointing --seed 45 --training-seed 45 \
    --noop-per-update 5 --adjacent-noop-fraction 0.30 \
    --trajectory-fraction 0.20 \
    --train-scenarios "${train_scenarios[@]}" \
    --validation-scenarios "${validation_scenarios[@]}" \
    --num-workers 4 --prefetch-factor 2 --pin-memory \
    --log-steps 5 --save-steps 0 --eval-steps 0 \
    --eval-max-rows 100 --eval-max-batches 1 \
    --eval-adjacent-noop-fraction 0.50 --false-update-threshold 0.20 \
    --skip-generation-validation
else
  echo "[$(date -Is)] Training checkpoint already exists; reusing it."
fi

checkpoint=$("${python}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "${run_dir}/best-checkpoint.json")
epoch=$(basename "${checkpoint}")
validation_output=${run_dir}/eval-fixed-v2-validation-best-${epoch}.json
if [[ ! -s "${validation_output}" ]]; then
  echo "[$(date -Is)] Starting best-checkpoint Validation (${epoch})."
  "${python}" -m memory_training.validate_hf_checkpoint_v2 \
    --run-id "${run_id}" --checkpoint "${checkpoint}" \
    --output "${validation_output}" --workspace "${workspace}" \
    --data-root "${eval_root}" --catalog-path "${eval_root}/catalog.sqlite" \
    --validation-scenarios "${validation_scenarios[@]}" \
    --full-scenarios "${validation_scenarios[@]}" --closed-loop-only \
    --scenario-batch-size "${test_batch_size}" --quiz-batch-size 16 \
    --no-status-update \
    >"${run_dir}/eval-fixed-v2-validation-best-${epoch}.log" 2>&1
fi

test_output=${run_dir}/eval-fixed-v2-test-best-${epoch}
if [[ $(jq -r '.complete // false' "${test_output}/summary.json" 2>/dev/null || true) != true ]]; then
  echo "[$(date -Is)] Starting fixed 24-scenario Test (${epoch})."
  "${python}" -m memory_training.evaluate_hf_closed_loop \
    --model "${model}" --method mem0_one_pass --checkpoint "${checkpoint}" \
    --output-dir "${test_output}" --workspace "${workspace}" \
    --data-root "${eval_root}" --catalog-path "${eval_root}/catalog.sqlite" \
    --scenarios "${test_scenarios[@]}" --max-length 2048 \
    --max-new-tokens 768 --quiz-max-new-tokens 256 \
    --scenario-batch-size "${test_batch_size}" --quiz-batch-size 16 \
    >"${run_dir}/eval-fixed-v2-test-best-${epoch}.log" 2>&1
fi
echo "[$(date -Is)] COMPLETED ${model} Mem0 one-pass training, Validation, and Test."
