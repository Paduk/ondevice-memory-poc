#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"
model="${2:?model key is required: qwen3.5-2b, qwen3.5-4b, or qwen3.5-9b}"
method="${3:?method is required: summary, patch, or delta_v3}"
training_seed="${4:?training seed is required}"
epochs="${5:-4}"
run_tag="${6:-r1}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
split_config="${repo_root}/memory_training/configs/canary_12pct_operation_matched_v1.json"
data_root="$(${python_bin} -c 'import json,sys; print(json.load(open(sys.argv[1]))["data_root"])' "${split_config}")"

case "${model}" in
  qwen3.5-2b|qwen3.5-4b)
    batch_size=2
    gradient_accumulation=8
    ;;
  qwen3.5-9b)
    batch_size=1
    gradient_accumulation=16
    ;;
  *)
    echo "Unsupported model: ${model}" >&2
    exit 2
    ;;
esac

case "${method}" in
  summary|patch)
    method_args=()
    ;;
  delta_v3)
    method_args=(--delta-v3-noop-pending-weights 40 30 15 10 5)
    ;;
  *)
    echo "Unsupported method: ${method}" >&2
    exit 2
    ;;
esac

mapfile -t train_scenarios < <(
  "${python_bin}" -c \
    'import json,sys; print(*json.load(open(sys.argv[1]))["train_scenarios"], sep="\n")' \
    "${split_config}"
)
mapfile -t validation_scenarios < <(
  "${python_bin}" -c \
    'import json,sys; print(*json.load(open(sys.argv[1]))["validation"]["scenarios"], sep="\n")' \
    "${split_config}"
)

model_slug="${model//./}"
method_slug="${method//_/-}"
run_id="${model_slug}-${method_slug}-canary12pct-opmatch-e${epochs}-trainseed${training_seed}-${run_tag}"

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python_bin}" -m memory_training.train \
  --model "${model}" \
  --method "${method}" \
  --run-id "${run_id}" \
  --workspace "${workspace}" \
  --data-root "${data_root}" \
  --catalog-path "${data_root}/catalog.sqlite" \
  --epochs "${epochs}" \
  --batch-size "${batch_size}" \
  --eval-batch-size 1 \
  --gradient-accumulation "${gradient_accumulation}" \
  --learning-rate 0.0002 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --max-length 2048 \
  --max-new-tokens 768 \
  --multitask \
  --quiz-total-passes 2 \
  --quiz-batch-size 2 \
  --quiz-validation-rows-per-scenario 2 \
  --gold-quiz-validation-rows 10 \
  --quiz-validation-seed 42 \
  --quiz-validation-max-new-tokens 256 \
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
  --validation-scenarios "${validation_scenarios[@]}" \
  --num-workers 4 \
  --prefetch-factor 2 \
  --pin-memory \
  --log-steps 5 \
  --save-steps 200 \
  --eval-steps 0 \
  --eval-max-rows 100 \
  --one-step-max-rows 100 \
  --eval-adjacent-noop-fraction 0.50 \
  --closed-loop-full-scenarios 83 \
  --closed-loop-sparse-scenarios 111 \
  --closed-loop-quiz \
  --closed-loop-sparse-noop-keep-fraction 0.10 \
  --false-update-threshold 0.20
