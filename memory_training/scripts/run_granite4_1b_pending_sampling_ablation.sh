#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"
condition="${2:?Condition must be unstratified or depth-weighted}"
training_seed="${3:?Training seed is required, e.g. 45, 46, or 47}"
run_tag="${4:-r1}"

case "${condition}" in
  unstratified|depth-weighted) ;;
  *)
    echo "Condition must be unstratified or depth-weighted: ${condition}" >&2
    exit 2
    ;;
esac

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_parent=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training
train_root="${data_parent}/grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2-delta-v3-compact-k5-v1"
eval_root="${data_parent}/grouped-v2-v1-10-eval-fixed-noop5-seed45-v1-delta-v3-compact-k5-v1"
method=delta_v3_compact_k5
epochs=4
run_id="granite4-1b-${method}-pending-ablation-${condition}-e${epochs}-b2-seed${training_seed}-${run_tag}"
run_dir="${workspace}/runs/${run_id}"

for root in "${train_root}" "${eval_root}"; do
  if [[ ! -f "${root}/COMPLETED" || ! -f "${root}/catalog.sqlite" ]]; then
    echo "Prepared data root is incomplete: ${root}" >&2
    exit 2
  fi
done

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

train_scenarios=(
  $(seq 15 80)
  $(seq 101 110)
  202 205 206 214 217 223 231 232 233 236
)
validation_scenarios=(81 82 83 84 85 111)
test_scenarios=(
  $(seq 86 100)
  $(seq 112 120)
)

final_epoch_label=$(printf '%02d' "${epochs}")
final_checkpoint="${run_dir}/checkpoints/epoch-${final_epoch_label}"
if [[ ! -d "${final_checkpoint}/adapter" ]]; then
  echo "[$(date -Is)] Starting ${condition}, seed ${training_seed}."
  "${python_bin}" -m memory_training.train \
    --model granite4-1b \
    --method "${method}" \
    --run-id "${run_id}" \
    --workspace "${workspace}" \
    --data-root "${train_root}" \
    --catalog-path "${train_root}/catalog.sqlite" \
    --epochs "${epochs}" \
    --batch-size 2 \
    --eval-batch-size 2 \
    --gradient-accumulation 8 \
    --learning-rate 0.0002 \
    --weight-decay 0.01 \
    --warmup-ratio 0.03 \
    --max-length 4096 \
    --max-new-tokens 768 \
    --multitask \
    --quiz-total-passes 2 \
    --quiz-batch-size 2 \
    --quiz-validation-rows-per-scenario 4 \
    --gold-quiz-validation-rows 0 \
    --quiz-validation-seed 42 \
    --quiz-validation-max-new-tokens 256 \
    --quiz-validation-batch-size 2 \
    --skip-generation-validation \
    --skip-quiz-validation \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --gradient-checkpointing \
    --seed 45 \
    --training-seed "${training_seed}" \
    --noop-per-update 5 \
    --delta-v3-noop-sampling "${condition}" \
    --delta-v3-noop-pending-weights 40 30 15 10 5 \
    --adjacent-noop-fraction 0.30 \
    --trajectory-fraction 0.20 \
    --trajectory-sampling-seed 45 \
    --train-scenarios "${train_scenarios[@]}" \
    --validation-scenarios "${validation_scenarios[@]}" \
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
else
  echo "[$(date -Is)] Training already complete; reusing checkpoints."
fi

for epoch in $(seq 1 "${epochs}"); do
  epoch_label=$(printf '%02d' "${epoch}")
  checkpoint="${run_dir}/checkpoints/epoch-${epoch_label}"
  output="${run_dir}/eval-fixed-validation-epoch-${epoch_label}.json"
  if [[ -s "${output}" ]]; then
    continue
  fi
  echo "[$(date -Is)] Validating epoch ${epoch_label}."
  "${python_bin}" -m memory_training.validate_hf_checkpoint_v2 \
    --run-id "${run_id}" \
    --checkpoint "${checkpoint}" \
    --output "${output}" \
    --workspace "${workspace}" \
    --data-root "${eval_root}" \
    --catalog-path "${eval_root}/catalog.sqlite" \
    --validation-scenarios "${validation_scenarios[@]}" \
    --full-scenarios "${validation_scenarios[@]}" \
    --closed-loop-only \
    --scenario-batch-size 3 \
    --quiz-batch-size 16 \
    --no-status-update \
    >"${run_dir}/eval-fixed-validation-epoch-${epoch_label}.log" 2>&1
done

selection="${run_dir}/eval-fixed-best-checkpoint.json"
"${python_bin}" -m memory_training.select_fixed_validation_checkpoint \
  --run-dir "${run_dir}" \
  --epochs 1 2 3 4 \
  --output "${selection}"

winner=$(jq -r '.winner.checkpoint' "${selection}")
winner_epoch=$(basename "${winner}")
test_output="${run_dir}/eval-fixed-test-best-${winner_epoch}"
if [[ ! -s "${test_output}/summary.json" ]]; then
  echo "[$(date -Is)] Running closed-loop Test with ${winner_epoch}."
  "${python_bin}" -m memory_training.evaluate_hf_closed_loop \
    --model granite4-1b \
    --method "${method}" \
    --checkpoint "${winner}" \
    --output-dir "${test_output}" \
    --workspace "${workspace}" \
    --data-root "${eval_root}" \
    --catalog-path "${eval_root}/catalog.sqlite" \
    --scenarios "${test_scenarios[@]}" \
    --max-length 4096 \
    --max-new-tokens 768 \
    --quiz-max-new-tokens 256 \
    --scenario-batch-size 16 \
    --quiz-batch-size 16
fi

diagnostic="${run_dir}/pending-depth-test-${winner_epoch}.json"
if [[ ! -s "${diagnostic}" ]]; then
  echo "[$(date -Is)] Running fixed gold-state pending-depth diagnostic."
  "${python_bin}" -m memory_training.evaluate_delta_pending_depth \
    --run-id "${run_id}" \
    --checkpoint "${winner}" \
    --output "${diagnostic}" \
    --workspace "${workspace}" \
    --data-root "${eval_root}" \
    --catalog-path "${eval_root}/catalog.sqlite" \
    --scenarios "${test_scenarios[@]}" \
    --rows-per-depth-decision 64 \
    --evaluation-seed 42 \
    --batch-size 8 \
    --max-length 4096 \
    --max-new-tokens 768
fi

echo "[$(date -Is)] Completed ${condition}, seed ${training_seed}."
