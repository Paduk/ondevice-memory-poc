#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"
run_id="${2:?Run ID is required}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2
run_dir="${workspace}/runs/${run_id}"
candidate_manifest="${run_dir}/mini-validation-candidates.json"

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

"${python_bin}" -m memory_training.validation_candidates mini \
  --run-dir "${run_dir}" \
  --output "${candidate_manifest}" \
  --false-update-threshold 0.20

mapfile -t checkpoints < <(jq -r '.candidates[].checkpoint' "${candidate_manifest}")
for checkpoint in "${checkpoints[@]}"; do
  epoch=$(basename "${checkpoint}" | sed 's/^epoch-//')
  "${python_bin}" -m memory_training.validate_hf_checkpoint_v2 \
    --run-id "${run_id}" \
    --checkpoint "${checkpoint}" \
    --output "${run_dir}/validation-v2-candidate-epoch-${epoch}.json" \
    --validation-scenarios 81 82 83 84 85 111 \
    --full-scenarios 81 82 83 84 85 111 \
    --closed-loop-only \
    --scenario-batch-size 6 \
    --quiz-batch-size 16 \
    --finalize-run
done

"${python_bin}" -m memory_training.validation_candidates full \
  --run-dir "${run_dir}" \
  --output "${run_dir}/best-full-validation-checkpoint.json"

winner=$(jq -r '.winner.checkpoint' "${run_dir}/best-full-validation-checkpoint.json")
winner_epoch=$(basename "${winner}")
test_output="${run_dir}/test-hf-closed-loop-s86-s100-t12-t20-best-${winner_epoch}"

"${python_bin}" -m memory_training.evaluate_hf_closed_loop \
  --model granite4-350m \
  --method patch \
  --checkpoint "${winner}" \
  --output-dir "${test_output}" \
  --workspace "${workspace}" \
  --data-root "${data_root}" \
  --catalog-path "${data_root}/catalog.sqlite" \
  --scenarios 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 112 113 114 115 116 117 118 119 120 \
  --max-length 2048 \
  --max-new-tokens 768 \
  --quiz-max-new-tokens 256 \
  --scenario-batch-size 16 \
  --quiz-batch-size 16
