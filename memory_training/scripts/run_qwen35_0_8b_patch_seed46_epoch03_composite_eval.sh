#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-0}"
repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
eval_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-v2-v1-10-eval-fixed-noop5-seed45-v1
run_id=qwen35-0.8b-patch-multitask-noop5-grouped-v2-v1-10-e4-b8-trainseed46-evalfixed-noop5-r1
run_dir="${workspace}/runs/${run_id}"
checkpoint="${run_dir}/checkpoints/epoch-03"
validation_output="${run_dir}/eval-fixed-composite-validation-epoch-03.json"
validation_log="${run_dir}/eval-fixed-composite-validation-epoch-03.log"
test_output="${run_dir}/eval-fixed-composite-test-epoch-03"
pipeline_log="${run_dir}/eval-fixed-composite-epoch-03-pipeline.log"

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"
exec > >(tee -a "${pipeline_log}") 2>&1

echo "[$(date -Is)] Starting seed46 epoch-03 composite-policy validation on GPU ${gpu}."
"${python_bin}" -m memory_training.validate_hf_checkpoint_v2 \
  --run-id "${run_id}" --checkpoint "${checkpoint}" --output "${validation_output}" \
  --workspace "${workspace}" --data-root "${eval_root}" --catalog-path "${eval_root}/catalog.sqlite" \
  --validation-scenarios 81 82 83 84 85 111 --full-scenarios 81 82 83 84 85 111 \
  --closed-loop-only --scenario-batch-size 3 --quiz-batch-size 16 --no-status-update \
  >"${validation_log}" 2>&1

echo "[$(date -Is)] Starting seed46 epoch-03 fixed test on GPU ${gpu}."
"${python_bin}" -m memory_training.evaluate_hf_closed_loop \
  --model qwen3.5-0.8b --method patch --checkpoint "${checkpoint}" \
  --output-dir "${test_output}" --workspace "${workspace}" \
  --data-root "${eval_root}" --catalog-path "${eval_root}/catalog.sqlite" \
  --scenarios $(seq 86 100) $(seq 112 120) \
  --max-length 2048 --max-new-tokens 768 --quiz-max-new-tokens 256 \
  --scenario-batch-size 16 --quiz-batch-size 16

echo "[$(date -Is)] COMPLETED seed46 epoch-03 validation and test."
