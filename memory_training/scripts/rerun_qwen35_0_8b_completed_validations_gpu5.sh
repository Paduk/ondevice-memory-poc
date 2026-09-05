#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
eval_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-v2-v1-10-eval-fixed-noop5-seed45-v1
eval_catalog="${eval_root}/catalog.sqlite"

export CUDA_VISIBLE_DEVICES=5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

runs=(
  qwen35-0.8b-patch-multitask-noop5-grouped-v2-v1-10-e4-b8-trainseed45-evalfixed-noop5-r1
  qwen35-0.8b-delta-v3-multitask-noop5-grouped-v2-v1-10-e4-b4-trainseed45-evalfixed-noop5-r2
  qwen35-0.8b-summary-multitask-noop5-grouped-v2-v1-10-e4-b4-trainseed45-evalfixed-noop5-r2
  qwen35-0.8b-patch-multitask-noop5-grouped-v2-v1-10-e4-b8-trainseed46-evalfixed-noop5-r1
)

validate_run() {
  local run_id=$1
  local run_dir checkpoint epoch output log
  run_dir="${workspace}/runs/${run_id}"
  checkpoint=$(jq -r '.winner.checkpoint' "${run_dir}/eval-fixed-best-checkpoint.json")
  epoch=$(basename "${checkpoint}")
  output="${run_dir}/eval-fixed-validation-repeat2-best-${epoch}.json"
  log="${run_dir}/eval-fixed-validation-repeat2-best-${epoch}.log"
  echo "[$(date -Is)] Revalidating ${run_id} ${epoch}."
  "${python_bin}" -m memory_training.validate_hf_checkpoint_v2 \
    --run-id "${run_id}" \
    --checkpoint "${checkpoint}" \
    --output "${output}" \
    --workspace "${workspace}" \
    --data-root "${eval_root}" \
    --catalog-path "${eval_catalog}" \
    --validation-scenarios 81 82 83 84 85 111 \
    --full-scenarios 81 82 83 84 85 111 \
    --closed-loop-only \
    --scenario-batch-size 3 \
    --quiz-batch-size 16 \
    --no-status-update \
    >"${log}" 2>&1
  echo "[$(date -Is)] Completed ${run_id} ${epoch}."
}

cd "${repo_root}"
for ((start=0; start<${#runs[@]}; start+=2)); do
  validate_run "${runs[start]}" &
  first_pid=$!
  validate_run "${runs[start+1]}" &
  second_pid=$!
  failed=0
  wait "${first_pid}" || failed=1
  wait "${second_pid}" || failed=1
  if (( failed )); then
    echo "[$(date -Is)] A repeat validation failed in wave $((start / 2 + 1))." >&2
    exit 1
  fi
done

echo "[$(date -Is)] All four repeat validations completed."
