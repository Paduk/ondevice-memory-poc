#!/usr/bin/env bash
set -euo pipefail

repo=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-v2-v1-10-eval-fixed-noop5-seed45-v1
export CUDA_VISIBLE_DEVICES=5 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${repo}:${repo}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

runs=(
  qwen35-0.8b-patch-multitask-noop5-grouped-v2-v1-10-e4-b8-trainseed45-evalfixed-noop5-r1
  qwen35-0.8b-delta-v3-multitask-noop5-grouped-v2-v1-10-e4-b4-trainseed45-evalfixed-noop5-r2
)

validate() {
  local run_id=$1 run_dir checkpoint epoch output log
  run_dir="${workspace}/runs/${run_id}"
  checkpoint=$(jq -r '.winner.checkpoint' "${run_dir}/eval-fixed-best-checkpoint.json")
  epoch=$(basename "${checkpoint}")
  output="${run_dir}/eval-fixed-validation-evalseed777-best-${epoch}.json"
  log="${run_dir}/eval-fixed-validation-evalseed777-best-${epoch}.log"
  "${python}" -m memory_training.validate_hf_checkpoint_v2 \
    --run-id "${run_id}" --checkpoint "${checkpoint}" --output "${output}" \
    --workspace "${workspace}" --data-root "${data}" --catalog-path "${data}/catalog.sqlite" \
    --evaluation-seed 777 --validation-scenarios 81 82 83 84 85 111 \
    --full-scenarios 81 82 83 84 85 111 --closed-loop-only \
    --scenario-batch-size 3 --quiz-batch-size 16 --no-status-update \
    >"${log}" 2>&1
}

cd "${repo}"
validate "${runs[0]}" & first=$!
validate "${runs[1]}" & second=$!
failed=0
wait "${first}" || failed=1
wait "${second}" || failed=1
exit "${failed}"
