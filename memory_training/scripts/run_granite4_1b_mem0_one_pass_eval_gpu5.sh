#!/usr/bin/env bash
set -euo pipefail

repo=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/mem0-style-one-pass-eval-fixed-noop5-seed45-v1
run_id=granite4-1b-mem0-one-pass-multitask-noop5-e4-b8-trainseed45-r1
run_dir=${workspace}/runs/${run_id}
checkpoint=$(${python} -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "${run_dir}/best-checkpoint.json")
epoch=$(basename "${checkpoint}")

export CUDA_VISIBLE_DEVICES=5 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PALMCLAW_MEM0_RETRIEVAL_DEVICE=cuda
export PYTHONPATH="${repo}:${repo}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${repo}"
validation_output=${run_dir}/eval-fixed-v2-validation-best-${epoch}.json
validation_log=${run_dir}/eval-fixed-v2-validation-best-${epoch}.log
"${python}" -m memory_training.validate_hf_checkpoint_v2 \
  --run-id "${run_id}" --checkpoint "${checkpoint}" \
  --output "${validation_output}" --workspace "${workspace}" \
  --data-root "${data}" --catalog-path "${data}/catalog.sqlite" \
  --validation-scenarios 81 82 83 84 85 111 \
  --full-scenarios 81 82 83 84 85 111 --closed-loop-only \
  --scenario-batch-size 6 --quiz-batch-size 16 --no-status-update \
  >"${validation_log}" 2>&1

test_output=${run_dir}/eval-fixed-v2-test-best-${epoch}
test_log=${run_dir}/eval-fixed-v2-test-best-${epoch}.log
"${python}" -m memory_training.evaluate_hf_closed_loop \
  --model granite4-1b --method mem0_one_pass --checkpoint "${checkpoint}" \
  --output-dir "${test_output}" --workspace "${workspace}" \
  --data-root "${data}" --catalog-path "${data}/catalog.sqlite" \
  --scenarios 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 \
    112 113 114 115 116 117 118 119 120 \
  --max-length 2048 --max-new-tokens 768 --quiz-max-new-tokens 256 \
  --scenario-batch-size 12 --quiz-batch-size 16 \
  >"${test_log}" 2>&1
