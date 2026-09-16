#!/usr/bin/env bash
set -euo pipefail

repo=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/mem0-style-one-pass-eval-fixed-noop5-seed45-v1
run=${workspace}/runs/granite4-1b-mem0-one-pass-multitask-noop5-e4-b8-trainseed45-r1
output=${run}/eval-diagnostic-invalid-s100-s112-epoch-02

export CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PALMCLAW_MEM0_RETRIEVAL_DEVICE=cuda
export PYTHONPATH="${repo}:${repo}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${repo}"
"${python}" -m memory_training.evaluate_hf_closed_loop \
  --model granite4-1b --method mem0_one_pass \
  --checkpoint "${run}/checkpoints/epoch-02" --output-dir "${output}" \
  --workspace "${workspace}" --data-root "${data}" \
  --catalog-path "${data}/catalog.sqlite" --scenarios 100 112 \
  --max-length 2048 --max-new-tokens 768 --quiz-max-new-tokens 256 \
  --scenario-batch-size 2 --quiz-batch-size 16 \
  >"${output}.log" 2>&1
