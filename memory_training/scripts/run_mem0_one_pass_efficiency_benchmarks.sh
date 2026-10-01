#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-0}
only_model=${2:-all}
repo=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/mem0-style-one-pass-eval-fixed-noop5-seed45-v1
manifest=${workspace}/benchmarks/mem0-natural-s86-s90-manifest-20260918-v1.json
campaign=${workspace}/benchmarks/mem0-efficiency-controlled-s86-s90-20260918-v1

export CUDA_VISIBLE_DEVICES=${gpu}
export HF_HOME=${workspace}/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PALMCLAW_MEM0_RETRIEVAL_DEVICE=cuda
export PYTHONPATH=${repo}:${repo}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}

models=(
  granite4-350m
  qwen3.5-0.8b
  granite4-1b
  llama3.2-1b
  qwen3.5-2b
  llama3.2-3b
)
runs=(
  granite4-350m-mem0-one-pass-multitask-noop5-e4-b8-trainseed45-r1
  qwen35-0.8b-mem0-one-pass-multitask-noop5-e4-b8-trainseed45-r1
  granite4-1b-mem0-one-pass-multitask-noop5-e4-b8-trainseed45-r1
  llama3.2-1b-mem0-one-pass-multitask-noop5-e4-b8-trainseed45-r1
  qwen35-2b-mem0-one-pass-multitask-noop5-e4-b4-trainseed45-r1
  llama3.2-3b-mem0-one-pass-multitask-noop5-e4-b4-trainseed45-r1
)
epochs=(03 03 02 03 03 03)
slugs=(granite4-350m qwen35-0.8b granite4-1b llama3.2-1b qwen35-2b llama3.2-3b)

cd ${repo}
for index in "${!models[@]}"; do
  model=${models[$index]}
  if [[ ${only_model} != all && ${model} != "${only_model}" ]]; then
    continue
  fi
  run=${runs[$index]}
  epoch=${epochs[$index]}
  slug=${slugs[$index]}
  checkpoint=${workspace}/runs/${run}/checkpoints/epoch-${epoch}
  output=${campaign}/${slug}/cache/mem0_one_pass/on
  if [[ -s ${output}/summary.json && -s ${output}/turns.jsonl ]]; then
    echo "[$(date -Is)] Reusing completed ${model} one-pass benchmark."
    continue
  fi
  echo "[$(date -Is)] Starting ${model} one-pass benchmark."
  ${python} -m memory_training.benchmark_hf_prefix_cache \
    --model ${model} --method mem0_one_pass --checkpoint ${checkpoint} \
    --turn-manifest ${manifest} --output-dir ${output} \
    --workspace ${workspace} --data-root ${data_root} \
    --catalog-path ${data_root}/catalog.sqlite \
    --cache-mode on --replay-mode controlled \
    --max-length 2048 --max-new-tokens 768 \
    --warmup-turns 3 --repetitions 1
done

echo "[$(date -Is)] All Mem0 one-pass efficiency benchmarks completed."
