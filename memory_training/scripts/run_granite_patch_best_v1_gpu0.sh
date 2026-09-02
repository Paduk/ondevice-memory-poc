#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/hj153lee/PalmClaw"
runs_root="/mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs"
python_bin="/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python"
bench_root="/home/hj153lee/VehicleMemBench"

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

run_test() {
  local model_key="$1"
  local run_id="$2"
  local epoch="$3"
  local run_root="${runs_root}/${run_id}"
  local checkpoint="${run_root}/checkpoints/epoch-${epoch}"
  local output_dir="${run_root}/test-hf-v1-s1-s50-best-esm-epoch${epoch}-b16"

  if [[ ! -d "${checkpoint}/adapter" ]]; then
    echo "[$(date -Is)] Missing checkpoint adapter: ${checkpoint}/adapter"
    return 1
  fi

  echo "[$(date -Is)] Starting ${model_key} V1 S1-S50 from epoch ${epoch}."
  cd "${repo_root}"
  "${python_bin}" -m memory_training.evaluate_hf_v1 \
    --model "${model_key}" \
    --method patch \
    --checkpoint "${checkpoint}" \
    --output-dir "${output_dir}" \
    --vehiclemembench-root "${bench_root}" \
    --scenarios {1..50} \
    --max-length 2048 \
    --max-new-tokens 768 \
    --quiz-max-new-tokens 256 \
    --scenario-batch-size 16 \
    --quiz-batch-size 32 \
    --checkpoint-interval 50
  echo "[$(date -Is)] Completed ${model_key} V1 S1-S50."
}

run_test \
  granite4-350m \
  granite4-350m-patch-multitask-noop5-full6val-grouped-v2-v1-10-e4-b8-trainseed45-r1 \
  04

run_test \
  granite4-1b \
  granite4-1b-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1 \
  03
