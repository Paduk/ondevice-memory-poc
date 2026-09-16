#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"
run_tag="${2:-20260907-v1}"
target="${3:-both}"
method_filter="${4:-all}"

if [[ "$target" != both && "$target" != granite && "$target" != qwen ]]; then
  echo "Target must be one of: both, granite, qwen" >&2
  exit 2
fi

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
data_parent=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python

base_data="${data_parent}/grouped-v2-v1-10-eval-fixed-noop5-seed45-v1"
k2_data="${base_data}-delta-v3-compact-k2-v1"
k5_data="${base_data}-delta-v3-compact-k5-v1"
k10_data="${base_data}-delta-v3-compact-k10-v1"
manifest_root="${workspace}/benchmarks/natural-s86-s90-manifests-20260905-v1"

granite_output="${workspace}/benchmarks/granite4-350m-natural-s86-s90-cache-once-${run_tag}"
qwen_output="${workspace}/benchmarks/qwen35-2b-natural-s86-s90-cache-once-${run_tag}"

granite_patch="${workspace}/runs/granite4-350m-patch-multitask-noop5-full6val-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-04"
granite_summary="${workspace}/runs/granite4-350m-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-04"
granite_k2="${workspace}/runs/granite4-350m-delta_v3_compact_k2-multitask-noop5-uniform-depth-e4-b8-trainseed45-r1/checkpoints/epoch-03"
granite_k5="${workspace}/runs/granite4-350m-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b8-trainseed45-r1/checkpoints/epoch-03"
granite_k10="${workspace}/runs/granite4-350m-delta_v3_compact_k10-multitask-noop5-uniform-depth-e4-b8-trainseed45-r1/checkpoints/epoch-04"

qwen_patch="${workspace}/runs/qwen35-2b-patch-multitask-noop5-grouped-v2-v1-10-e5-b4-r1/checkpoints/epoch-03"
qwen_summary="${workspace}/runs/qwen35-2b-summary-multitask-noop5-grouped-v2-v1-10-e4-b2-r1/checkpoints/epoch-04"
qwen_k2="${workspace}/runs/qwen35-2b-delta_v3_compact_k2-multitask-noop5-uniform-depth-e4-b2-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03"
qwen_k5="${workspace}/runs/qwen35-2b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b2-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-04"
qwen_k10="${workspace}/runs/qwen35-2b-delta_v3_compact_k10-multitask-noop5-uniform-depth-e4-b2-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-04"

for checkpoint in \
  "$granite_patch" "$granite_summary" "$granite_k2" "$granite_k5" "$granite_k10" \
  "$qwen_patch" "$qwen_summary" "$qwen_k2" "$qwen_k5" "$qwen_k10"; do
  if [[ ! -d "${checkpoint}/adapter" ]]; then
    echo "Checkpoint adapter not found: ${checkpoint}" >&2
    exit 2
  fi
done

for manifest in base k2 k5 k10; do
  if [[ ! -s "${manifest_root}/${manifest}.json" ]]; then
    echo "Natural S86-S90 manifest not found: ${manifest_root}/${manifest}.json" >&2
    exit 2
  fi
done

cd "$repo_root"
export CUDA_VISIBLE_DEVICES="$gpu"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

run_model() {
  local model=$1
  local output_root=$2
  shift 2
  local method checkpoint data_root manifest output

  mkdir -p "${output_root}/logs"
  while [[ "$#" -gt 0 ]]; do
    method=$1
    checkpoint=$2
    data_root=$3
    manifest=$4
    shift 4
    if [[ "$method_filter" != all && ",${method_filter}," != *",${method},"* ]]; then
      continue
    fi
    output="${output_root}/cache/${method}/on"
    if [[ -s "${output}/summary.json" ]]; then
      echo "[$(date -Is)] reuse GPU=${gpu} model=${model} method=${method}"
      continue
    fi
    echo "[$(date -Is)] start GPU=${gpu} model=${model} method=${method}"
    "$python_bin" -m memory_training.benchmark_hf_prefix_cache \
      --model "$model" \
      --method "$method" \
      --checkpoint "$checkpoint" \
      --turn-manifest "$manifest" \
      --output-dir "$output" \
      --workspace "$workspace" \
      --data-root "$data_root" \
      --catalog-path "${data_root}/catalog.sqlite" \
      --cache-mode on \
      --replay-mode controlled \
      --max-length 4096 \
      --max-new-tokens 768 \
      --warmup-turns 3 \
      --repetitions 1
    echo "[$(date -Is)] done GPU=${gpu} model=${model} method=${method}"
  done
}

mkdir -p "${granite_output}/logs" "${qwen_output}/logs"

if [[ "$target" == both || "$target" == granite ]]; then
  run_model granite4-350m "$granite_output" \
    patch "$granite_patch" "$base_data" "${manifest_root}/base.json" \
    summary "$granite_summary" "$base_data" "${manifest_root}/base.json" \
    delta_v3_compact_k2 "$granite_k2" "$k2_data" "${manifest_root}/k2.json" \
    delta_v3_compact_k5 "$granite_k5" "$k5_data" "${manifest_root}/k5.json" \
    delta_v3_compact_k10 "$granite_k10" "$k10_data" "${manifest_root}/k10.json" \
    2>&1 | tee "${granite_output}/logs/cache.log"
fi

if [[ "$target" == both || "$target" == qwen ]]; then
  run_model qwen3.5-2b "$qwen_output" \
    patch "$qwen_patch" "$base_data" "${manifest_root}/base.json" \
    summary "$qwen_summary" "$base_data" "${manifest_root}/base.json" \
    delta_v3_compact_k2 "$qwen_k2" "$k2_data" "${manifest_root}/k2.json" \
    delta_v3_compact_k5 "$qwen_k5" "$k5_data" "${manifest_root}/k5.json" \
    delta_v3_compact_k10 "$qwen_k10" "$k10_data" "${manifest_root}/k10.json" \
    2>&1 | tee "${qwen_output}/logs/cache.log"
fi

echo "[$(date -Is)] completed Granite 350M and Qwen 2B natural S86-S90 cache benchmark."
