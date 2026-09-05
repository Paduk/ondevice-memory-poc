#!/usr/bin/env bash
set -euo pipefail

run_tag="${1:-20260905-v1}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
data_parent=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python

base_data="${data_parent}/grouped-v2-v1-10-eval-fixed-noop5-seed45-v1"
k2_data="${base_data}-delta-v3-compact-k2-v1"
k5_data="${base_data}-delta-v3-compact-k5-v1"
k10_data="${base_data}-delta-v3-compact-k10-v1"

qwen_output="${workspace}/benchmarks/qwen35-0.8b-natural-s86-s90-cache-once-${run_tag}"
granite_output="${workspace}/benchmarks/granite4-1b-natural-s86-s90-cache-once-${run_tag}"
manifest_root="${workspace}/benchmarks/natural-s86-s90-manifests-${run_tag}"

qwen_patch="${workspace}/runs/qwen35-0.8b-patch-multitask-noop5-grouped-v2-v1-10-e4-b8-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03"
qwen_summary="${workspace}/runs/qwen35-0.8b-summary-multitask-noop5-grouped-v2-v1-10-e4-b4-trainseed45-evalfixed-noop5-r2/checkpoints/epoch-03"
qwen_k2="${workspace}/runs/qwen35-0.8b-delta_v3_compact_k2-multitask-noop5-uniform-depth-e4-b4-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03"
qwen_k5="${workspace}/runs/qwen35-0.8b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b4-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03"
qwen_k10="${workspace}/runs/qwen35-0.8b-delta_v3_compact_k10-multitask-noop5-uniform-depth-e4-b4-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03"

granite_patch="${workspace}/runs/granite4-1b-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-03"
granite_summary="${workspace}/runs/granite4-1b-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-03"
granite_k2="${workspace}/runs/granite4-1b-delta_v3_compact_k2-multitask-noop5-uniform-depth-e4-b2-trainseed45-r1/checkpoints/epoch-03"
granite_k5="${workspace}/runs/granite4-1b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b2-trainseed45-r1/checkpoints/epoch-04"
granite_k10="${workspace}/runs/granite4-1b-delta_v3_compact_k10-multitask-noop5-uniform-depth-e4-b2-trainseed45-r1/checkpoints/epoch-04"

for checkpoint in \
  "$qwen_patch" "$qwen_summary" "$qwen_k2" "$qwen_k5" "$qwen_k10" \
  "$granite_patch" "$granite_summary" "$granite_k2" "$granite_k5" "$granite_k10"; do
  if [[ ! -d "${checkpoint}/adapter" ]]; then
    echo "Checkpoint adapter not found: ${checkpoint}" >&2
    exit 2
  fi
done

cd "$repo_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "$manifest_root" "${qwen_output}/logs" "${granite_output}/logs"

prepare_manifest() {
  local name=$1
  local data_root=$2
  local output="${manifest_root}/${name}.json"
  if [[ ! -s "$output" ]]; then
    "$python_bin" -m memory_training.cache_benchmark_manifest \
      --output "$output" \
      --workspace "$workspace" \
      --data-root "$data_root" \
      --catalog-path "${data_root}/catalog.sqlite" \
      --scenarios 86 87 88 89 90 \
      --split validation \
      --noop-per-update 5 \
      --sampling-seed 45
  fi
}

prepare_manifest base "$base_data"
prepare_manifest k2 "$k2_data"
prepare_manifest k5 "$k5_data"
prepare_manifest k10 "$k10_data"

run_model() {
  local gpu=$1
  local model=$2
  local output_root=$3
  shift 3
  local method checkpoint data_root manifest output

  export CUDA_VISIBLE_DEVICES="$gpu"
  while [[ "$#" -gt 0 ]]; do
    method=$1
    checkpoint=$2
    data_root=$3
    manifest=$4
    shift 4
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

run_model 4 qwen3.5-0.8b "$qwen_output" \
  patch "$qwen_patch" "$base_data" "${manifest_root}/base.json" \
  summary "$qwen_summary" "$base_data" "${manifest_root}/base.json" \
  delta_v3_compact_k2 "$qwen_k2" "$k2_data" "${manifest_root}/k2.json" \
  delta_v3_compact_k5 "$qwen_k5" "$k5_data" "${manifest_root}/k5.json" \
  delta_v3_compact_k10 "$qwen_k10" "$k10_data" "${manifest_root}/k10.json" \
  >"${qwen_output}/logs/cache.log" 2>&1 &
qwen_pid=$!

run_model 6 granite4-1b "$granite_output" \
  patch "$granite_patch" "$base_data" "${manifest_root}/base.json" \
  summary "$granite_summary" "$base_data" "${manifest_root}/base.json" \
  delta_v3_compact_k2 "$granite_k2" "$k2_data" "${manifest_root}/k2.json" \
  delta_v3_compact_k5 "$granite_k5" "$k5_data" "${manifest_root}/k5.json" \
  delta_v3_compact_k10 "$granite_k10" "$k10_data" "${manifest_root}/k10.json" \
  >"${granite_output}/logs/cache.log" 2>&1 &
granite_pid=$!

status=0
wait "$qwen_pid" || status=1
wait "$granite_pid" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "One or more natural S86-S90 cache workers failed." >&2
  exit "$status"
fi

echo "[$(date -Is)] completed natural S86-S90 cache benchmark (one repetition)."
