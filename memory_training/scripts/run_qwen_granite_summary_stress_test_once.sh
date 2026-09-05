#!/usr/bin/env bash
set -euo pipefail

run_tag="${1:-20260904-v1}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
data_parent=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python

qwen_checkpoint="${workspace}/runs/qwen35-0.8b-summary-multitask-noop5-grouped-v2-v1-10-e4-b4-trainseed45-evalfixed-noop5-r2/checkpoints/epoch-03"
granite_checkpoint="${workspace}/runs/granite4-1b-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-03"

for checkpoint in "$qwen_checkpoint" "$granite_checkpoint"; do
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

run_model() {
  local gpu=$1
  local model=$2
  local checkpoint=$3
  local output_root=$4
  local update_count data_root cache_output composite_output

  export CUDA_VISIBLE_DEVICES="$gpu"
  mkdir -p "${output_root}/logs"
  for update_count in 20 40 60 80; do
    data_root="${data_parent}/grouped-v2-v1-10-eval-stress-s86-s90-update${update_count}-front-middle-v2"
    cache_output="${output_root}/cache/update${update_count}/summary/on"
    if [[ ! -s "${cache_output}/summary.json" ]]; then
      echo "[$(date -Is)] cache start GPU=${gpu} model=${model} update=${update_count} method=summary"
      "$python_bin" -m memory_training.benchmark_hf_prefix_cache \
        --model "$model" \
        --method summary \
        --checkpoint "$checkpoint" \
        --turn-manifest "${data_root}/turn_manifest.json" \
        --output-dir "$cache_output" \
        --workspace "$workspace" \
        --data-root "$data_root" \
        --catalog-path "${data_root}/catalog.sqlite" \
        --cache-mode on \
        --replay-mode controlled \
        --max-length 4096 \
        --max-new-tokens 768 \
        --warmup-turns 3 \
        --repetitions 1
      echo "[$(date -Is)] cache done GPU=${gpu} model=${model} update=${update_count} method=summary"
    else
      echo "[$(date -Is)] cache reuse model=${model} update=${update_count} method=summary"
    fi

    composite_output="${output_root}/composite/update${update_count}/summary"
    if [[ ! -s "${composite_output}/summary.json" ]] || \
       [[ "$(jq -r '.complete' "${composite_output}/summary.json")" != true ]]; then
      echo "[$(date -Is)] composite start GPU=${gpu} model=${model} update=${update_count} method=summary"
      "$python_bin" -m memory_training.evaluate_hf_closed_loop \
        --model "$model" \
        --method summary \
        --checkpoint "$checkpoint" \
        --output-dir "$composite_output" \
        --workspace "$workspace" \
        --data-root "$data_root" \
        --catalog-path "${data_root}/catalog.sqlite" \
        --scenarios 86 87 88 89 90 \
        --max-length 4096 \
        --max-new-tokens 768 \
        --quiz-max-new-tokens 256 \
        --scenario-batch-size 5 \
        --quiz-batch-size 16
      echo "[$(date -Is)] composite done GPU=${gpu} model=${model} update=${update_count} method=summary"
    else
      echo "[$(date -Is)] composite reuse model=${model} update=${update_count} method=summary"
    fi
  done
}

qwen_root="${workspace}/benchmarks/qwen35-0.8b-stress-test-once-${run_tag}"
granite_root="${workspace}/benchmarks/granite4-1b-stress-test-once-${run_tag}"
mkdir -p "${qwen_root}/logs" "${granite_root}/logs"

run_model 2 qwen3.5-0.8b "$qwen_checkpoint" "$qwen_root" >"${qwen_root}/logs/summary.log" 2>&1 &
qwen_pid=$!
run_model 3 granite4-1b "$granite_checkpoint" "$granite_root" >"${granite_root}/logs/summary.log" 2>&1 &
granite_pid=$!

status=0
wait "$qwen_pid" || status=1
wait "$granite_pid" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "One or more Summary stress workers failed." >&2
  exit "$status"
fi

echo "[$(date -Is)] completed Qwen and Granite Summary stress Test."
