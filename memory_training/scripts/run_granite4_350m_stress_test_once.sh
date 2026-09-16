#!/usr/bin/env bash
set -euo pipefail

run_tag="${1:-20260905-v1}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
data_parent=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
output_root="${workspace}/benchmarks/granite4-350m-stress-test-once-${run_tag}"

patch_checkpoint="${workspace}/runs/granite4-350m-patch-multitask-noop5-full6val-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-04"
summary_checkpoint="${workspace}/runs/granite4-350m-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-04"
k2_checkpoint="${workspace}/runs/granite4-350m-delta_v3_compact_k2-multitask-noop5-uniform-depth-e4-b8-trainseed45-r1/checkpoints/epoch-03"
k5_checkpoint="${workspace}/runs/granite4-350m-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b8-trainseed45-r1/checkpoints/epoch-03"
k10_checkpoint="${workspace}/runs/granite4-350m-delta_v3_compact_k10-multitask-noop5-uniform-depth-e4-b8-trainseed45-r1/checkpoints/epoch-04"

for checkpoint in "$patch_checkpoint" "$summary_checkpoint" "$k2_checkpoint" "$k5_checkpoint" "$k10_checkpoint"; do
  if [[ ! -d "${checkpoint}/adapter" ]]; then
    echo "Checkpoint adapter not found: ${checkpoint}" >&2
    exit 2
  fi
done

mkdir -p "${output_root}/logs"
cd "$repo_root"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

run_method() {
  local gpu=$1
  local method=$2
  local checkpoint=$3
  local update_count data_root cache_output composite_output

  export CUDA_VISIBLE_DEVICES="$gpu"
  for update_count in 20 40 60 80; do
    data_root="${data_parent}/grouped-v2-v1-10-eval-stress-s86-s90-update${update_count}-front-middle-v2"
    if [[ "$method" == delta_v3_compact_k* ]]; then
      data_root="${data_root}-delta-v3-compact-k${method##*k}-v1"
    fi

    cache_output="${output_root}/cache/update${update_count}/${method}/on"
    if [[ ! -s "${cache_output}/summary.json" ]]; then
      echo "[$(date -Is)] cache start GPU=${gpu} update=${update_count} method=${method}"
      "$python_bin" -m memory_training.benchmark_hf_prefix_cache \
        --model granite4-350m \
        --method "$method" \
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
      echo "[$(date -Is)] cache done GPU=${gpu} update=${update_count} method=${method}"
    else
      echo "[$(date -Is)] cache reuse update=${update_count} method=${method}"
    fi

    composite_output="${output_root}/composite/update${update_count}/${method}"
    if [[ ! -s "${composite_output}/summary.json" ]] || \
       [[ "$(jq -r '.complete' "${composite_output}/summary.json")" != true ]]; then
      echo "[$(date -Is)] composite start GPU=${gpu} update=${update_count} method=${method}"
      "$python_bin" -m memory_training.evaluate_hf_closed_loop \
        --model granite4-350m \
        --method "$method" \
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
      echo "[$(date -Is)] composite done GPU=${gpu} update=${update_count} method=${method}"
    else
      echo "[$(date -Is)] composite reuse update=${update_count} method=${method}"
    fi
  done
}

run_method 4 summary "$summary_checkpoint" >>"${output_root}/logs/summary.log" 2>&1 &
summary_pid=$!

(
  run_method 6 patch "$patch_checkpoint"
  run_method 6 delta_v3_compact_k2 "$k2_checkpoint"
  run_method 6 delta_v3_compact_k5 "$k5_checkpoint"
  run_method 6 delta_v3_compact_k10 "$k10_checkpoint"
) >>"${output_root}/logs/patch-delta.log" 2>&1 &
other_pid=$!

status=0
wait "$summary_pid" || status=1
wait "$other_pid" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "One or more Granite 4 350M stress workers failed." >&2
  exit "$status"
fi

echo "[$(date -Is)] completed Granite 4 350M stress Test."
