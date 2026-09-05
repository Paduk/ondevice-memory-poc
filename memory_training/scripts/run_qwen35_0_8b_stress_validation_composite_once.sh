#!/usr/bin/env bash
set -euo pipefail

run_tag="${1:-20260904-v1}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
data_parent=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
output_root="${workspace}/benchmarks/qwen35-0.8b-stress-validation-composite-once-${run_tag}"

patch_checkpoint="${workspace}/runs/qwen35-0.8b-patch-multitask-noop5-grouped-v2-v1-10-e4-b8-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03"
k2_checkpoint="${workspace}/runs/qwen35-0.8b-delta_v3_compact_k2-multitask-noop5-uniform-depth-e4-b4-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03"
k5_checkpoint="${workspace}/runs/qwen35-0.8b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b4-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03"
k10_checkpoint="${workspace}/runs/qwen35-0.8b-delta_v3_compact_k10-multitask-noop5-uniform-depth-e4-b4-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03"

for checkpoint in "$patch_checkpoint" "$k2_checkpoint" "$k5_checkpoint" "$k10_checkpoint"; do
  if [[ ! -d "${checkpoint}/adapter" ]]; then
    echo "Checkpoint adapter not found: ${checkpoint}" >&2
    exit 2
  fi
done

mkdir -p "$output_root"
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
  local update_count data_root output_dir

  export CUDA_VISIBLE_DEVICES="$gpu"
  for update_count in 20 40 60 80; do
    data_root="${data_parent}/grouped-v2-v1-10-eval-stress-s81-s85-update${update_count}-front-middle-v2"
    if [[ "$method" != patch ]]; then
      data_root="${data_root}-delta-v3-compact-k${method##*k}-v1"
    fi
    output_dir="${output_root}/update${update_count}/${method}"
    if [[ -s "${output_dir}/summary.json" ]] && [[ "$(jq -r '.complete' "${output_dir}/summary.json")" == true ]]; then
      echo "[$(date -Is)] reuse update${update_count}/${method}"
      continue
    fi
    echo "[$(date -Is)] start GPU=${gpu} update=${update_count} method=${method}"
    "$python_bin" -m memory_training.evaluate_hf_closed_loop \
      --model qwen3.5-0.8b \
      --method "$method" \
      --checkpoint "$checkpoint" \
      --output-dir "$output_dir" \
      --workspace "$workspace" \
      --data-root "$data_root" \
      --catalog-path "${data_root}/catalog.sqlite" \
      --scenarios 81 82 83 84 85 \
      --max-length 4096 \
      --max-new-tokens 768 \
      --quiz-max-new-tokens 256 \
      --scenario-batch-size 5 \
      --quiz-batch-size 16
    echo "[$(date -Is)] done GPU=${gpu} update=${update_count} method=${method}"
  done
}

run_method 0 patch "$patch_checkpoint" >"${output_root}/patch.log" 2>&1 &
patch_pid=$!
run_method 1 delta_v3_compact_k2 "$k2_checkpoint" >"${output_root}/k2.log" 2>&1 &
k2_pid=$!
run_method 2 delta_v3_compact_k5 "$k5_checkpoint" >"${output_root}/k5.log" 2>&1 &
k5_pid=$!
run_method 7 delta_v3_compact_k10 "$k10_checkpoint" >"${output_root}/k10.log" 2>&1 &
k10_pid=$!

status=0
for pid in "$patch_pid" "$k2_pid" "$k5_pid" "$k10_pid"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "One or more stress Composite workers failed: ${output_root}" >&2
  exit "$status"
fi

echo "[$(date -Is)] completed ${output_root}"
