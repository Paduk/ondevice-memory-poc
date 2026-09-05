#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"
run_tag="${2:-v1}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-v2-v1-10-eval-s81-update40-front-middle-v1
manifest="${data_root}/turn_manifest.json"
output_root="${workspace}/benchmarks/kv-cache-granite4-1b-s81-update40-front-middle-patch-delta-v3-${run_tag}"

patch_checkpoint="${workspace}/runs/granite4-1b-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-03"
delta_checkpoint="${workspace}/runs/granite4-1b-delta-v3-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-03"

for checkpoint in \
  "${patch_checkpoint}" \
  "${delta_checkpoint}"
do
  if [[ ! -d "${checkpoint}/adapter" ]]; then
    echo "Checkpoint adapter not found: ${checkpoint}" >&2
    exit 2
  fi
done

if [[ ! -s "${manifest}" ]]; then
  echo "Turn manifest not found: ${manifest}" >&2
  exit 2
fi

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

run_benchmark() {
  local method=$1
  local checkpoint=$2
  local variant=$3
  local output_dir="${output_root}/controlled/${method}/${variant}"
  local cache_mode=off
  local -a extra_args=()
  if [[ "${variant}" != off ]]; then
    cache_mode=on
  fi
  if [[ "${variant}" == on-background ]]; then
    extra_args+=(--background-prefill)
  fi
  if [[ -s "${output_dir}/summary.json" ]]; then
    echo "[$(date -Is)] Reusing controlled/${method}/${variant}."
    return
  fi
  echo "[$(date -Is)] Running controlled/${method}/${variant} on GPU ${gpu}."
  "${python_bin}" -m memory_training.benchmark_hf_prefix_cache \
    --model granite4-1b \
    --method "${method}" \
    --checkpoint "${checkpoint}" \
    --turn-manifest "${manifest}" \
    --output-dir "${output_dir}" \
    --workspace "${workspace}" \
    --data-root "${data_root}" \
    --catalog-path "${data_root}/catalog.sqlite" \
    --cache-mode "${cache_mode}" \
    --replay-mode controlled \
    --max-length 4096 \
    --max-new-tokens 768 \
    --warmup-turns 3 \
    --repetitions 1 \
    "${extra_args[@]}"
}

for method in patch delta_v3; do
  case "${method}" in
    patch) checkpoint=${patch_checkpoint} ;;
    delta_v3) checkpoint=${delta_checkpoint} ;;
  esac
  run_benchmark "${method}" "${checkpoint}" off
  run_benchmark "${method}" "${checkpoint}" on
  run_benchmark "${method}" "${checkpoint}" on-background
done

comparison_dir="${output_root}/controlled/comparison"
comparison_inputs=()
for method in patch delta_v3; do
  for variant in off on on-background; do
    comparison_inputs+=(
      --input "${output_root}/controlled/${method}/${variant}/summary.json"
    )
  done
done
"${python_bin}" -m memory_training.compare_hf_prefix_cache \
  "${comparison_inputs[@]}" \
  --output-dir "${comparison_dir}" \
  --projected-prefill-tokens-per-second 30 \
  --force

echo "[$(date -Is)] Completed: ${output_root}"
