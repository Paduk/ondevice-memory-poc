#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"
summary_checkpoint="${2:?Summary checkpoint is required}"
patch_checkpoint="${3:?Patch checkpoint is required}"
delta_append_checkpoint="${4:?Delta-v3 append checkpoint is required}"
run_tag="${5:-v1}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-v2-v1-10-eval-fixed-noop5-seed45-v1
manifest="${workspace}/benchmarks/kv-cache-validation-noop5-s81-s85-t11-v1/fixed-data-turn-manifest.json"
output_root="${workspace}/benchmarks/kv-cache-granite4-1b-three-methods-${run_tag}"

for checkpoint in \
  "${summary_checkpoint}" \
  "${patch_checkpoint}" \
  "${delta_append_checkpoint}"
do
  if [[ ! -d "${checkpoint}/adapter" ]]; then
    echo "Checkpoint adapter not found: ${checkpoint}" >&2
    exit 2
  fi
done

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
  local replay=$3
  local variant=$4
  local output_dir="${output_root}/${replay}/${method}/${variant}"
  local cache_mode=off
  local -a extra_args=()
  if [[ "${variant}" != off ]]; then
    cache_mode=on
  fi
  if [[ "${variant}" == on-background ]]; then
    extra_args+=(--background-prefill)
  fi
  if [[ "${method}" == delta_v3_append ]]; then
    # The fixed 5:1 workload intentionally omits source turns. Performance-only
    # replay treats the retained rows as consecutive requests; quality evaluation
    # must continue to use the full trajectory instead.
    extra_args+=(--append-allow-turn-gaps --delta-v3-append-max-turns 32)
  fi
  if [[ "${cache_mode}" == on ]]; then
    extra_args+=(
      --reference-turns
      "${output_root}/${replay}/${method}/off/turns.jsonl"
    )
  fi
  if [[ -s "${output_dir}/summary.json" ]]; then
    echo "[$(date -Is)] Reusing ${replay}/${method}/${variant}."
    return
  fi
  echo "[$(date -Is)] Running ${replay}/${method}/${variant} on GPU ${gpu}."
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
    --replay-mode "${replay}" \
    --max-length 4096 \
    --max-new-tokens 768 \
    --warmup-turns 3 \
    --repetitions 1 \
    "${extra_args[@]}"
}

for replay in controlled predicted; do
  for method in summary patch delta_v3_append; do
    case "${method}" in
      summary) checkpoint=${summary_checkpoint} ;;
      patch) checkpoint=${patch_checkpoint} ;;
      delta_v3_append) checkpoint=${delta_append_checkpoint} ;;
    esac
    run_benchmark "${method}" "${checkpoint}" "${replay}" off
    run_benchmark "${method}" "${checkpoint}" "${replay}" on
    run_benchmark "${method}" "${checkpoint}" "${replay}" on-background
  done

  comparison_dir="${output_root}/${replay}/comparison"
  comparison_inputs=()
  for method in summary patch delta_v3_append; do
    for variant in off on on-background; do
      comparison_inputs+=(
        --input "${output_root}/${replay}/${method}/${variant}/summary.json"
      )
    done
  done
  "${python_bin}" -m memory_training.compare_hf_prefix_cache \
    "${comparison_inputs[@]}" \
    --output-dir "${comparison_dir}" \
    --projected-prefill-tokens-per-second 30 \
    --force
done

echo "[$(date -Is)] Completed: ${output_root}"

