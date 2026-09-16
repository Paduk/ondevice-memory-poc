#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"
interval="${2:?Compaction interval must be 2, 5, or 10}"
run_tag="${3:-20260905-v1}"

case "${interval}" in
  2|5|10) ;;
  *)
    echo "Compaction interval must be 2, 5, or 10: ${interval}" >&2
    exit 2
    ;;
esac

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
data_parent=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
method="delta_v3_compact_k${interval}"
run_id="qwen35-2b-${method}-multitask-noop5-uniform-depth-e4-b2-trainseed45-evalfixed-noop5-r1"
run_dir="${workspace}/runs/${run_id}"
output_root="${workspace}/benchmarks/qwen35-2b-stress-test-once-${run_tag}"
cache_root="${output_root}/cache"
composite_root="${output_root}/composite"

status_path="${run_dir}/status.json"
selection_path="${run_dir}/eval-fixed-best-checkpoint.json"
if [[ ! -s "${status_path}" ]] || [[ "$(jq -r '.state // .final_state' "${status_path}")" != COMPLETED ]]; then
  echo "Training/fixed-Test pipeline is not complete: ${run_id}" >&2
  exit 2
fi
if [[ ! -s "${selection_path}" ]]; then
  echo "Best-checkpoint selection not found: ${selection_path}" >&2
  exit 2
fi

checkpoint=$(jq -er '.winner.checkpoint' "${selection_path}")
test_summary=$(jq -er '.external_test' "${status_path}")
if [[ ! -d "${checkpoint}/adapter" ]]; then
  echo "Checkpoint adapter not found: ${checkpoint}" >&2
  exit 2
fi
if [[ ! -s "${test_summary}" ]] || ! jq -e \
  '.complete == true and (.completed_scenarios | length) == 24 and (.expected_scenarios | length) == 24' \
  "${test_summary}" >/dev/null; then
  echo "Completed 24/24 fixed Test not found: ${test_summary}" >&2
  exit 2
fi

for update_count in 20 40 60 80; do
  data_root="${data_parent}/grouped-v2-v1-10-eval-stress-s86-s90-update${update_count}-front-middle-v2-delta-v3-compact-k${interval}-v1"
  for required in COMPLETED catalog.sqlite turn_manifest.json; do
    if [[ ! -f "${data_root}/${required}" ]]; then
      echo "Stress data is incomplete: ${data_root}/${required}" >&2
      exit 2
    fi
  done
done

mkdir -p "${cache_root}" "${composite_root}"
cd "${repo_root}"

export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$(date -Is)] starting Qwen3.5 2B stress Test GPU=${gpu} method=${method} checkpoint=$(basename "${checkpoint}")"
for update_count in 20 40 60 80; do
  data_root="${data_parent}/grouped-v2-v1-10-eval-stress-s86-s90-update${update_count}-front-middle-v2-delta-v3-compact-k${interval}-v1"
  cache_output="${cache_root}/update${update_count}/${method}/on"
  if [[ ! -s "${cache_output}/summary.json" ]]; then
    echo "[$(date -Is)] cache start GPU=${gpu} update=${update_count} method=${method}"
    "${python_bin}" -m memory_training.benchmark_hf_prefix_cache \
      --model qwen3.5-2b \
      --method "${method}" \
      --checkpoint "${checkpoint}" \
      --turn-manifest "${data_root}/turn_manifest.json" \
      --output-dir "${cache_output}" \
      --workspace "${workspace}" \
      --data-root "${data_root}" \
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

  composite_output="${composite_root}/update${update_count}/${method}"
  if [[ ! -s "${composite_output}/summary.json" ]] || \
     [[ "$(jq -r '.complete // false' "${composite_output}/summary.json")" != true ]]; then
    echo "[$(date -Is)] composite start GPU=${gpu} update=${update_count} method=${method}"
    "${python_bin}" -m memory_training.evaluate_hf_closed_loop \
      --model qwen3.5-2b \
      --method "${method}" \
      --checkpoint "${checkpoint}" \
      --output-dir "${composite_output}" \
      --workspace "${workspace}" \
      --data-root "${data_root}" \
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

echo "[$(date -Is)] completed Qwen3.5 2B stress Test method=${method}"
