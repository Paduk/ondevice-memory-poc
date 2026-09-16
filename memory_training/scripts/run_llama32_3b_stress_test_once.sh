#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"
method="${2:?Method is required}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
data_parent=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
output_root="${workspace}/benchmarks/llama32-3b-stress-test-once-20260908-v1"
natural_manifest_root="${workspace}/benchmarks/natural-s86-s90-manifests-20260905-v1"
log_root="${output_root}/logs"

case "${method}" in
  summary)
    run_dir="${workspace}/runs/llama3.2-3b-summary-multitask-noop5-e4-b2-trainseed45-evalfixed-noop5-r1"
    ;;
  patch)
    run_dir="${workspace}/runs/llama3.2-3b-patch-multitask-noop5-e4-b4-trainseed45-evalfixed-noop5-r1"
    ;;
  delta_v3_compact_k2|delta_v3_compact_k5|delta_v3_compact_k10)
    run_dir="${workspace}/runs/llama3.2-3b-${method}-multitask-noop5-e4-b1-trainseed45-evalfixed-noop5-r1"
    ;;
  *)
    echo "Unsupported method: ${method}" >&2
    exit 2
    ;;
esac

data_root_for() {
  local load=$1 root
  if [[ "${load}" == base ]]; then
    root="${data_parent}/grouped-v2-v1-10-eval-fixed-noop5-seed45-v1"
  else
    root="${data_parent}/grouped-v2-v1-10-eval-stress-s86-s90-update${load}-front-middle-v2"
  fi
  if [[ "${method}" == delta_v3_compact_k* ]]; then
    root="${root}-delta-v3-compact-k${method##*k}-v1"
  fi
  echo "${root}"
}

turn_manifest_for() {
  local load=$1 data_root=$2
  if [[ "${load}" != base ]]; then
    echo "${data_root}/turn_manifest.json"
  elif [[ "${method}" == patch || "${method}" == summary ]]; then
    echo "${natural_manifest_root}/base.json"
  else
    echo "${natural_manifest_root}/k${method##*k}.json"
  fi
}

status_path="${run_dir}/status.json"
selection_path="${run_dir}/eval-fixed-best-checkpoint.json"
if [[ ! -s "${status_path}" || ! -s "${selection_path}" ]]; then
  echo "Completed 3B pipeline metadata is missing: ${run_dir}" >&2
  exit 2
fi
checkpoint=$(jq -er '.winner.checkpoint' "${selection_path}")
test_summary=$(jq -er '.external_test' "${status_path}")
if [[ ! -d "${checkpoint}/adapter" ]]; then
  echo "Selected checkpoint adapter not found: ${checkpoint}" >&2
  exit 2
fi
if [[ ! -s "${test_summary}" ]] || ! jq -e \
  '.complete == true and (.completed_scenarios | length) == 24 and (.expected_scenarios | length) == 24' \
  "${test_summary}" >/dev/null; then
  echo "Completed 24/24 fixed Test not found: ${test_summary}" >&2
  exit 2
fi

mkdir -p "${output_root}/cache" "${output_root}/composite" "${log_root}"
exec > >(tee -a "${log_root}/gpu${gpu}-${method}.log") 2>&1

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HOME="${workspace}/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$(date -Is)] starting Llama 3.2 3B stress Test GPU=${gpu} method=${method} checkpoint=$(basename "${checkpoint}")"
for load in base 20 40 60 80; do
  data_root=$(data_root_for "${load}")
  turn_manifest=$(turn_manifest_for "${load}" "${data_root}")
  if [[ ! -f "${data_root}/catalog.sqlite" || ! -f "${turn_manifest}" ]]; then
    echo "Stress data or manifest is incomplete: ${data_root} ${turn_manifest}" >&2
    exit 2
  fi

  cache_output="${output_root}/cache/${load}/${method}/on"
  if [[ ! -s "${cache_output}/summary.json" ]]; then
    echo "[$(date -Is)] cache start GPU=${gpu} load=${load} method=${method}"
    "${python_bin}" -m memory_training.benchmark_hf_prefix_cache \
      --model llama3.2-3b \
      --method "${method}" \
      --checkpoint "${checkpoint}" \
      --turn-manifest "${turn_manifest}" \
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
    echo "[$(date -Is)] cache done GPU=${gpu} load=${load} method=${method}"
  else
    echo "[$(date -Is)] cache reuse load=${load} method=${method}"
  fi

  composite_output="${output_root}/composite/${load}/${method}"
  if [[ ! -s "${composite_output}/summary.json" ]] || \
     [[ $(jq -r '.complete // false' "${composite_output}/summary.json") != true ]]; then
    echo "[$(date -Is)] composite start GPU=${gpu} load=${load} method=${method}"
    "${python_bin}" -m memory_training.evaluate_hf_closed_loop \
      --model llama3.2-3b \
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
      --scenario-batch-size 2 \
      --quiz-batch-size 8
    echo "[$(date -Is)] composite done GPU=${gpu} load=${load} method=${method}"
  else
    echo "[$(date -Is)] composite reuse load=${load} method=${method}"
  fi
done

echo "[$(date -Is)] completed Llama 3.2 3B stress Test method=${method}"
