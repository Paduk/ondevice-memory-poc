#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
data_parent=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
output_root="${workspace}/benchmarks/llama32-1b-stress-composite-epoch03-20260908-v1"
log_root="${output_root}/logs"

methods=(delta_v3_compact_k5 delta_v3_compact_k10)

resolve_run_dir() {
  case "$1" in
    delta_v3_compact_k5)
      echo "${workspace}/runs/llama3.2-1b-delta_v3_compact_k5-multitask-noop5-e4-b2-trainseed45-evalfixed-noop5-r2"
      ;;
    delta_v3_compact_k10)
      echo "${workspace}/runs/llama3.2-1b-delta_v3_compact_k10-multitask-noop5-e4-b2-trainseed45-evalfixed-noop5-r1"
      ;;
    *)
      echo "Unsupported method: $1" >&2
      return 2
      ;;
  esac
}

data_root_for() {
  local method=$1 load=$2 root
  if [[ "${load}" == base ]]; then
    root="${data_parent}/grouped-v2-v1-10-eval-fixed-noop5-seed45-v1"
  else
    root="${data_parent}/grouped-v2-v1-10-eval-stress-s86-s90-update${load}-front-middle-v2"
  fi
  echo "${root}-delta-v3-compact-k${method##*k}-v1"
}

mkdir -p "${output_root}/composite" "${log_root}"
exec > >(tee -a "${log_root}/gpu${gpu}.log") 2>&1

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HOME="${workspace}/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

for method in "${methods[@]}"; do
  run_dir=$(resolve_run_dir "${method}")
  checkpoint="${run_dir}/checkpoints/epoch-03"
  if [[ ! -d "${checkpoint}/adapter" ]]; then
    echo "Epoch 3 checkpoint adapter not found: ${checkpoint}" >&2
    exit 2
  fi

  for load in base 20 40 60 80; do
    data_root=$(data_root_for "${method}" "${load}")
    if [[ ! -f "${data_root}/catalog.sqlite" ]]; then
      echo "Evaluation catalog is missing: ${data_root}/catalog.sqlite" >&2
      exit 2
    fi

    composite_output="${output_root}/composite/${load}/${method}"
    if [[ -s "${composite_output}/summary.json" ]] &&
       [[ $(jq -r '.complete // false' "${composite_output}/summary.json") == true ]]; then
      echo "[$(date -Is)] composite reuse GPU=${gpu} load=${load} method=${method} checkpoint=epoch-03"
      continue
    fi

    echo "[$(date -Is)] composite start GPU=${gpu} load=${load} method=${method} checkpoint=epoch-03"
    "${python_bin}" -m memory_training.evaluate_hf_closed_loop \
      --model llama3.2-1b \
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
    echo "[$(date -Is)] composite done GPU=${gpu} load=${load} method=${method} checkpoint=epoch-03"
  done
done

echo "[$(date -Is)] completed Llama 3.2 1B epoch-03 stress Composite-only evaluation"
