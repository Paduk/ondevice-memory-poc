#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"
shift
if [[ $# -lt 1 ]]; then
  echo "usage: $0 GPU MODEL_KEY [MODEL_KEY ...]" >&2
  exit 2
fi

repo=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_parent=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training
analysis_root=${workspace}/evaluations/patch-vs-delta-disagreement-analysis
type_accuracy=${workspace}/evaluations/three-method-type-accuracy.json
requests=${analysis_root}/snapshot-request-manifest.json
output_root=${analysis_root}/memory-snapshots
log_root=${analysis_root}/logs

export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${repo}:${repo}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

model_label() {
  case "$1" in
    granite4-350m) echo "Granite 350M" ;;
    qwen3.5-0.8b) echo "Qwen 0.8B" ;;
    granite4-1b) echo "Granite 1B" ;;
    llama3.2-1b) echo "Llama 3.2 1B" ;;
    qwen3.5-2b) echo "Qwen 2B" ;;
    llama3.2-3b) echo "Llama 3.2 3B" ;;
    *) echo "Unsupported model: $1" >&2; return 2 ;;
  esac
}

run_one() {
  local model=$1
  local label=$2
  local method=$3
  local display data max_length artifact checkpoint output log
  if [[ ${method} == patch ]]; then
    display=Patch
    data=${data_parent}/grouped-v2-v1-10-eval-fixed-noop5-seed45-v1
    max_length=2048
  else
    display=Delta-v3
    data=${data_parent}/grouped-v2-v1-10-eval-fixed-noop5-seed45-v1-delta-v3-compact-k5-v1
    max_length=4096
  fi
  artifact=$(jq -r --arg model "${label}" --arg method "${display}" \
    '.results[$model][$method].artifact' "${type_accuracy}")
  checkpoint=$(jq -r '.checkpoint' "$(dirname "${artifact}")/manifest.json")
  output=${output_root}/${model}/${method}
  log=${log_root}/${model}-${method}-gpu${gpu}.log
  mkdir -p "${output}" "${log_root}"
  echo "$(date -u +%FT%TZ) START model=${model} method=${method} gpu=${gpu}" | tee -a "${log}"
  "${python_bin}" -m memory_training.capture_hf_memory_snapshots \
    --model "${model}" \
    --model-label "${label}" \
    --method "${method}" \
    --checkpoint "${checkpoint}" \
    --data-root "${data}" \
    --catalog-path "${data}/catalog.sqlite" \
    --request-manifest "${requests}" \
    --output-dir "${output}" \
    --workspace "${workspace}" \
    --max-length "${max_length}" \
    --max-new-tokens 768 \
    --scenario-batch-size 8 \
    >>"${log}" 2>&1
  echo "$(date -u +%FT%TZ) COMPLETE model=${model} method=${method} gpu=${gpu}" | tee -a "${log}"
}

cd "${repo}"
for model in "$@"; do
  label=$(model_label "${model}")
  run_one "${model}" "${label}" patch
  run_one "${model}" "${label}" delta_v3_compact_k5
done
