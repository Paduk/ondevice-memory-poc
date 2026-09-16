#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU MODEL_KEY [MODEL_KEY ...]" >&2
  exit 2
fi

gpu=$1
shift
repo=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2
output_root=${workspace}/evaluations/quiz-only-e3-baselines

export CUDA_VISIBLE_DEVICES=${gpu}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${repo}:${repo}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

run_id_for() {
  case "$1" in
    llama3.2-3b) echo llama3.2-3b-quiz-only-e4-qpasses2-qb4-trainseed45-r1 ;;
    qwen3.5-2b) echo qwen35-2b-quiz-only-e4-qpasses2-qb4-trainseed45-r1 ;;
    granite4-1b) echo granite4-1b-quiz-only-e4-qpasses2-qb4-trainseed45-r1 ;;
    granite4-350m) echo granite4-350m-quiz-only-e4-qpasses2-qb8-trainseed45-r1 ;;
    qwen3.5-0.8b) echo qwen35-0.8b-quiz-only-e4-qpasses2-qb8-trainseed45-r1 ;;
    llama3.2-1b) echo llama3.2-1b-quiz-only-e4-qpasses2-qb8-trainseed45-r1 ;;
    *) echo "unknown model: $1" >&2; return 2 ;;
  esac
}

mkdir -p "${output_root}/logs"
cd "${repo}"
for model in "$@"; do
  run_id=$(run_id_for "${model}")
  checkpoint=${workspace}/runs/${run_id}/checkpoints/epoch-03
  output=${output_root}/${model}/no_memory
  log=${output_root}/logs/${model}-no_memory-gpu${gpu}.log
  echo "$(date -u +%FT%TZ) START ${model} gpu=${gpu}" | tee -a "${log}"
  "${python}" -m memory_training.evaluate_quiz_baseline \
    --model "${model}" \
    --checkpoint "${checkpoint}" \
    --profile no_memory \
    --data-root "${data}" \
    --output-dir "${output}" \
    --workspace "${workspace}" \
    --max-length 2048 \
    --max-new-tokens 256 \
    --quiz-batch-size 16 \
    >>"${log}" 2>&1
  echo "$(date -u +%FT%TZ) COMPLETE ${model} gpu=${gpu}" | tee -a "${log}"
done
