#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 GPU [MODEL_KEY ...]" >&2
  echo "If MODEL_KEY is omitted, all six models run sequentially." >&2
  exit 2
fi

gpu=$1
shift

repo=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2
output_root=${workspace}/evaluations/quiz-only-e3-baselines

all_models=(
  granite4-350m
  qwen3.5-0.8b
  granite4-1b
  llama3.2-1b
  qwen3.5-2b
  llama3.2-3b
)
if [[ $# -eq 0 ]]; then
  models=("${all_models[@]}")
else
  models=("$@")
fi

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

for model in "${models[@]}"; do
  run_id_for "${model}" >/dev/null
done

mkdir -p "${output_root}/logs"
exec 9>"${output_root}/gold-memory-oracle-gpu${gpu}.lock"
if ! flock -n 9; then
  echo "Another Gold-memory Oracle queue already owns GPU ${gpu}." >&2
  exit 3
fi

memory_used=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
pids=$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
if [[ -n "${pids}" || ! "${memory_used}" =~ ^[0-9]+$ || "${memory_used}" -gt 100 ]]; then
  echo "GPU ${gpu} is not empty (memory=${memory_used} MiB, pids=${pids:-none})." >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES=${gpu}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${repo}:${repo}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${repo}"
for model in "${models[@]}"; do
  run_id=$(run_id_for "${model}")
  checkpoint=${workspace}/runs/${run_id}/checkpoints/epoch-03
  output=${output_root}/${model}/gold_memory
  log=${output_root}/logs/${model}-gold_memory-gpu${gpu}.log

  [[ -d "${checkpoint}/adapter" ]] || {
    echo "Checkpoint adapter not found: ${checkpoint}" >&2
    exit 5
  }

  echo "$(date -u +%FT%TZ) START ${model} gold_memory gpu=${gpu}" | tee -a "${log}"
  "${python}" -m memory_training.evaluate_quiz_baseline \
    --model "${model}" \
    --checkpoint "${checkpoint}" \
    --profile gold_memory \
    --data-root "${data}" \
    --output-dir "${output}" \
    --workspace "${workspace}" \
    --max-length 2048 \
    --max-new-tokens 256 \
    --quiz-batch-size 16 \
    >>"${log}" 2>&1
  echo "$(date -u +%FT%TZ) COMPLETE ${model} gold_memory gpu=${gpu}" | tee -a "${log}"
done
