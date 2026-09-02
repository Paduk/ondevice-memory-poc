#!/usr/bin/env bash
set -euo pipefail

run_id="${1:?run id is required}"
method="${2:?method is required}"
gpu="${3:?gpu index is required}"
epoch="${4:-04}"

repo_root="/home/hj153lee/PalmClaw"
runs_root="/mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs"
python_bin="/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python"
run_root="${runs_root}/${run_id}"
checkpoint="${run_root}/checkpoints/epoch-${epoch}"
validation="${run_root}/validation-epoch-${epoch}.json"
output_dir="${run_root}/test-hf-closed-loop-s86-s100-epoch${epoch}"
watch_log="${run_root}/test-after-validation.log"

exec > >(tee -a "${watch_log}") 2>&1
echo "[$(date -Is)] Waiting for ${validation}"

missing_process_checks=0
while [[ ! -f "${validation}" ]]; do
  if pgrep -f "memory_training.train.*--run-id ${run_id}" >/dev/null; then
    missing_process_checks=0
  else
    missing_process_checks=$((missing_process_checks + 1))
    if (( missing_process_checks >= 4 )); then
      echo "[$(date -Is)] Training/validation exited without ${validation}; test not started."
      exit 1
    fi
  fi
  sleep 30
done

echo "[$(date -Is)] Final validation artifact found; waiting for trainer shutdown."
while pgrep -f "memory_training.train.*--run-id ${run_id}" >/dev/null; do
  sleep 10
done

if [[ ! -d "${checkpoint}" ]]; then
  echo "[$(date -Is)] Missing checkpoint: ${checkpoint}"
  exit 1
fi

mkdir -p "${output_dir}"
echo "[$(date -Is)] Starting S86-S100 closed-loop test on GPU ${gpu}."
cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

exec "${python_bin}" -m memory_training.evaluate_hf_closed_loop \
  --model qwen3.5-4b \
  --method "${method}" \
  --checkpoint "${checkpoint}" \
  --output-dir "${output_dir}" \
  --scenarios 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 \
  --max-length 2048 \
  --max-new-tokens 768 \
  --quiz-max-new-tokens 256 \
  --scenario-batch-size 8 \
  --quiz-batch-size 16
