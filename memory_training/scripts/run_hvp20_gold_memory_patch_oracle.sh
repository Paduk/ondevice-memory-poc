#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-2}
repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/human-authored-vehicle-memory/hvp01-hvp20-multiparty-quiz-expansion-v1-eval
checkpoint="${workspace}/runs/granite4-1b-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-03"
output_dir="${workspace}/evaluations/hvp01-hvp20-multiparty-quiz-expansion-v1/granite4-1b/gold-memory-patch-reader"
log_path="${output_dir}/evaluation.log"
scenarios=(901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920)

mkdir -p "$output_dir"
exec 9>"${output_dir}/scheduler.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] another Gold-memory oracle evaluation is running" >&2
  exit 3
fi

memory_used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
if [[ -n "$pids" || ! "$memory_used" =~ ^[0-9]+$ || "$memory_used" -gt 100 ]]; then
  echo "[$(date -Is)] GPU ${gpu} is not empty (memory=${memory_used} MiB, pids=${pids:-none})" >&2
  exit 4
fi

[[ -d "${checkpoint}/adapter" ]] || {
  echo "[$(date -Is)] checkpoint adapter not found: ${checkpoint}" >&2
  exit 2
}

cd "$repo_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$(date -Is)] start gold-memory Patch reader GPU=${gpu}" | tee -a "$log_path"
CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m memory_training.evaluate_quiz_baseline \
  --model granite4-1b \
  --checkpoint "$checkpoint" \
  --profile gold_memory \
  --data-root "$data_root" \
  --output-dir "$output_dir" \
  --workspace "$workspace" \
  --vehiclemembench-root /home/hj153lee/VehicleMemBench \
  --scenarios "${scenarios[@]}" \
  --max-length 4096 \
  --max-new-tokens 256 \
  --quiz-batch-size 16 \
  >>"$log_path" 2>&1
echo "[$(date -Is)] completed gold-memory Patch reader GPU=${gpu}" | tee -a "$log_path"
