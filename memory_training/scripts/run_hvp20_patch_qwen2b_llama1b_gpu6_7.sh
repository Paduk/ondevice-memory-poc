#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/human-authored-vehicle-memory/hvp01-hvp20-multiparty-quiz-expansion-v1-eval
evaluation_root="${workspace}/evaluations/hvp01-hvp20-multiparty-quiz-expansion-v1"
qwen_checkpoint="${workspace}/runs/qwen35-2b-patch-multitask-noop5-grouped-v2-v1-10-e5-b4-r1/checkpoints/epoch-03"
llama_checkpoint="${workspace}/runs/llama3.2-1b-patch-multitask-noop5-e4-b8-trainseed45-evalfixed-noop5-r2/checkpoints/epoch-03"
scenarios=(901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920)

scheduler_root="${evaluation_root}/patch-qwen2b-llama1b-scheduler"
mkdir -p "$scheduler_root"
exec 9>"${scheduler_root}/scheduler.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] another Qwen2B/Llama1B HVP Patch scheduler is running" >&2
  exit 3
fi

gpu_is_empty() {
  local gpu=$1
  local memory_used pids
  memory_used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
  [[ -z "$pids" && "$memory_used" =~ ^[0-9]+$ && "$memory_used" -le 100 ]]
}

for gpu in 6 7; do
  if ! gpu_is_empty "$gpu"; then
    echo "[$(date -Is)] GPU ${gpu} is no longer empty" >&2
    exit 4
  fi
done
for checkpoint in "$qwen_checkpoint" "$llama_checkpoint"; do
  [[ -d "${checkpoint}/adapter" ]] || {
    echo "[$(date -Is)] checkpoint adapter not found: ${checkpoint}" >&2
    exit 2
  }
done

cd "$repo_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

run_model() {
  local gpu=$1
  local model=$2
  local checkpoint=$3
  local output_dir=$4
  local log_path="${output_dir}/evaluation.log"
  mkdir -p "$output_dir"
  if [[ -s "${output_dir}/summary.json" ]] \
      && [[ "$(jq -r '.complete' "${output_dir}/summary.json")" == true ]]; then
    echo "[$(date -Is)] reuse completed model=${model}" | tee -a "${scheduler_root}/scheduler.log"
    return 0
  fi
  echo "[$(date -Is)] start model=${model} GPU=${gpu}" | tee -a "${scheduler_root}/scheduler.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m memory_training.evaluate_hf_closed_loop \
    --model "$model" \
    --method patch \
    --checkpoint "$checkpoint" \
    --output-dir "$output_dir" \
    --workspace "$workspace" \
    --data-root "$data_root" \
    --catalog-path "${data_root}/catalog.sqlite" \
    --vehicle-tools-path "${data_root}/vehicle_tools.json" \
    --quiz-sft-path "${data_root}/quiz_sft.jsonl" \
    --vehiclemembench-root /home/hj153lee/VehicleMemBench \
    --scenarios "${scenarios[@]}" \
    --max-length 4096 \
    --max-new-tokens 768 \
    --quiz-max-new-tokens 256 \
    --scenario-batch-size 5 \
    --quiz-batch-size 16 \
    >>"$log_path" 2>&1
  echo "[$(date -Is)] done model=${model} GPU=${gpu}" | tee -a "${scheduler_root}/scheduler.log"
}

qwen_output="${evaluation_root}/qwen3.5-2b/patch"
llama_output="${evaluation_root}/llama3.2-1b/patch"
run_model 6 qwen3.5-2b "$qwen_checkpoint" "$qwen_output" &
qwen_pid=$!
run_model 7 llama3.2-1b "$llama_checkpoint" "$llama_output" &
llama_pid=$!

failed=0
if ! wait "$qwen_pid"; then failed=1; fi
if ! wait "$llama_pid"; then failed=1; fi
if [[ "$failed" -ne 0 ]]; then
  echo "[$(date -Is)] at least one HVP Patch evaluation failed" | tee -a "${scheduler_root}/scheduler.log" >&2
  exit 1
fi
echo "[$(date -Is)] all Qwen2B/Llama1B HVP Patch evaluations completed" | tee -a "${scheduler_root}/scheduler.log"
