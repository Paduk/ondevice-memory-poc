#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-v2-v1-10-eval-fixed-noop5-seed45-v1
checkpoint="${workspace}/runs/granite4-1b-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-03"
output_root="${workspace}/evaluations/fixed-v2-test-24/gpt-5.6-luna-patch-writer-granite4-1b-reader"
memory_root="${output_root}/cloud-memory"
reader_output="${output_root}/hf-reader"
log_root="${output_root}/logs"
scenarios=(86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 112 113 114 115 116 117 118 119 120)
cloud_parallelism=4

mkdir -p "$memory_root" "$reader_output" "$log_root"
exec 9>"${output_root}/scheduler.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] another fixed-24 Luna Patch evaluation is running" >&2
  exit 3
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[$(date -Is)] OPENAI_API_KEY is required" >&2
  exit 2
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

run_cloud_scenario() {
  local scenario=$1 scenario_tag scenario_output scenario_log
  printf -v scenario_tag 's%03d' "$scenario"
  scenario_output="${memory_root}/${scenario_tag}"
  scenario_log="${log_root}/cloud-${scenario_tag}.log"
  if [[ -s "${scenario_output}/summary.json" ]] \
      && [[ "$(jq -r '.complete' "${scenario_output}/summary.json")" == true ]]; then
    echo "[$(date -Is)] reuse Cloud memory S${scenario}" | tee -a "$scenario_log"
    return 0
  fi
  echo "[$(date -Is)] start Cloud memory S${scenario}" | tee -a "$scenario_log"
  "$python_bin" -m memory_training.evaluate_cloud_closed_loop \
    --method patch \
    --scenario "$scenario" \
    --model gpt-5.6-luna \
    --output-dir "$scenario_output" \
    --data-root "$data_root" \
    --workspace "$workspace" \
    --catalog-path "${data_root}/catalog.sqlite" \
    --vehicle-tools-path "${data_root}/vehicle_tools.json" \
    --quiz-sft-path "${data_root}/quiz_sft.jsonl" \
    --vehiclemembench-root /home/hj153lee/VehicleMemBench \
    --memory-max-output-tokens 768 \
    --reasoning-effort low \
    --skip-quiz \
    >>"$scenario_log" 2>&1
  echo "[$(date -Is)] completed Cloud memory S${scenario}" | tee -a "$scenario_log"
}

echo "[$(date -Is)] start fixed-24 Luna Patch memory generation parallelism=${cloud_parallelism}" | tee -a "${output_root}/scheduler.log"
for ((offset=0; offset<${#scenarios[@]}; offset+=cloud_parallelism)); do
  pids=()
  batch=("${scenarios[@]:offset:cloud_parallelism}")
  for scenario in "${batch[@]}"; do
    run_cloud_scenario "$scenario" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "[$(date -Is)] Cloud memory batch failed: ${batch[*]}" | tee -a "${output_root}/scheduler.log" >&2
    exit 1
  fi
done
echo "[$(date -Is)] all fixed-24 Luna Patch memories completed" | tee -a "${output_root}/scheduler.log"

gpu_is_empty() {
  local gpu=$1 memory_used pids
  memory_used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
  [[ -z "$pids" && "$memory_used" =~ ^[0-9]+$ && "$memory_used" -le 100 ]]
}

reader_gpu=""
while [[ -z "$reader_gpu" ]]; do
  for gpu in 0 1 2 3 4 5 6 7; do
    if gpu_is_empty "$gpu"; then
      reader_gpu=$gpu
      break
    fi
  done
  if [[ -z "$reader_gpu" ]]; then
    echo "[$(date -Is)] no empty GPU for Granite reader; retry in 60s" | tee -a "${output_root}/scheduler.log"
    sleep 60
  fi
done

echo "[$(date -Is)] start Granite 1B reader GPU=${reader_gpu}" | tee -a "${output_root}/scheduler.log"
CUDA_VISIBLE_DEVICES="$reader_gpu" "$python_bin" -m memory_training.evaluate_quiz_baseline \
  --model granite4-1b \
  --checkpoint "$checkpoint" \
  --profile cloud_patch \
  --memory-steps-root "$memory_root" \
  --data-root "$data_root" \
  --output-dir "$reader_output" \
  --workspace "$workspace" \
  --vehiclemembench-root /home/hj153lee/VehicleMemBench \
  --scenarios "${scenarios[@]}" \
  --max-length 4096 \
  --max-new-tokens 256 \
  --quiz-batch-size 16 \
  >>"${log_root}/hf-reader.log" 2>&1
echo "[$(date -Is)] completed Granite 1B reader GPU=${reader_gpu}" | tee -a "${output_root}/scheduler.log"
echo "[$(date -Is)] all fixed-24 Luna Patch writer + Granite reader evaluation completed" | tee -a "${output_root}/scheduler.log"
