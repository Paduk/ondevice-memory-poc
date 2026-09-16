#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/human-authored-vehicle-memory/hvp01-hvp20-multiparty-quiz-expansion-v1-eval
output_root="${workspace}/evaluations/hvp01-hvp20-multiparty-quiz-expansion-v1/granite4-1b"
log_root="${output_root}/logs"

summary_checkpoint="${workspace}/runs/granite4-1b-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-03"
patch_checkpoint="${workspace}/runs/granite4-1b-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-03"
delta_checkpoint="${workspace}/runs/granite4-1b-delta-v3-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-02"
scenarios=(901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920)

mkdir -p "$log_root"
exec 9>"${output_root}/scheduler.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] another multiparty HVP20 scheduler is running" >&2
  exit 3
fi

gpu_is_empty() {
  local gpu=$1
  local memory_used pids
  memory_used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
  [[ -z "$pids" && "$memory_used" =~ ^[0-9]+$ && "$memory_used" -le 100 ]]
}

for gpu in 2 4; do
  if ! gpu_is_empty "$gpu"; then
    echo "[$(date -Is)] required GPU ${gpu} is no longer empty" >&2
    exit 4
  fi
done

for checkpoint in "$summary_checkpoint" "$patch_checkpoint" "$delta_checkpoint"; do
  [[ -d "${checkpoint}/adapter" ]] || {
    echo "[$(date -Is)] checkpoint adapter not found: ${checkpoint}" >&2
    exit 2
  }
done

for required in catalog.sqlite summary.jsonl patch.jsonl delta.jsonl turn_quiz.jsonl final_quiz.jsonl quiz_sft.jsonl vehicle_tools.json; do
  [[ -s "${data_root}/${required}" ]] || {
    echo "[$(date -Is)] evaluation input not found: ${data_root}/${required}" >&2
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

run_method() {
  local gpu=$1
  local method=$2
  local checkpoint=$3
  local method_output="${output_root}/${method}"
  local method_log="${log_root}/${method}.log"

  if [[ -s "${method_output}/summary.json" ]] \
      && [[ "$(jq -r '.complete' "${method_output}/summary.json")" == true ]]; then
    echo "[$(date -Is)] reuse completed method=${method}"
    return 0
  fi

  echo "[$(date -Is)] start method=${method} GPU=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m memory_training.evaluate_hf_closed_loop \
    --model granite4-1b \
    --method "$method" \
    --checkpoint "$checkpoint" \
    --output-dir "$method_output" \
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
    >>"$method_log" 2>&1
  echo "[$(date -Is)] done method=${method} GPU=${gpu}"
}

# Start two methods immediately. Delta-v3 is then assigned to whichever GPU
# finishes first, so the slower first-stage method continues in parallel.
run_method 2 summary "$summary_checkpoint" &
summary_pid=$!
run_method 4 patch "$patch_checkpoint" &
patch_pid=$!

set +e
first_pid=""
wait -n -p first_pid "$summary_pid" "$patch_pid"
first_status=$?
set -e
if [[ $first_status -ne 0 ]]; then
  echo "[$(date -Is)] first method failed pid=${first_pid} status=${first_status}" >&2
  wait "$summary_pid" 2>/dev/null || true
  wait "$patch_pid" 2>/dev/null || true
  exit "$first_status"
fi

if [[ "$first_pid" == "$summary_pid" ]]; then
  free_gpu=2
  remaining_pid=$patch_pid
else
  free_gpu=4
  remaining_pid=$summary_pid
fi

run_method "$free_gpu" delta_v3 "$delta_checkpoint" &
delta_pid=$!

set +e
wait "$remaining_pid"
remaining_status=$?
wait "$delta_pid"
delta_status=$?
set -e
if [[ $remaining_status -ne 0 || $delta_status -ne 0 ]]; then
  echo "[$(date -Is)] evaluation failed remaining=${remaining_status} delta=${delta_status}" >&2
  exit 1
fi

echo "[$(date -Is)] all methods completed: ${output_root}"
