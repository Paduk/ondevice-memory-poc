#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/human-authored-vehicle-memory/hvp01-hvp20-external-adapted-eval-v1
output_root="${workspace}/evaluations/hvp01-hvp20-external-adapted-v1/granite4-1b"
log_root="${output_root}/logs"
poll_seconds="${POLL_SECONDS:-180}"
stability_seconds="${GPU_STABILITY_SECONDS:-300}"

summary_checkpoint="${workspace}/runs/granite4-1b-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-03"
patch_checkpoint="${workspace}/runs/granite4-1b-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-03"
delta_checkpoint="${workspace}/runs/granite4-1b-delta-v3-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-02"
scenarios=(901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920)
# GPU 2 is eligible again after an extended idle interval. The persistent CUDA
# reservation below protects transitions between evaluator processes.
candidate_gpus=(0 1 2 3 4 5 6)

mkdir -p "$log_root"
exec 9>"${output_root}/scheduler.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] another HVP20 scheduler is already running" >&2
  exit 3
fi

for checkpoint in "$summary_checkpoint" "$patch_checkpoint" "$delta_checkpoint"; do
  if [[ ! -d "${checkpoint}/adapter" ]]; then
    echo "[$(date -Is)] checkpoint adapter not found: ${checkpoint}" >&2
    exit 2
  fi
done

for required in catalog.sqlite summary.jsonl patch.jsonl delta.jsonl turn_quiz.jsonl final_quiz.jsonl quiz_sft.jsonl vehicle_tools.json; do
  if [[ ! -s "${data_root}/${required}" ]]; then
    echo "[$(date -Is)] evaluation input not found: ${data_root}/${required}" >&2
    exit 2
  fi
done

gpu_is_empty() {
  local gpu=$1
  local memory_used pids
  memory_used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
  [[ -z "$pids" && "$memory_used" =~ ^[0-9]+$ && "$memory_used" -le 100 ]]
}

choose_empty_gpu() {
  local gpu
  for gpu in "${candidate_gpus[@]}"; do
    if gpu_is_empty "$gpu"; then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

echo "[$(date -Is)] waiting for a strictly empty GPU; polling every ${poll_seconds}s"
selected_gpu=""
while [[ -z "$selected_gpu" ]]; do
  selected_gpu=$(choose_empty_gpu || true)
  if [[ -z "$selected_gpu" ]]; then
    echo "[$(date -Is)] no empty GPU"
    sleep "$poll_seconds"
  fi
done

# Require a stability window to avoid racing another user's scheduler.
echo "[$(date -Is)] candidate GPU ${selected_gpu}; verifying it stays empty for ${stability_seconds}s"
sleep "$stability_seconds"
if ! gpu_is_empty "$selected_gpu"; then
  echo "[$(date -Is)] GPU ${selected_gpu} was claimed during stability check; resuming wait"
  exec "$0"
fi

echo "[$(date -Is)] acquired GPU ${selected_gpu}; starting Granite 4 1B HVP01-HVP20 evaluation"
cd "$repo_root"
export CUDA_VISIBLE_DEVICES="$selected_gpu"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

# Keep a small CUDA context alive across evaluator process transitions. Without
# this reservation, another scheduler can mistake the unload/reload gap between
# Summary, Patch, and Delta-v3 for an idle GPU.
reservation_pid=""
cleanup_reservation() {
  if [[ -n "$reservation_pid" ]] && kill -0 "$reservation_pid" 2>/dev/null; then
    kill "$reservation_pid" 2>/dev/null || true
    wait "$reservation_pid" 2>/dev/null || true
  fi
}
trap cleanup_reservation EXIT
trap 'exit 130' INT TERM
trap 'exit 129' HUP

"$python_bin" -c \
  'import time; import torch; torch.empty(1, device="cuda"); print("GPU reservation active", flush=True); time.sleep(86400)' \
  >>"${log_root}/reservation.log" 2>&1 &
reservation_pid=$!
reservation_ready=false
for _ in $(seq 1 30); do
  if nvidia-smi -i "$selected_gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | rg -x "$reservation_pid" >/dev/null; then
    reservation_ready=true
    break
  fi
  if ! kill -0 "$reservation_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [[ "$reservation_ready" != true ]]; then
  echo "[$(date -Is)] failed to establish GPU reservation on ${selected_gpu}" >&2
  exit 4
fi
echo "[$(date -Is)] reservation PID ${reservation_pid} active on GPU ${selected_gpu}"

run_method() {
  local method=$1
  local checkpoint=$2
  local method_output="${output_root}/${method}"
  local method_log="${log_root}/${method}.log"

  if [[ -s "${method_output}/summary.json" ]] && [[ "$(jq -r '.complete' "${method_output}/summary.json")" == true ]]; then
    echo "[$(date -Is)] reuse completed method=${method}"
    return 0
  fi

  echo "[$(date -Is)] start method=${method} GPU=${selected_gpu}"
  "$python_bin" -m memory_training.evaluate_hf_closed_loop \
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
  echo "[$(date -Is)] done method=${method} GPU=${selected_gpu}"
}

run_method summary "$summary_checkpoint"
run_method patch "$patch_checkpoint"
run_method delta_v3 "$delta_checkpoint"

echo "[$(date -Is)] all methods completed: ${output_root}"
