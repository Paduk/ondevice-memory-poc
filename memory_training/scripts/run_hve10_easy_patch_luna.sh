#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
hve_version="${HVE_VERSION:-v1}"
data_root="/mnt/data/hj153lee/PalmClaw/evaluation/human-authored-vehicle-memory/hve01-hve10-easy-explicit-${hve_version}-eval"
checkpoint="${workspace}/runs/granite4-1b-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-03"
output_root="${workspace}/evaluations/hve01-hve10-easy-explicit-${hve_version}"
local_output="${output_root}/granite4-1b/patch"
cloud_root="${output_root}/gpt-5.6-luna/patch"
log_root="${output_root}/logs"
scenarios=(921 922 923 924 925 926 927 928 929 930)
cloud_parallelism=4
quiz_workers=4
local_gpu="${LOCAL_GPU:-7}"

mkdir -p "$local_output" "$cloud_root" "$log_root"
exec 9>"${output_root}/scheduler.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] another HVE Easy evaluation is running" >&2
  exit 3
fi
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "OPENAI_API_KEY is required" >&2; exit 2; }
[[ -d "${checkpoint}/adapter" ]] || { echo "checkpoint missing: ${checkpoint}" >&2; exit 2; }

used_mib=$(nvidia-smi -i "$local_gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [[ ! "$used_mib" =~ ^[0-9]+$ ]] || (( used_mib > 10000 )); then
  echo "[$(date -Is)] GPU ${local_gpu} has ${used_mib} MiB allocated; refusing local launch" >&2
  exit 4
fi

cd "$repo_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

run_local() {
  if [[ -s "${local_output}/summary.json" ]] \
      && [[ "$(jq -r '.complete // false' "${local_output}/summary.json")" == true ]]; then
    echo "[$(date -Is)] reuse completed Granite Patch"
    return
  fi
  echo "[$(date -Is)] start Granite 4 1B Patch on GPU ${local_gpu}"
  CUDA_VISIBLE_DEVICES="$local_gpu" "$python_bin" -m memory_training.evaluate_hf_closed_loop \
    --model granite4-1b \
    --method patch \
    --checkpoint "$checkpoint" \
    --output-dir "$local_output" \
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
    >>"${log_root}/granite4-1b-patch.log" 2>&1
  echo "[$(date -Is)] completed Granite 4 1B Patch"
}

run_cloud_scenario() {
  local scenario=$1
  local scenario_output="${cloud_root}/s${scenario}"
  local scenario_log="${log_root}/luna-patch-s${scenario}.log"
  if [[ -s "${scenario_output}/summary.json" ]] \
      && [[ "$(jq -r '.complete // false' "${scenario_output}/summary.json")" == true ]]; then
    echo "[$(date -Is)] reuse Luna S${scenario}"
    return
  fi
  echo "[$(date -Is)] start Luna Patch S${scenario}"
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
    --quiz-max-output-tokens 256 \
    --quiz-workers "$quiz_workers" \
    --reasoning-effort low \
    >>"$scenario_log" 2>&1
  echo "[$(date -Is)] completed Luna Patch S${scenario}"
}

run_local &
local_pid=$!

cloud_failed=0
for ((offset=0; offset<${#scenarios[@]}; offset+=cloud_parallelism)); do
  pids=()
  batch=("${scenarios[@]:offset:cloud_parallelism}")
  for scenario in "${batch[@]}"; do
    run_cloud_scenario "$scenario" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then cloud_failed=1; fi
  done
  if (( cloud_failed != 0 )); then break; fi
done

local_failed=0
wait "$local_pid" || local_failed=1
if (( local_failed != 0 || cloud_failed != 0 )); then
  echo "[$(date -Is)] evaluation failed local=${local_failed} cloud=${cloud_failed}" >&2
  exit 1
fi
echo "[$(date -Is)] HVE Easy Granite Patch and Luna Patch completed"
