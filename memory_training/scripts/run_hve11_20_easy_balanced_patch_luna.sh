#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/human-authored-vehicle-memory/hve01-hve20-easy-natural-balanced-v1-eval
checkpoint="${workspace}/runs/granite4-1b-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-03"
output_root="${workspace}/evaluations/hve11-hve20-easy-natural-balanced-v1"
local_output="${output_root}/granite4-1b/patch"
cloud_root="${output_root}/gpt-5.6-luna/patch"
log_root="${output_root}/logs"
scenarios=(931 932 933 934 935 936 937 938 939 940)
local_gpu="${LOCAL_GPU:-7}"

mkdir -p "$local_output" "$cloud_root" "$log_root"
exec 9>"${output_root}/scheduler.lock"
flock -n 9 || { echo "another balanced HVE evaluation is running" >&2; exit 3; }
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "OPENAI_API_KEY is required" >&2; exit 2; }
used_mib=$(nvidia-smi -i "$local_gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
(( used_mib <= 10000 )) || { echo "GPU ${local_gpu} has ${used_mib} MiB allocated" >&2; exit 4; }

cd "$repo_root"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

run_local() {
  echo "[$(date -Is)] start Granite balanced HVE11-HVE20 GPU=${local_gpu}"
  CUDA_VISIBLE_DEVICES="$local_gpu" "$python_bin" -m memory_training.evaluate_hf_closed_loop \
    --model granite4-1b --method patch --checkpoint "$checkpoint" \
    --output-dir "$local_output" --workspace "$workspace" --data-root "$data_root" \
    --catalog-path "${data_root}/catalog.sqlite" \
    --vehicle-tools-path "${data_root}/vehicle_tools.json" \
    --quiz-sft-path "${data_root}/quiz_sft.jsonl" \
    --vehiclemembench-root /home/hj153lee/VehicleMemBench \
    --scenarios "${scenarios[@]}" --max-length 4096 --max-new-tokens 768 \
    --quiz-max-new-tokens 256 --scenario-batch-size 5 --quiz-batch-size 16 \
    >>"${log_root}/granite4-1b-patch.log" 2>&1
  echo "[$(date -Is)] completed Granite balanced HVE11-HVE20"
}

run_cloud() {
  local scenario=$1
  echo "[$(date -Is)] start Luna balanced S${scenario}"
  "$python_bin" -m memory_training.evaluate_cloud_closed_loop \
    --method patch --scenario "$scenario" --model gpt-5.6-luna \
    --output-dir "${cloud_root}/s${scenario}" --data-root "$data_root" \
    --workspace "$workspace" --catalog-path "${data_root}/catalog.sqlite" \
    --vehicle-tools-path "${data_root}/vehicle_tools.json" \
    --quiz-sft-path "${data_root}/quiz_sft.jsonl" \
    --vehiclemembench-root /home/hj153lee/VehicleMemBench \
    --memory-max-output-tokens 768 --quiz-max-output-tokens 256 \
    --quiz-workers 4 --reasoning-effort low \
    >>"${log_root}/luna-patch-s${scenario}.log" 2>&1
  echo "[$(date -Is)] completed Luna balanced S${scenario}"
}

run_local &
local_pid=$!
cloud_failed=0
for ((offset=0; offset<10; offset+=4)); do
  pids=()
  for scenario in "${scenarios[@]:offset:4}"; do run_cloud "$scenario" & pids+=("$!"); done
  for pid in "${pids[@]}"; do wait "$pid" || cloud_failed=1; done
  (( cloud_failed == 0 )) || break
done
local_failed=0
wait "$local_pid" || local_failed=1
(( local_failed == 0 && cloud_failed == 0 )) || exit 1
echo "[$(date -Is)] balanced HVE11-HVE20 evaluation completed"
