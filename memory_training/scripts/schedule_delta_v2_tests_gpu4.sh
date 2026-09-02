#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/hj153lee/PalmClaw"
workspace="/mnt/data/hj153lee/PalmClaw/on-device-memory-training"
python_bin="/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python"
data_root="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2"
v2_test_script="${repo_root}/memory_training/scripts/run_qwen35_4b_grouped_v1_test.sh"
run_id="qwen35-4b-delta-v2-multitask-noop5-grouped-v2-v1-10-r1"
run_root="${workspace}/runs/${run_id}"
status_path="${run_root}/status.json"
checkpoint="${run_root}/checkpoints/epoch-04"
v1_output="${run_root}/test-hf-v1-s1-s50-epoch04"
log_path="${workspace}/delta-v2-tests-gpu4-scheduler.log"
cancel_marker="${workspace}/cancel-delta-v2-tests-gpu4"
gpu=4

exec > >(tee -a "${log_path}") 2>&1
if [[ -f "${cancel_marker}" ]]; then
  echo "[$(date -Is)] Delta-v2 follow-up Tests cancelled by marker."
  exit 0
fi
echo "[$(date -Is)] Waiting for ${run_id} final Validation."

while true; do
  if [[ -f "${cancel_marker}" ]]; then
    echo "[$(date -Is)] Delta-v2 follow-up Tests cancelled by marker."
    exit 0
  fi
  state="MISSING"
  if [[ -f "${status_path}" ]]; then
    state="$(${python_bin} -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state", "UNKNOWN"))' "${status_path}")"
  fi
  case "${state}" in
    COMPLETED)
      while pgrep -f "memory_training.train.*--run-id ${run_id}" >/dev/null; do
        sleep 10
      done
      if [[ ! -d "${checkpoint}/adapter" ]]; then
        echo "[$(date -Is)] Missing checkpoint adapter: ${checkpoint}/adapter" >&2
        exit 1
      fi
      break
      ;;
    FAILED)
      echo "[$(date -Is)] Delta-v2 training failed; Tests will not start." >&2
      exit 1
      ;;
  esac
  sleep 30
done

if [[ -f "${cancel_marker}" ]]; then
  echo "[$(date -Is)] Delta-v2 follow-up Tests cancelled by marker."
  exit 0
fi

echo "[$(date -Is)] Starting Delta-v2 V2 Test on GPU ${gpu}."
"${v2_test_script}" delta_v2 "${gpu}" 04
echo "[$(date -Is)] Delta-v2 V2 Test completed. Starting V1 S1-S50 Test."

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

"${python_bin}" -m memory_training.evaluate_hf_v1 \
  --model qwen3.5-4b \
  --method delta_v2 \
  --checkpoint "${checkpoint}" \
  --output-dir "${v1_output}" \
  --workspace "${workspace}" \
  --data-root "${data_root}" \
  --vehiclemembench-root /home/hj153lee/VehicleMemBench \
  --scenarios {1..50} \
  --max-length 2048 \
  --max-new-tokens 768 \
  --quiz-max-new-tokens 256 \
  --scenario-batch-size 8 \
  --quiz-batch-size 16 \
  --checkpoint-interval 50

echo "[$(date -Is)] Delta-v2 V2 and V1 Tests completed."
