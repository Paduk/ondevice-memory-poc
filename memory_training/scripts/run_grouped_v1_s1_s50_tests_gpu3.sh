#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/hj153lee/PalmClaw"
workspace="/mnt/data/hj153lee/PalmClaw/on-device-memory-training"
python_bin="/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python"
data_root="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2"
gpu=3
epoch=04
log_path="${workspace}/grouped-v1-s1-s50-tests-gpu3.log"

exec > >(tee -a "${log_path}") 2>&1
cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

for method in patch summary; do
  run_id="qwen35-4b-${method}-multitask-noop5-grouped-v2-v1-10-r1"
  run_root="${workspace}/runs/${run_id}"
  checkpoint="${run_root}/checkpoints/epoch-${epoch}"
  output_dir="${run_root}/test-hf-v1-s1-s50-epoch${epoch}"

  if [[ ! -d "${checkpoint}/adapter" ]]; then
    echo "[$(date -Is)] Missing checkpoint adapter: ${checkpoint}/adapter" >&2
    exit 1
  fi
  echo "[$(date -Is)] Starting ${method} V1 S1-S50 Test on GPU ${gpu}."
  "${python_bin}" -m memory_training.evaluate_hf_v1 \
    --model qwen3.5-4b \
    --method "${method}" \
    --checkpoint "${checkpoint}" \
    --output-dir "${output_dir}" \
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
  echo "[$(date -Is)] Completed ${method} V1 S1-S50 Test."
done

echo "[$(date -Is)] All grouped V1 S1-S50 Tests completed."
