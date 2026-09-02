#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/hj153lee/PalmClaw"
workspace="/mnt/data/hj153lee/PalmClaw/on-device-memory-training"
python_bin="/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python"
data_root="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2"
run_id="qwen35-2b-patch-multitask-noop5-grouped-v2-v1-10-e5-b4-r1"
checkpoint="${workspace}/runs/${run_id}/checkpoints/epoch-04"
pilot_root="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-style-augmentation-v1/pilot-s44-style-v2"
output_dir="${workspace}/runs/${run_id}/test-hf-v1-s44-vs-style-v2-s301-epoch04"
log_path="${output_dir}.log"

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python_bin}" -m memory_training.evaluate_hf_v1 \
  --model qwen3.5-2b \
  --method patch \
  --checkpoint "${checkpoint}" \
  --output-dir "${output_dir}" \
  --workspace "${workspace}" \
  --data-root "${data_root}" \
  --vehiclemembench-root /home/hj153lee/VehicleMemBench \
  --scenarios 44 301 \
  --scenario-variant "301:44:${pilot_root}/benchmark/history/history_301.txt:${pilot_root}/benchmark/qa_data/qa_301.json" \
  --max-length 2048 \
  --max-new-tokens 768 \
  --quiz-max-new-tokens 256 \
  --scenario-batch-size 2 \
  --quiz-batch-size 2 \
  --checkpoint-interval 50 \
  2>&1 | tee -a "${log_path}"
