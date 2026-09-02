#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/hj153lee/PalmClaw"
main_workspace="/mnt/data/hj153lee/PalmClaw/on-device-memory-training"
workspace="/mnt/data/hj153lee/PalmClaw/on-device-memory-training-temporal-eval"
run_root="${main_workspace}/runs/qwen35-4b-patch-multitask-noop5-r1"
checkpoint="${run_root}/checkpoints/epoch-04"
data_root="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-temporal/patch-noop5-eval-t15-t16-t17-t19/data"
output_dir="${run_root}/test-hf-temporal-t15-t16-t17-t19-epoch04"
python_bin="/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python"

mkdir -p "${workspace}/cache" "${workspace}/tmp" "${output_dir}"
if [[ ! -e "${workspace}/cache/huggingface" ]]; then
  ln -s "${main_workspace}/cache/huggingface" "${workspace}/cache/huggingface"
fi
if [[ ! -e "${workspace}/cache/torch" ]]; then
  ln -s "${main_workspace}/cache/torch" "${workspace}/cache/torch"
fi

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python_bin}" -m memory_training.evaluate_hf_closed_loop \
  --model qwen3.5-4b \
  --method patch \
  --checkpoint "${checkpoint}" \
  --output-dir "${output_dir}" \
  --workspace "${workspace}" \
  --data-root "${data_root}" \
  --vehiclemembench-root /home/hj153lee/VehicleMemBench \
  --scenarios 15 16 17 19 \
  --max-length 2048 \
  --max-new-tokens 768 \
  --quiz-max-new-tokens 256 \
  --scenario-batch-size 4 \
  --quiz-batch-size 16
