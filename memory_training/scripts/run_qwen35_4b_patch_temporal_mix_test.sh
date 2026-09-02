#!/usr/bin/env bash
set -euo pipefail

variant="${1:?variant must be patch-r2 or temporal-patch-r1}"
gpu="${2:?GPU index is required}"
epoch="${3:-04}"

repo_root="/home/hj153lee/PalmClaw"
workspace="/mnt/data/hj153lee/PalmClaw/on-device-memory-training"
python_bin="/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python"
data_parent="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training"

case "${variant}" in
  patch-r2)
    method="patch"
    run_id="qwen35-4b-patch-multitask-noop5-st-t1t10-r2"
    data_root="${data_parent}/hybrid-s1-s100-plus-temporal-t1-t20-patch-t1t10-v2"
    ;;
  temporal-patch-r1)
    method="temporal_patch"
    run_id="qwen35-4b-temporal-patch-multitask-noop5-st-t1t10-r1"
    data_root="${data_parent}/hybrid-s1-s100-plus-temporal-t1-t20-temporal-patch-t1t10-terra-audited-v2"
    ;;
  *)
    echo "Unknown variant: ${variant}" >&2
    exit 2
    ;;
esac

run_root="${workspace}/runs/${run_id}"
checkpoint="${run_root}/checkpoints/epoch-${epoch}"
output_dir="${run_root}/test-hf-closed-loop-s86-s100-t12-t20-epoch${epoch}"

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python_bin}" -m memory_training.evaluate_hf_closed_loop \
  --model qwen3.5-4b \
  --method "${method}" \
  --checkpoint "${checkpoint}" \
  --output-dir "${output_dir}" \
  --workspace "${workspace}" \
  --data-root "${data_root}" \
  --catalog-path "${data_root}/catalog.sqlite" \
  --scenarios 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 112 113 114 115 116 117 118 119 120 \
  --max-length 2048 \
  --max-new-tokens 768 \
  --quiz-max-new-tokens 256 \
  --scenario-batch-size 8 \
  --quiz-batch-size 16
