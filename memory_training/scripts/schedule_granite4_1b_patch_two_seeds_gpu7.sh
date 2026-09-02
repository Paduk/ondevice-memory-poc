#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
train_runner="${repo_root}/memory_training/scripts/run_granite4_1b_patch_train_first.sh"
validation_runner="${repo_root}/memory_training/scripts/run_checkpoint_6full_validation_pairwise.sh"
gpu=7
epochs=4

cd "${repo_root}"
export HF_HOME="${workspace}/cache/huggingface"
export HF_HUB_CACHE="${workspace}/cache/huggingface/hub"
export HF_XET_CACHE="${workspace}/cache/huggingface/xet"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$(date -Is)] Prefetching the shared Granite 4.0 1B checkpoint."
"${python_bin}" -c \
  'from huggingface_hub import snapshot_download; snapshot_download("ibm-granite/granite-4.0-1b", cache_dir="/mnt/data/hj153lee/PalmClaw/on-device-memory-training/cache/huggingface/hub")'

echo "[$(date -Is)] Waiting for GPU ${gpu} to become idle."
while true; do
  used_mib=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if (( used_mib < 2048 )); then
    break
  fi
  sleep 30
done

echo "[$(date -Is)] Running the Granite 4.0 1B backward canary on GPU ${gpu}."
CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m memory_training.model_canary \
  --model granite4-1b \
  --mode backward \
  --device cuda:0 \
  --workspace "${workspace}"

for training_seed in 45 46; do
  run_id="granite4-1b-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed${training_seed}-r1"
  echo "[$(date -Is)] Starting Seed ${training_seed} training on GPU ${gpu}."
  bash "${train_runner}" "${gpu}" "${epochs}" "${training_seed}" r1
  echo "[$(date -Is)] Starting Seed ${training_seed} Epoch 1-4 Validation."
  bash "${validation_runner}" "${gpu}" "${run_id}" 1 4
done

echo "[$(date -Is)] Completed both Granite 4.0 1B Patch runs."
