#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?GPU index required}
repo=/home/hj153lee/PalmClaw
runner=${repo}/memory_training/scripts/run_llama_mem0_one_pass_train_eval.sh
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
sequence_log=${workspace}/logs/llama32-mem0-one-pass-recovery.log

mkdir -p "$(dirname "${sequence_log}")"
exec > >(tee -a "${sequence_log}") 2>&1

echo "[$(date -Is)] Recovery sequence start on GPU ${gpu}: Llama 3B eval -> Llama 1B train/eval."
bash "${runner}" "${gpu}" llama3.2-3b
echo "[$(date -Is)] Llama 3B pipeline complete; starting Llama 1B."
bash "${runner}" "${gpu}" llama3.2-1b
echo "[$(date -Is)] Recovery sequence completed."
