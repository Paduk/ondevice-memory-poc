#!/usr/bin/env bash
set -euo pipefail

repo=/home/hj153lee/PalmClaw
runner=${repo}/memory_training/scripts/run_llama_mem0_one_pass_train_eval.sh
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
sequence_log=${workspace}/logs/llama32-mem0-one-pass-sequence-gpu4.log

mkdir -p "$(dirname "${sequence_log}")"
exec > >(tee -a "${sequence_log}") 2>&1

echo "[$(date -Is)] Sequence start: Llama 3.2 3B -> Llama 3.2 1B on GPU 4."
bash "${runner}" 4 llama3.2-3b
echo "[$(date -Is)] Llama 3.2 3B pipeline succeeded; starting 1B."
bash "${runner}" 4 llama3.2-1b
echo "[$(date -Is)] Sequence completed."
