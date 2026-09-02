#!/usr/bin/env bash
set -euo pipefail

gpu=5
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
log_path="${workspace}/delta-v3-e5-gpu5-scheduler.log"
runner=/home/hj153lee/PalmClaw/memory_training/scripts/run_qwen35_4b_delta_v3.sh

exec > >(tee -a "${log_path}") 2>&1
echo "[$(date -Is)] Waiting for GPU ${gpu} to become available."
while true; do
  used=$(nvidia-smi --id="${gpu}" --query-compute-apps=used_memory \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk '{sum += $1} END {print sum + 0}')
  if (( used < 4096 )); then
    break
  fi
  echo "[$(date -Is)] GPU ${gpu} still uses ${used} MiB."
  sleep 30
done

echo "[$(date -Is)] GPU ${gpu} is available; starting Delta-v3 for 5 epochs."
exec bash "${runner}" "${gpu}" 5
