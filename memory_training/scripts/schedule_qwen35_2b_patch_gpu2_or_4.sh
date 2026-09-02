#!/usr/bin/env bash
set -euo pipefail

workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
runner=/home/hj153lee/PalmClaw/memory_training/scripts/run_qwen35_2b_patch.sh
log_path="${workspace}/qwen35-2b-patch-gpu2-or-4-scheduler.log"
idle_required=600
poll_seconds=30
idle_since_2=0
idle_since_4=0

exec > >(tee -a "${log_path}") 2>&1
echo "[$(date -Is)] Waiting for GPU 2 or 4 to remain idle for 10 minutes."

while true; do
  now=$(date +%s)
  for gpu in 2 4; do
    used=$(nvidia-smi --id="${gpu}" --query-compute-apps=used_memory \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk '{sum += $1} END {print sum + 0}')
    variable="idle_since_${gpu}"
    idle_since="${!variable}"
    if (( used == 0 )); then
      if (( idle_since == 0 )); then
        printf -v "${variable}" '%s' "${now}"
        idle_since="${now}"
        echo "[$(date -Is)] GPU ${gpu} became idle; starting 10-minute timer."
      fi
      if (( now - idle_since >= idle_required )); then
        echo "[$(date -Is)] GPU ${gpu} stayed idle for 10 minutes; starting Qwen3.5-2B Patch E4."
        exec bash "${runner}" "${gpu}" 4
      fi
    else
      if (( idle_since != 0 )); then
        echo "[$(date -Is)] GPU ${gpu} was reused; resetting idle timer."
      fi
      printf -v "${variable}" '%s' 0
    fi
  done
  sleep "${poll_seconds}"
done
